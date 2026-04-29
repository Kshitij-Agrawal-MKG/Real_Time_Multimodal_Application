"""
main.py -- Voice Assistant entry point (Windows 11).

Runs all pipeline stages as concurrent asyncio tasks.
Also starts the dashboard WebSocket server and metrics HTTP server
inline so a single  python main.py  launches everything.

Usage:
    python main.py                    live microphone
    python main.py --no-vad           skip silence filter
    python main.py --record           save session to sessions\
    python main.py --replay NAME      replay a saved session
    python main.py --replay-speed 0   replay at max speed (benchmark)
    python main.py --reset-memory     clear conversation history
    python main.py --list-sessions    list saved sessions and exit
"""

import asyncio
import argparse
import signal
import sys
import logging
from pathlib import Path

# Windows: SelectorEventLoop required for WebSockets + asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load .env before any environment variable is read
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import Config
from mic import MicrophoneStream
from vad import VoiceActivityDetector
from asr import VoskASR, UTTERANCE_END
from llm import GeminiLLM
from tts import PollyTTS
from latency import LatencyTracker
from memory import ConversationMemory
from resilience import ResilienceWrapper
from replay import SessionRecorder, SessionReplayer, list_sessions
from hot_config import HotConfig
from metrics import MetricsCollector
from logger import setup_logger

log = setup_logger("main")


async def run(cfg: Config, args: argparse.Namespace):

    # ── Cross-cutting services ─────────────────────────────────────────────────
    tracker   = LatencyTracker()
    memory    = ConversationMemory(
        max_turns    = 20,
        token_budget = 4000,
        persist_path = Path("memory.json") if args.persist_memory else None,
    )
    if args.reset_memory:
        memory.clear()

    hot_cfg   = HotConfig("config_overrides.json")
    metrics   = MetricsCollector()
    resilience = ResilienceWrapper(cfg)

    # ── Pipeline stages ────────────────────────────────────────────────────────
    asr = VoskASR(cfg, tracker)
    llm = GeminiLLM(cfg, tracker, memory=memory)
    tts = PollyTTS(cfg, tracker)
    vad = VoiceActivityDetector(cfg) if not args.no_vad else None

    # ── Queues ─────────────────────────────────────────────────────────────────
    audio_q:      asyncio.Queue = asyncio.Queue(maxsize=500)
    vad_q:        asyncio.Queue = asyncio.Queue(maxsize=500)
    transcript_q: asyncio.Queue = asyncio.Queue(maxsize=50)
    token_q:      asyncio.Queue = asyncio.Queue(maxsize=200)

    recorder = SessionRecorder(cfg) if args.record else None

    # Pre-warm Vosk model so first utterance is not slow
    log.info("[ASR] pre-loading Vosk model...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, asr._load_model)

    # Pre-warm Polly connection so first TTS is not slow
    log.info("[TTS] pre-warming Polly connection...")
    try:
        await loop.run_in_executor(None, tts._get_polly)
        log.info("[TTS] Polly ready")
    except Exception as e:
        log.warning(f"[TTS] Polly pre-warm failed: {e}")

    # ── Dashboard WebSocket (inline, same process) ─────────────────────────────
    import json
    from collections import deque
    _dash_clients: set = set()
    _dash_history: deque = deque(maxlen=100)

    async def _dash_push(record: dict):
        """Broadcast a latency record to all connected dashboard clients."""
        _dash_history.append(record)
        if not _dash_clients:
            return
        msg = json.dumps({"type": "utterance", "data": record,
                          "history": list(_dash_history)})
        await asyncio.gather(
            *[c.send(msg) for c in _dash_clients],
            return_exceptions=True
        )

    async def _dash_handler(ws):
        _dash_clients.add(ws)
        log.info(f"Dashboard connected: {ws.remote_address}")
        await ws.send(json.dumps({"type": "history",
                                  "history": list(_dash_history)}))
        try:
            async for _ in ws:
                pass
        finally:
            _dash_clients.discard(ws)

    import websockets
    dash_server = await websockets.serve(
        _dash_handler, "localhost", cfg.dashboard_port
    )
    log.info(f"Dashboard WebSocket on ws://localhost:{cfg.dashboard_port}")
    log.info(f"Open dashboard.html in your browser to view live metrics")

    # ── Metrics HTTP server ────────────────────────────────────────────────────
    asyncio.create_task(metrics.serve(cfg.metrics_port), name="metrics")

    # ── Hot config watcher ─────────────────────────────────────────────────────
    asyncio.create_task(hot_cfg.watch(), name="hot_config")

    # ── Stage coroutines ───────────────────────────────────────────────────────

    async def audio_source():
        if args.replay:
            replayer = SessionReplayer(cfg, args.replay, speed=args.replay_speed)
            async for chunk in replayer.stream():
                await audio_q.put(chunk)
        else:
            mic = MicrophoneStream(cfg)
            async for chunk in mic.stream():
                if recorder:
                    recorder.record(chunk)
                await audio_q.put(chunk)

    async def vad_stage():
        if vad:
            async for chunk in vad.filter(audio_q):
                if chunk is None:
                    # Utterance boundary -- tell Vosk to flush its buffer
                    await vad_q.put(UTTERANCE_END)
                else:
                    await vad_q.put(chunk)
        else:
            # No VAD: pass all audio through; Vosk handles its own segmentation
            while True:
                await vad_q.put(await audio_q.get())

    async def asr_stage():
        """Run Vosk continuously. Restart on error; propagate CancelledError."""
        while True:
            try:
                async for text in asr.stream(vad_q):
                    if recorder:
                        recorder.add_transcript(text)
                    log.info(f"[ASR] {text!r}")
                    await transcript_q.put(text)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error(f"[ASR] error: {exc!r} -- restarting in 1s")
                await asyncio.sleep(1.0)

    async def _call_llm(text: str):
        async for token in llm.stream(text):
            await token_q.put(token)
        await token_q.put(None)

    async def llm_stage():
        while True:
            text = await transcript_q.get()
            if text is None:
                break
            try:
                await asyncio.wait_for(_call_llm(text), timeout=cfg.llm_timeout_s)
            except asyncio.TimeoutError:
                log.warning(f"[LLM] timeout after {cfg.llm_timeout_s}s")
                await token_q.put("Sorry, that took too long. Please try again.")
                await token_q.put(None)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error(f"[LLM] error: {exc!r}")
                await token_q.put(None)

    async def tts_stage():
        buf = []
        while True:
            token = await token_q.get()
            if token is None:
                if buf:
                    text = "".join(buf)
                    log.info(f"[TTS] {text[:70]!r}{'...' if len(text) > 70 else ''}")
                    await resilience.run_with_timeout(
                        "TTS",
                        lambda t=text: tts.speak(t),
                        cfg.tts_timeout_s,
                        on_timeout=lambda: tts.speak_cached("processing"),
                    )
                    tracker.mark_audio_playing()
                    tracker.log_summary()

                    # Publish to dashboard and metrics
                    record = tracker.latest_dict()
                    await _dash_push(record)
                    metrics.record_utterance(record)
                    log.info(f"[Memory] {memory.summary_str()}")
                    buf.clear()
            else:
                buf.append(token)

    # ── Startup banner ─────────────────────────────────────────────────────────
    log.info("=" * 56)
    log.info("  Voice Assistant  --  ready")
    log.info(f"  VAD      : {'on' if vad else 'off'}")
    log.info(f"  Source   : {'replay: ' + args.replay if args.replay else 'microphone'}")
    log.info(f"  Memory   : {memory.summary_str()}")
    log.info(f"  Dashboard: open dashboard.html in browser")
    log.info(f"  Metrics  : http://localhost:{cfg.metrics_port}/summary")
    log.info("=" * 56)

    tasks = [
        asyncio.create_task(audio_source(), name="audio"),
        asyncio.create_task(vad_stage(),    name="vad"),
        asyncio.create_task(asr_stage(),    name="asr"),
        asyncio.create_task(llm_stage(),    name="llm"),
        asyncio.create_task(tts_stage(),    name="tts"),
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        log.info("Shutting down...")
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        dash_server.close()
        if recorder and recorder._chunks:
            log.info(f"Session saved -> {recorder.save()}")
        tracker.export_json(Path("latency_log.json"))
        log.info("Latency log -> latency_log.json")
        if vad:
            s = vad.stats()
            log.info(f"VAD: {s['utterances']} utterances, {s['speech_ratio_pct']}% speech")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time voice assistant")
    p.add_argument("--replay",         type=str,   default=None,  metavar="NAME")
    p.add_argument("--replay-speed",   type=float, default=1.0,   metavar="N")
    p.add_argument("--record",         action="store_true")
    p.add_argument("--no-vad",         action="store_true")
    p.add_argument("--reset-memory",   action="store_true")
    p.add_argument("--persist-memory", action="store_true", default=True)
    p.add_argument("--list-sessions",  action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.list_sessions:
        sessions = list_sessions()
        if sessions:
            for s in sessions:
                print(f"  {s['name']}  {s['duration_s']}s  {s['recorded_at']}")
        else:
            print("  No sessions recorded yet.")
        return

    cfg  = Config.from_env()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _shutdown(*_):
        for t in asyncio.all_tasks(loop):
            t.cancel()

    signal.signal(signal.SIGINT, _shutdown)

    try:
        loop.run_until_complete(run(cfg, args))
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()

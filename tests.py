"""
tests.py — Full test suite for the Voice Assistant pipeline.

66 tests covering all three phases and all modules.
No API keys required -- all external calls are mocked.

Usage:
    python tests.py                 run all 66 tests
    python tests.py --phase 1       Phase 1: streaming pipeline
    python tests.py --phase 2       Phase 2: latency and memory
    python tests.py --phase 3       Phase 3: resilience and replay
    python tests.py -v              show error details on failure
"""

import argparse
import asyncio
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

# ── Minimal test runner ────────────────────────────────────────────────────────

GREEN = "\033[32m"; RED = "\033[31m"; YELLOW = "\033[33m"
CYAN  = "\033[36m"; RESET = "\033[0m"; BOLD = "\033[1m"

_registry: list[dict] = []


def test(name: str, phase: int = 0):
    def wrap(fn):
        _registry.append({"name": name, "phase": phase, "fn": fn,
                           "status": None, "error": None, "ms": None})
        return fn
    return wrap


async def _run_all(phase_filter: int | None, verbose: bool) -> bool:
    passed = failed = 0
    for r in _registry:
        if phase_filter and r["phase"] != phase_filter:
            continue
        t0 = time.perf_counter()
        try:
            await r["fn"]()
            r["status"] = "pass"; passed += 1
            icon = f"{GREEN}v{RESET}"
        except Exception as e:
            r["status"] = "fail"; r["error"] = f"{type(e).__name__}: {e}"
            failed += 1
            icon = f"{RED}x{RESET}"
        r["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        label = f"Phase {r['phase']}" if r["phase"] else "     "
        print(f"  {icon}  [{label}]  {r['name']}  {CYAN}{r['ms']}ms{RESET}"
              + (f"\n         {RED}{r['error']}{RESET}" if r["error"] and verbose else ""))

    total  = passed + failed
    colour = GREEN if not failed else RED
    print(f"\n{colour}{BOLD}{passed}/{total} passed{RESET}"
          + (f"  {RED}{failed} failed{RESET}" if failed else ""))
    return failed == 0


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Streaming pipeline
# ══════════════════════════════════════════════════════════════════════════════

@test("Config: loads credentials from environment", phase=1)
async def _():
    import os; from config import Config
    os.environ["AWS_ACCESS_KEY_ID"]     = "aws_test"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "aws_secret_test"
    os.environ["GEMINI_API_KEY"]        = "gm_test"
    cfg = Config.from_env()
    assert cfg.aws_access_key == "aws_test"
    assert cfg.gemini_api_key == "gm_test"
    assert cfg.sample_rate    == 16_000


@test("Config: chunk_frames is computed correctly", phase=1)
async def _():
    from config import Config
    assert Config(sample_rate=16_000, chunk_ms=20).chunk_frames == 320
    assert Config(sample_rate=16_000, chunk_ms=30).chunk_frames == 480


@test("Config: default values are sane", phase=1)
async def _():
    from config import Config
    cfg = Config()
    assert cfg.polly_engine    == "neural"
    assert cfg.polly_format    == "pcm"
    assert cfg.max_retries     == 3
    assert cfg.llm_timeout_s   >= 5.0   # may be tuned; just check it's set


@test("ASR: VoskASR raises FileNotFoundError for missing model path", phase=1)
async def _():
    from config import Config; from asr import VoskASR; from latency import LatencyTracker
    asr = VoskASR(Config(vosk_model_path="/nonexistent/vosk_model"), LatencyTracker())
    try:
        asr._load_model()
        assert False, "expected an error"
    except FileNotFoundError as e:
        # vosk installed but model missing -- correct path
        assert "vosk_model" in str(e).lower() or "nonexistent" in str(e)
    except ImportError:
        # vosk not installed in this environment -- acceptable in CI/sandbox
        pass


@test("ASR: VoskASR error message contains download URL", phase=1)
async def _():
    from config import Config; from asr import VoskASR; from latency import LatencyTracker
    asr = VoskASR(Config(vosk_model_path="/no/such/path"), LatencyTracker())
    try:
        asr._load_model()
    except FileNotFoundError as e:
        assert "alphacephei.com" in str(e)
    except ImportError:
        pass  # vosk not installed -- skip content check


@test("Config: vosk_model_path defaults to vosk_model/ directory", phase=1)
async def _():
    from config import Config
    assert "vosk_model" in Config().vosk_model_path


@test("Config: VOSK_MODEL_PATH env var overrides default", phase=1)
async def _():
    import os; from config import Config
    os.environ["VOSK_MODEL_PATH"] = "/custom/model/path"
    cfg = Config.from_env()
    assert cfg.vosk_model_path == "/custom/model/path"
    del os.environ["VOSK_MODEL_PATH"]


@test("Queue: producer/consumer ordering is preserved", phase=1)
async def _():
    q: asyncio.Queue = asyncio.Queue()
    for i in range(5): await q.put(i)
    received = [await q.get() for _ in range(5)]
    assert received == list(range(5))


@test("Queue: bounded queue blocks producer when full", phase=1)
async def _():
    q: asyncio.Queue = asyncio.Queue(maxsize=3)
    for i in range(3): q.put_nowait(i)
    assert q.full()


@test("Queue: non-blocking pipeline -- fast producer, slow consumer", phase=1)
async def _():
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    produced = []; consumed = []
    async def produce():
        for i in range(20): await q.put(i); produced.append(i); await asyncio.sleep(0)
    async def consume():
        for _ in range(20): consumed.append(await q.get()); await asyncio.sleep(0.002)
    await asyncio.gather(produce(), consume())
    assert produced == consumed == list(range(20))


@test("VAD: silent frame has zero RMS", phase=1)
async def _():
    from vad import _rms
    assert _rms(b"\x00" * 960) == 0.0


@test("VAD: loud frame has high RMS", phase=1)
async def _():
    import struct; from vad import _rms
    loud = struct.pack("<" + "h" * 480, *[32767] * 480)
    assert _rms(loud) == 32767.0


@test("VAD: speech start opens utterance after consecutive loud frames", phase=1)
async def _():
    import struct; from config import Config; from vad import VoiceActivityDetector, _SPEECH_START_FRAMES
    cfg  = Config()
    vad  = VoiceActivityDetector(cfg, threshold=1000)
    loud = struct.pack("<" + "h" * cfg.chunk_frames, *[32767] * cfg.chunk_frames)
    q: asyncio.Queue = asyncio.Queue()
    for _ in range(_SPEECH_START_FRAMES + 2): await q.put(loud)
    yielded = []
    async for chunk in vad.filter(q):
        yielded.append(chunk)
        if len(yielded) >= 2: break
    assert any(c is not None for c in yielded)


@test("VAD: silence sentinel yielded after speech ends", phase=1)
async def _():
    import struct
    from config import Config
    from vad import VoiceActivityDetector, _SPEECH_START_FRAMES, _SILENCE_END_FRAMES
    cfg    = Config()
    vad    = VoiceActivityDetector(cfg, threshold=1000)
    loud   = struct.pack("<" + "h" * cfg.chunk_frames, *[32767] * cfg.chunk_frames)
    silent = b"\x00" * (cfg.chunk_frames * 2)
    q: asyncio.Queue = asyncio.Queue()
    for _ in range(_SPEECH_START_FRAMES + 2): await q.put(loud)
    for _ in range(_SILENCE_END_FRAMES  + 2): await q.put(silent)
    got_sentinel = False
    count = 0
    async for chunk in vad.filter(q):
        if chunk is None: got_sentinel = True; break
        count += 1
        if count > 50: break
    assert got_sentinel


@test("TTS: canned phrases dict covers all resilience keys", phase=1)
async def _():
    src = open("tts.py").read()
    for key in ("processing", "asr_error", "llm_error", "tts_error"):
        assert key in src, f"missing canned key: {key!r}"


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Latency tracking, memory, metrics, hot config, monitor
# ══════════════════════════════════════════════════════════════════════════════

@test("Latency: all stage timestamps are recorded", phase=2)
async def _():
    from latency import LatencyTracker
    t = LatencyTracker()
    t.mark_asr_start();  await asyncio.sleep(0.03)
    t.mark_asr_first();  await asyncio.sleep(0.05)
    t.mark_llm_first();  await asyncio.sleep(0.03)
    t.mark_tts_first();  t.mark_audio_playing()
    t.log_summary()
    r = t.all_dicts()[0]
    assert r["asr_latency_ms"] > 0
    assert r["llm_ttft_ms"]    > 0
    assert r["tts_ttfb_ms"]    > 0
    assert r["total_ms"]       > r["asr_latency_ms"]


@test("Latency: utterance IDs increment per log_summary", phase=2)
async def _():
    from latency import LatencyTracker
    t = LatencyTracker()
    for _ in range(3):
        t.mark_asr_start(); t.mark_asr_first()
        t.mark_llm_first(); t.mark_tts_first()
        t.log_summary()
    assert [r["utterance_id"] for r in t.all_dicts()] == [1, 2, 3]


@test("Latency: missing stages produce None values, not errors", phase=2)
async def _():
    from latency import UtteranceRecord
    r = UtteranceRecord(utterance_id=1, t_asr_start=1.0)
    r.compute()
    assert r.asr_latency_ms is None
    assert r.total_ms        is None


@test("Latency: export_json produces valid JSON list", phase=2)
async def _():
    from latency import LatencyTracker
    t = LatencyTracker()
    t.mark_asr_start(); t.mark_asr_first()
    t.mark_llm_first(); t.mark_tts_first()
    t.log_summary()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        p = Path(f.name)
    t.export_json(p)
    data = json.loads(p.read_text())
    assert isinstance(data, list) and "total_ms" in data[0]
    p.unlink(missing_ok=True)


@test("Memory: add_user / add_assistant / turn_count", phase=2)
async def _():
    from memory import ConversationMemory
    m = ConversationMemory()
    m.add_user("Hello"); m.add_assistant("Hi!")
    assert m.turn_count == 2


@test("Memory: max_turns cap enforced", phase=2)
async def _():
    from memory import ConversationMemory
    m = ConversationMemory(max_turns=4)
    for i in range(10): m.add_user(f"msg {i}")
    assert m.turn_count <= 4


@test("Memory: token budget trims old turns", phase=2)
async def _():
    from memory import ConversationMemory
    m = ConversationMemory(max_turns=100, token_budget=100)
    for _ in range(10): m.add_user("word " * 100)
    assert m.token_estimate < 300


@test("Memory: as_gemini_messages uses 'model' role for assistant", phase=2)
async def _():
    from memory import ConversationMemory
    m = ConversationMemory()
    m.add_user("Hello"); m.add_assistant("Hi!")
    roles = [msg["role"] for msg in m.as_gemini_messages()]
    assert "model" in roles and "user" in roles


@test("Memory: clear resets turns and summary", phase=2)
async def _():
    from memory import ConversationMemory
    m = ConversationMemory()
    m.add_user("test"); m.clear()
    assert m.turn_count == 0 and m.token_estimate == 0


@test("Memory: persist and reload round-trip", phase=2)
async def _():
    from memory import ConversationMemory
    with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as f:
        p = Path(f.name)
    m1 = ConversationMemory(persist_path=p)
    m1.add_user("remember"); m1.add_assistant("ok")
    assert p.exists()
    m2 = ConversationMemory(persist_path=p)
    assert m2.turn_count == 2
    p.unlink(missing_ok=True)


@test("Metrics: Counter increments and label filtering work", phase=2)
async def _():
    from metrics import Counter
    c = Counter("test", "help")
    c.inc(); c.inc(3); c.inc(stage="asr"); c.inc(stage="asr")
    assert c.get()           == 4
    assert c.get(stage="asr") == 2


@test("Metrics: Histogram observe and summary", phase=2)
async def _():
    from metrics import Histogram
    h = Histogram("x", "y")
    for v in [100, 200, 300, 500, 800, 1200]: h.observe(v)
    s = h.summary()
    assert s["count"] == 6 and "p95" in s and "mean" in s


@test("Metrics: Gauge set/inc/dec", phase=2)
async def _():
    from metrics import Gauge
    g = Gauge("g", "h")
    g.set(5); g.inc(2); g.dec(1)
    assert g._v == 6.0


@test("Metrics: record_utterance populates all histograms", phase=2)
async def _():
    from metrics import MetricsCollector
    mc = MetricsCollector()
    mc.record_utterance({"utterance_id": 1, "asr_latency_ms": 280.0,
                         "llm_ttft_ms": 420.0, "tts_ttfb_ms": 190.0, "total_ms": 930.0})
    assert mc.utterances.get()           == 1
    assert mc.asr_latency._obs           == [280.0]
    assert mc.total_lat._obs             == [930.0]


@test("Metrics: Prometheus output has expected metric names", phase=2)
async def _():
    from metrics import MetricsCollector
    mc = MetricsCollector()
    mc.inc_error("asr"); mc.inc_barge_in()
    out = mc.render_prometheus()
    for name in ("voice_pipeline_errors_total", "voice_pipeline_barge_ins_total",
                 "voice_pipeline_uptime_seconds"):
        assert name in out, f"missing: {name}"


@test("Metrics: summary_json is valid JSON with expected keys", phase=2)
async def _():
    from metrics import MetricsCollector
    data = json.loads(MetricsCollector().summary_json())
    for k in ("utterances", "asr_latency_ms", "total_latency_ms", "errors"):
        assert k in data


@test("HotConfig: get returns default when file absent", phase=2)
async def _():
    from hot_config import HotConfig
    hc = HotConfig("/nonexistent/path.json")
    assert hc.get("vad_threshold", default=300) == 300
    assert hc.get("missing")                    is None


@test("HotConfig: set persists to disk with type coercion", phase=2)
async def _():
    from hot_config import HotConfig
    with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as f:
        p = Path(f.name)
    hc = HotConfig(p)
    hc.set("vad_threshold", "500", persist=True)
    assert p.exists()
    assert json.loads(p.read_text())["vad_threshold"] == 500.0
    p.unlink(missing_ok=True)


@test("HotConfig: type coercion for all schema types", phase=2)
async def _():
    from hot_config import HotConfig
    hc = HotConfig("/nonexistent.json")
    assert hc._coerce("vad_threshold",   "300.5") == 300.5
    assert hc._coerce("max_retries",     "5")     == 5
    assert hc._coerce("debug_logging",   "true")  is True
    assert hc._coerce("debug_logging",   "false") is False
    assert hc._coerce("polly_voice",     42)      == "42"


@test("HotConfig: on_change listener fires on set", phase=2)
async def _():
    from hot_config import HotConfig
    with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as f:
        p = Path(f.name)
    hc = HotConfig(p)
    changes = []
    hc.on_change("vad_threshold", lambda old, new: changes.append((old, new)))
    hc.set("vad_threshold", 600, persist=False)
    assert changes == [(None, 600.0)]
    p.unlink(missing_ok=True)


@test("HotConfig: immutable keys are silently blocked", phase=2)
async def _():
    from hot_config import HotConfig
    hc = HotConfig("/nonexistent.json")
    hc.set("sample_rate", 8000, persist=False)
    assert hc.get("sample_rate") is None


@test("HotConfig: delete removes override", phase=2)
async def _():
    from hot_config import HotConfig
    with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as f:
        p = Path(f.name)
    hc = HotConfig(p)
    hc.set("vad_threshold", 500, persist=False)
    hc.delete("vad_threshold")
    assert hc.get("vad_threshold") is None
    p.unlink(missing_ok=True)


@test("HotConfig: parse_cli_overrides parses key=value strings", phase=2)
async def _():
    from hot_config import parse_cli_overrides
    result = parse_cli_overrides(["vad_threshold=400", "polly_voice=Matthew"])
    assert result == {"vad_threshold": "400", "polly_voice": "Matthew"}


@test("HotConfig: file change triggers reload and listeners", phase=2)
async def _():
    from hot_config import HotConfig
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        p = Path(f.name)
        json.dump({"vad_threshold": 300}, f)
    try:
        hc = HotConfig(p, poll_interval_s=0.05)
        assert hc.get("vad_threshold") == 300.0
        changes = []
        hc.on_change("vad_threshold", lambda o, n: changes.append(n))
        p.write_text(json.dumps({"vad_threshold": 999}))
        hc._mtime = 0.0
        hc._reload_if_changed()
        assert hc.get("vad_threshold") == 999.0
        assert changes == [999.0]
    finally:
        p.unlink(missing_ok=True)


@test("Monitor: renders without error on empty state", phase=2)
async def _():
    from monitor import TerminalMonitor
    out = TerminalMonitor().render()
    assert "VOICE PIPELINE MONITOR" in out
    assert "Waiting" in out or "utterance" in out.lower()


@test("Monitor: renders last utterance correctly", phase=2)
async def _():
    from monitor import TerminalMonitor
    m = TerminalMonitor()
    m.ingest({"utterance_id": 7, "asr_latency_ms": 310.0,
              "llm_ttft_ms": 490.0, "tts_ttfb_ms": 210.0,
              "overhead_ms": 55.0,  "total_ms": 1065.0})
    assert "LAST UTTERANCE #7" in m.render()


@test("Monitor: log entries appear in render", phase=2)
async def _():
    from monitor import TerminalMonitor
    m = TerminalMonitor()
    m.add_log("ASR", "hello world")
    assert "hello world" in m.render()


@test("Monitor: state change is reflected", phase=2)
async def _():
    from monitor import TerminalMonitor
    m = TerminalMonitor()
    m.set_state("SPEAKING")
    assert "SPEAKING" in m.render()


@test("Monitor: rolling percentile is correct", phase=2)
async def _():
    from monitor import TerminalMonitor
    m = TerminalMonitor()
    for i, v in enumerate(range(10, 110, 10)):
        m.ingest({"utterance_id": i, "asr_latency_ms": float(v),
                  "llm_ttft_ms": float(v), "tts_ttfb_ms": float(v),
                  "overhead_ms": 5.0,      "total_ms": float(v * 3)})
    p50 = m._pct(m._roll("asr_latency_ms"), 50)
    assert p50 is not None and 40 <= p50 <= 70


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Resilience, wake word, interrupt, replay
# ══════════════════════════════════════════════════════════════════════════════

@test("Resilience: fast coroutine completes within timeout", phase=3)
async def _():
    from config import Config; from resilience import ResilienceWrapper
    called = []
    async def fast(): await asyncio.sleep(0.01); called.append(True)
    await ResilienceWrapper(Config()).run_with_timeout("t", fast, 1.0)
    assert called == [True]


@test("Resilience: slow coroutine triggers on_timeout callback", phase=3)
async def _():
    from config import Config; from resilience import ResilienceWrapper
    fired = []
    async def slow(): await asyncio.sleep(10)
    await ResilienceWrapper(Config()).run_with_timeout("t", slow, 0.05,
                                                       on_timeout=lambda: fired.append(True))
    assert fired == [True]


@test("Resilience: retry succeeds on second attempt", phase=3)
async def _():
    from config import Config; from resilience import ResilienceWrapper
    cfg = Config(max_retries=3, retry_backoff_base=0.01)
    attempts = []
    async def flaky():
        attempts.append(1)
        if len(attempts) < 2: raise ConnectionError("fail")
    await ResilienceWrapper(cfg).run_with_retry("t", flaky)
    assert len(attempts) == 2


@test("Resilience: exhausted retries push fallback to queue", phase=3)
async def _():
    from config import Config; from resilience import ResilienceWrapper
    cfg = Config(max_retries=2, retry_backoff_base=0.01)
    q: asyncio.Queue = asyncio.Queue()
    async def always_fail(): raise RuntimeError("broken")
    await ResilienceWrapper(cfg).run_with_retry("t", always_fail,
                                                fallback_queue=q, fallback="fallback")
    assert await q.get() == "fallback"


@test("Resilience: backoff delays increase between retries", phase=3)
async def _():
    from config import Config; from resilience import ResilienceWrapper
    cfg = Config(max_retries=3, retry_backoff_base=0.02)
    times = []
    async def fail(): times.append(time.monotonic()); raise RuntimeError("x")
    await ResilienceWrapper(cfg).run_with_retry("t", fail)
    gaps = [times[i+1] - times[i] for i in range(len(times)-1)]
    for i in range(1, len(gaps)):
        assert gaps[i] >= gaps[i-1] * 0.9


@test("WakeWord: starts in IDLE state", phase=3)
async def _():
    from wake_word import WakeWordDetector, WakeState
    from config import Config
    assert WakeWordDetector(Config()).state == WakeState.IDLE


@test("WakeWord: force_activate transitions to ACTIVE", phase=3)
async def _():
    from wake_word import WakeWordDetector, WakeState, WakeEvent
    from config import Config
    wwd = WakeWordDetector(Config())
    await wwd._activate(WakeEvent(wake_word="test"))
    assert wwd.state == WakeState.ACTIVE


@test("WakeWord: on_wake callback fires on activation", phase=3)
async def _():
    from wake_word import WakeWordDetector, WakeEvent
    from config import Config
    wwd = WakeWordDetector(Config())
    fired = []
    wwd.on_wake = lambda e: fired.append(e.wake_word)
    await wwd._activate(WakeEvent(wake_word="hey_assistant"))
    assert fired == ["hey_assistant"]


@test("WakeWord: force_idle resets to IDLE", phase=3)
async def _():
    from wake_word import WakeWordDetector, WakeState, WakeEvent
    from config import Config
    wwd = WakeWordDetector(Config())
    await wwd._activate(WakeEvent(wake_word="test"))
    wwd.force_idle()
    assert wwd.state == WakeState.IDLE


@test("WakeWord: loud audio triggers energy detector", phase=3)
async def _():
    import struct; from wake_word import WakeWordDetector; from config import Config
    cfg  = Config()
    wwd  = WakeWordDetector(cfg)
    loud = struct.pack("<" + "h" * cfg.chunk_frames, *[32767] * cfg.chunk_frames)
    assert wwd._energy_detect(loud) is not None


@test("WakeWord: silent audio does not trigger energy detector", phase=3)
async def _():
    from wake_word import WakeWordDetector; from config import Config
    wwd = WakeWordDetector(Config())
    assert wwd._energy_detect(b"\x00" * (Config().chunk_frames * 2)) is None


@test("Interrupt: starts IDLE with should_stop=False", phase=3)
async def _():
    from interrupt import InterruptController, InterruptState
    ic = InterruptController()
    assert ic.state == InterruptState.IDLE and not ic.should_stop


@test("Interrupt: barge_in ignored when not SPEAKING", phase=3)
async def _():
    from interrupt import InterruptController, InterruptState
    ic = InterruptController()
    ic.barge_in()
    assert ic.state == InterruptState.IDLE and not ic.should_stop


@test("Interrupt: barge_in while SPEAKING sets stop flag", phase=3)
async def _():
    from interrupt import InterruptController, InterruptState
    ic = InterruptController()
    ic.on_tts_start(); ic.barge_in()
    assert ic.should_stop and ic.state == InterruptState.BARGING


@test("Interrupt: on_tts_done resets to IDLE", phase=3)
async def _():
    from interrupt import InterruptController, InterruptState
    ic = InterruptController()
    ic.on_tts_start(); ic.on_tts_done()
    assert ic.state == InterruptState.IDLE and not ic.should_stop


@test("Interrupt: reset clears stop flag", phase=3)
async def _():
    from interrupt import InterruptController
    ic = InterruptController()
    ic.on_tts_start(); ic.barge_in(); ic.reset()
    assert not ic.should_stop


@test("Interrupt: callback fires with correct reason", phase=3)
async def _():
    from interrupt import InterruptController
    ic = InterruptController()
    fired = []
    ic.on_interrupt(lambda e: fired.append(e.reason))
    ic.on_tts_start(); ic.barge_in(reason="user_speech")
    assert fired == ["user_speech"]


@test("Interrupt: barge_in counter increments correctly", phase=3)
async def _():
    from interrupt import InterruptController
    ic = InterruptController()
    for _ in range(3): ic.on_tts_start(); ic.barge_in(); ic.reset()
    assert ic.stats()["barge_ins"] == 3


@test("BargeInDetector: loud chunk triggers controller", phase=3)
async def _():
    import struct
    from interrupt import InterruptController, BargeInDetector
    from config import Config
    ic = InterruptController(); ic.on_tts_start()
    bd = BargeInDetector(ic)
    loud = struct.pack("<" + "h" * Config().chunk_frames, *[5000] * Config().chunk_frames)
    for _ in range(3): bd.check(loud)
    assert ic.should_stop


@test("BargeInDetector: silent chunk does not trigger", phase=3)
async def _():
    from interrupt import InterruptController, BargeInDetector
    from config import Config
    ic = InterruptController(); ic.on_tts_start()
    bd = BargeInDetector(ic)
    for _ in range(10): bd.check(b"\x00" * (Config().chunk_frames * 2))
    assert not ic.should_stop


@test("Recorder: save and reload round-trip", phase=3)
async def _():
    import shutil, replay as rmod
    from config import Config; from replay import SessionRecorder, SessionReplayer
    orig = rmod._SESSIONS_DIR
    tmp  = Path(tempfile.mkdtemp())
    rmod._SESSIONS_DIR = tmp
    try:
        cfg = Config()
        rec = SessionRecorder(cfg)
        chunk = b"\x01\x02" * cfg.chunk_frames
        for _ in range(10): rec.record(chunk)
        rec.add_transcript("hello world")
        path = rec.save("test_session")
        assert (path / "audio.raw").exists()
        assert (path / "meta.json").exists()
        assert (path / "transcript.json").exists()
        rep = SessionReplayer(cfg, "test_session", speed=0.0)
        chunks = [c async for c in rep.stream()]
        assert len(chunks) == 10
    finally:
        rmod._SESSIONS_DIR = orig
        shutil.rmtree(tmp)


@test("Replayer: speed=0 completes faster than real-time", phase=3)
async def _():
    import shutil, replay as rmod
    from config import Config; from replay import SessionRecorder, SessionReplayer
    orig = rmod._SESSIONS_DIR
    tmp  = Path(tempfile.mkdtemp())
    rmod._SESSIONS_DIR = tmp
    try:
        cfg = Config()
        rec = SessionRecorder(cfg)
        for _ in range(30): rec.record(b"\x00" * (cfg.chunk_frames * 2))
        rec.save("speed_test")
        t0  = time.monotonic()
        rep = SessionReplayer(cfg, "speed_test", speed=0.0)
        cnt = 0
        async for _ in rep.stream():
            cnt += 1
        assert cnt == 30 and (time.monotonic() - t0) < 1.0
    finally:
        rmod._SESSIONS_DIR = orig
        shutil.rmtree(tmp)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

async def _main():
    p = argparse.ArgumentParser(description="Voice Assistant test suite")
    p.add_argument("--phase", type=int, default=None, help="Run only phase N (1, 2, or 3)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    label = f"Phase {args.phase}" if args.phase else "All phases"
    print(f"\n{BOLD}Voice Assistant -- Test Suite{RESET}  [{label}]\n")
    ok = await _run_all(args.phase, args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(_main())

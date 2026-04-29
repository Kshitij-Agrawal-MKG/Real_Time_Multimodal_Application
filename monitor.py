"""monitor.py -- ANSI terminal live monitor for Windows 11."""

import argparse
import asyncio
import json
import os
import sys
import io
import time
from collections import deque
from pathlib import Path
import ctypes
import ctypes.wintypes


# Enable ANSI + UTF-8 on Windows
def _init_terminal() -> bool:
    try:
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11); m = ctypes.wintypes.DWORD()
        k.GetConsoleMode(h, ctypes.byref(m))
        k.SetConsoleMode(h, m.value | 0x0004)
        k.SetConsoleOutputCP(65001)
        return True
    except Exception:
        return False


_ANSI = _init_terminal()

def _c(code: str) -> str: return code if _ANSI else ""

RESET   = _c("\033[0m");  BOLD    = _c("\033[1m");  DIM    = _c("\033[2m")
GREEN   = _c("\033[32m"); CYAN    = _c("\033[36m");  YELLOW = _c("\033[33m")
RED     = _c("\033[31m"); MAGENTA = _c("\033[35m")

# Box chars (Consolas/Lucida Console ship with Windows and support these)
try:
    test_str = "┌─┐│└┘├┤█░"
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    H, V = "─", "│"
    TL, TR, BL, BR, ML, MR = "┌", "┐", "└", "┘", "├", "┤"
    FILL, EMPTY = "█", "░"
except Exception:
    H, V = "-", "|"
    TL, TR, BL, BR, ML, MR = "+", "+", "+", "+", "+", "+"
    FILL, EMPTY = "#", "."

W = 62  # inner box width


def _ms(v, g=600, w=1200):
    if v is None: return DIM + "--" + RESET
    c = GREEN if v < g else (YELLOW if v < w else RED)
    return f"{c}{v:.0f}ms{RESET}"


def _bar(v, mx=1500, n=20):
    if not v: return DIM + EMPTY * n + RESET
    f = min(n, int(v / mx * n))
    return CYAN + FILL * f + DIM + EMPTY * (n - f) + RESET


def _row(content):
    import re
    plain = re.sub(r"\033\[[0-9;]*m", "", content)
    pad   = max(0, W - len(plain))
    return DIM + V + RESET + content + " " * pad + DIM + V + RESET


class TerminalMonitor:
    def __init__(self):
        self._records:  list[dict] = []
        self._log:      deque      = deque(maxlen=5)
        self._state:    str        = "IDLE"
        self._t0:       float      = time.monotonic()

    def ingest(self, r: dict):
        self._records.append(r)
        self._state = "IDLE"

    def set_state(self, s: str): self._state = s

    def add_log(self, stage: str, msg: str):
        self._log.append((time.strftime("%H:%M:%S"), stage, msg))

    def _pct(self, data: list, p: float):
        if not data: return None
        s = sorted(data)
        return s[min(int(len(s) * p / 100), len(s) - 1)]

    def _roll(self, key: str, n: int = 10):
        return [r[key] for r in self._records[-n:] if r.get(key)]

    def render(self) -> str:
        up  = int(time.monotonic() - self._t0)
        h, m, s = up // 3600, (up % 3600) // 60, up % 60
        utt = len(self._records)
        lines = []

        lines.append(DIM + TL + H * W + TR + RESET)
        lines.append(_row(f"  {BOLD}VOICE PIPELINE MONITOR{RESET}         {DIM}uptime: {h:02d}:{m:02d}:{s:02d}{RESET}"))
        lines.append(DIM + ML + H * W + MR + RESET)

        sc = {"IDLE": DIM, "LISTENING": GREEN, "SPEAKING": CYAN, "ERROR": RED}.get(self._state, DIM)
        lines.append(_row(f"  {BOLD}STATE:{RESET} {sc}{self._state:<10}{RESET}  {DIM}utterances:{RESET} {BOLD}{utt}{RESET}"))
        lines.append(DIM + ML + H * W + MR + RESET)

        if self._records:
            r   = self._records[-1]
            uid = r.get("utterance_id", "?")
            lines.append(_row(f"  {BOLD}LAST UTTERANCE #{uid}{RESET}"))
            for lbl, key in [("ASR", "asr_latency_ms"), ("LLM", "llm_ttft_ms"),
                              ("TTS", "tts_ttfb_ms"),   ("OH ", "overhead_ms")]:
                v = r.get(key) or 0
                lines.append(_row(f"  {DIM}{lbl}{RESET}  {_bar(v)}  {_ms(v)}"))
            total = r.get("total_ms")
            icon  = (GREEN + "OK" if (total or 9999) < 1500 else RED + "!!") + RESET
            lines.append(_row(f"  {'':>26}TOTAL: {_ms(total, 1000, 1800)}  [{icon}]"))
        else:
            lines.append(_row(f"  {DIM}Waiting for first utterance...{RESET}"))

        lines.append(DIM + ML + H * W + MR + RESET)
        lines.append(_row(f"  {BOLD}ROLLING AVERAGES  (last 10){RESET}"))
        for lbl, key in [("ASR", "asr_latency_ms"), ("LLM", "llm_ttft_ms"), ("TTS", "tts_ttfb_ms")]:
            data = self._roll(key)
            p50  = self._pct(data, 50)
            p95  = self._pct(data, 95)
            lines.append(_row(f"  {DIM}{lbl}{RESET}  p50={_ms(p50) if p50 else DIM+'--'+RESET}  "
                               f"p95={_ms(p95, 800, 1500) if p95 else DIM+'--'+RESET}"))

        lines.append(DIM + ML + H * W + MR + RESET)
        lines.append(_row(f"  {BOLD}RECENT LOG{RESET}"))
        if self._log:
            sc_map = {"ASR": CYAN, "LLM": MAGENTA, "TTS": YELLOW, "OK": GREEN, "ERR": RED}
            for ts, stage, msg in self._log:
                c   = sc_map.get(stage, DIM)
                msg = (msg[:38] + "...") if len(msg) > 40 else msg
                lines.append(_row(f"  {DIM}{ts}{RESET}  {c}{stage:<4}{RESET}  {msg}"))
        else:
            lines.append(_row(f"  {DIM}(waiting){RESET}"))

        while len(lines) < 22:
            lines.append(_row(""))

        lines.append(DIM + BL + H * W + BR + RESET)
        lines.append(f"{DIM}  press Ctrl+C to quit{RESET}")
        return "\n".join(lines)


async def run_live(port: int = 8765):
    m = TerminalMonitor()
    print(f"Connecting to ws://localhost:{port}...")
    try:
        import websockets
        async with websockets.connect(f"ws://localhost:{port}") as ws:
            sys.stdout.write(_c("\033[?25l"))
            async for raw in ws:
                msg = json.loads(raw)
                records = ([msg["data"]] if msg["type"] == "utterance"
                           else msg.get("history", []))
                for r in records:
                    m.ingest(r)
                    m.add_log("OK", f"#{r.get('utterance_id')} total={r.get('total_ms')}ms")
                os.system("cls"); print(m.render(), flush=True)
    except ImportError:
        print("websockets not installed: pip install websockets")
    except Exception as e:
        print(f"Connection failed: {e}")
    finally:
        sys.stdout.write(_c("\033[?25h"))


async def run_file(path: Path):
    m = TerminalMonitor()
    for r in json.loads(path.read_text(encoding="utf-8")):
        m.ingest(r)
        m.add_log("OK", f"#{r.get('utterance_id')} total={r.get('total_ms')}ms")
    os.system("cls")
    sys.stdout.write(_c("\033[?25l"))
    print(m.render())
    sys.stdout.write(_c("\033[?25h"))
    print(f"\n  {len(m._records)} utterances loaded from {path}")


async def run_demo():
    import random
    m = TerminalMonitor()
    uid = 0
    sys.stdout.write(_c("\033[?25l"))
    try:
        while True:
            m.set_state("LISTENING"); os.system("cls"); print(m.render(), flush=True)
            await asyncio.sleep(0.8)
            uid += 1
            asr = max(50,  random.gauss(180, 40))
            llm = max(80,  random.gauss(400, 100))
            tts = max(40,  random.gauss(200, 40))
            oh  = max(5,   random.gauss(30, 10))
            r   = {"utterance_id": uid,
                   "asr_latency_ms": round(asr, 1), "llm_ttft_ms": round(llm, 1),
                   "tts_ttfb_ms":    round(tts, 1), "overhead_ms": round(oh, 1),
                   "total_ms":       round(asr + llm + tts + oh, 1)}
            m.ingest(r); m.set_state("SPEAKING")
            m.add_log("ASR", f"utterance #{uid} transcribed")
            m.add_log("LLM", "response received")
            m.add_log("TTS", f"playing ({r['tts_ttfb_ms']}ms)")
            os.system("cls"); print(m.render(), flush=True)
            await asyncio.sleep(random.uniform(1.5, 3.0))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        sys.stdout.write(_c("\033[?25h") + RESET)


def main():
    p = argparse.ArgumentParser(description="Voice pipeline terminal monitor")
    p.add_argument("--file", type=Path, default=None, help="Load from latency_log.json")
    p.add_argument("--port", type=int,  default=8765, help="WebSocket port (default 8765)")
    p.add_argument("--demo", action="store_true",     help="Run with simulated data")
    args = p.parse_args()

    if args.demo:    asyncio.run(run_demo())
    elif args.file:  asyncio.run(run_file(args.file))
    else:            asyncio.run(run_live(args.port))


if __name__ == "__main__":
    main()

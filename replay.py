"""replay.py — Session recorder and replayer for debugging and benchmarking."""

import asyncio
import json
import logging
import shutil
import time
from pathlib import Path
from typing import AsyncIterator, Optional

log = logging.getLogger("replay")

_SESSIONS_DIR = Path("sessions")


class SessionRecorder:
    """Records live mic audio and transcripts to disk."""

    def __init__(self, cfg):
        self.cfg     = cfg
        self._chunks: list[bytes]  = []
        self._events: list[dict]   = []
        self._t0:     Optional[float] = None

    def record(self, chunk: bytes):
        if self._t0 is None:
            self._t0 = time.monotonic()
        self._chunks.append(chunk)

    def add_transcript(self, text: str, final: bool = True):
        elapsed = round(time.monotonic() - self._t0, 3) if self._t0 else 0
        self._events.append({"t_s": elapsed, "text": text, "final": final})

    def save(self, name: Optional[str] = None) -> Path:
        if not self._chunks:
            raise ValueError("Nothing recorded")
        name  = name or f"session_{int(time.time())}"
        out   = _SESSIONS_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "audio.raw").write_bytes(b"".join(self._chunks))
        (out / "meta.json").write_text(json.dumps({
            "name":        name,
            "sample_rate": self.cfg.sample_rate,
            "channels":    self.cfg.channels,
            "chunk_ms":    self.cfg.chunk_ms,
            "n_chunks":    len(self._chunks),
            "duration_s":  round(len(self._chunks) * self.cfg.chunk_ms / 1000, 2),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2))
        if self._events:
            (out / "transcript.json").write_text(json.dumps(self._events, indent=2))
        log.info(f"[Recorder] saved {len(self._chunks)} chunks -> {out}")
        return out

    def reset(self):
        self._chunks.clear()
        self._events.clear()
        self._t0 = None


class SessionReplayer:
    """Replays a saved session as an async generator of PCM chunks."""

    def __init__(self, cfg, session_name: str, speed: float = 1.0):
        self.cfg   = cfg
        self.speed = speed
        self._dir  = _SESSIONS_DIR / session_name
        if not self._dir.exists():
            raise FileNotFoundError(f"Session not found: {self._dir}")
        self.meta = json.loads((self._dir / "meta.json").read_text())
        log.info(f"[Replayer] {session_name} -- {self.meta['duration_s']}s, {self.meta['n_chunks']} chunks")

    async def stream(self) -> AsyncIterator[bytes]:
        audio      = (self._dir / "audio.raw").read_bytes()
        chunk_size = self.cfg.chunk_frames * 2
        sleep_s    = (self.cfg.chunk_ms / 1000) / max(self.speed, 0.001)
        count      = 0
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = chunk.ljust(chunk_size, b"\x00")
            yield chunk
            count += 1
            if self.speed > 0:
                await asyncio.sleep(sleep_s)
        log.info(f"[Replayer] done -- {count} chunks")

    def ground_truth(self) -> list[dict]:
        p = self._dir / "transcript.json"
        return json.loads(p.read_text()) if p.exists() else []


# ── Session management helpers ─────────────────────────────────────────────────

def list_sessions() -> list[dict]:
    if not _SESSIONS_DIR.exists():
        return []
    sessions = []
    for d in sorted(_SESSIONS_DIR.iterdir()):
        meta = d / "meta.json"
        if meta.exists():
            sessions.append(json.loads(meta.read_text()))
    return sessions


def delete_session(name: str):
    p = _SESSIONS_DIR / name
    if p.exists():
        shutil.rmtree(p)
        log.info(f"[Replay] deleted session: {name}")

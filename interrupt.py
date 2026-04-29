"""interrupt.py — Barge-in detection and TTS abort control."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

log = logging.getLogger("interrupt")


class InterruptState(Enum):
    IDLE     = auto()
    SPEAKING = auto()
    BARGING  = auto()


@dataclass
class InterruptEvent:
    reason:    str
    timestamp: float = field(default_factory=time.monotonic)


class InterruptController:
    """
    Shared stop-token threaded through TTS and the microphone monitor.

    TTS checks should_stop on every PCM chunk write.
    BargeInDetector calls barge_in() when it detects speech during playback.
    """

    def __init__(self):
        self.state      = InterruptState.IDLE
        self._stop      = asyncio.Event()
        self._callbacks: list[Callable] = []
        self._stats     = {"barge_ins": 0, "tts_completions": 0}

    # ── TTS lifecycle ──────────────────────────────────────────────────────────

    def on_tts_start(self):
        self.state = InterruptState.SPEAKING
        self._stop.clear()

    def on_tts_done(self):
        self.state = InterruptState.IDLE
        self._stats["tts_completions"] += 1

    @property
    def should_stop(self) -> bool:
        return self._stop.is_set()

    @property
    def is_speaking(self) -> bool:
        return self.state == InterruptState.SPEAKING

    @property
    def interrupted(self) -> bool:
        return self._stop.is_set()

    # ── Interrupt triggers ─────────────────────────────────────────────────────

    def barge_in(self, reason: str = "barge_in"):
        if self.state != InterruptState.SPEAKING:
            return
        self.state = InterruptState.BARGING
        self._stop.set()
        self._stats["barge_ins"] += 1
        log.info(f"[Interrupt] barge-in: {reason!r}")
        event = InterruptEvent(reason=reason)
        for cb in self._callbacks:
            try:
                result = cb(event)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                log.warning(f"[Interrupt] callback error: {e}")

    def reset(self):
        self.state = InterruptState.IDLE
        self._stop.clear()

    def on_interrupt(self, callback: Callable):
        self._callbacks.append(callback)
        return callback

    def stats(self) -> dict:
        return {**self._stats, "state": self.state.name}


class BargeInDetector:
    """
    Monitors mic audio during TTS playback and triggers barge-in on speech.
    More sensitive than the main VAD -- prefers false positives over missed interrupts.
    """

    _THRESHOLD     = 600  # RMS amplitude
    _CONFIRM_FRAMES = 2

    def __init__(self, controller: InterruptController):
        self._ctrl   = controller
        self._streak = 0

    def check(self, chunk: bytes):
        if not self._ctrl.is_speaking:
            self._streak = 0
            return
        import struct, math
        samples = struct.unpack_from(f"<{len(chunk) // 2}h", chunk)
        rms = math.sqrt(sum(s * s for s in samples) / max(len(samples), 1))
        if rms > self._THRESHOLD:
            self._streak += 1
            if self._streak >= self._CONFIRM_FRAMES:
                self._ctrl.barge_in()
                self._streak = 0
        else:
            self._streak = max(0, self._streak - 1)

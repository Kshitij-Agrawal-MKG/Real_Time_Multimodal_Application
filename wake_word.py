"""wake_word.py — Wake-word gating: IDLE -> ACTIVE -> IDLE state machine."""

import asyncio
import collections
import logging
import struct
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator, Callable, Optional

log = logging.getLogger("wake_word")

_ACTIVE_WINDOW_S    = 8.0   # seconds to stay active after detection
_PRE_ROLL_FRAMES    = 20    # frames buffered before speech start
_ENERGY_THRESHOLD   = 8000  # RMS threshold for energy-based fallback
_CONFIRM_FRAMES     = 3     # loud frames required to trigger


class WakeState(Enum):
    IDLE   = auto()
    ACTIVE = auto()


@dataclass
class WakeEvent:
    wake_word:  str
    confidence: float = 1.0
    timestamp:  float = field(default_factory=time.monotonic)


class WakeWordDetector:
    """
    Gates audio through to the rest of the pipeline.
    IDLE: consumes audio silently, listening for trigger phrase.
    ACTIVE: forwards all audio until timeout or manual deactivation.

    Backends (tried in order):
      1. openWakeWord (neural ONNX) if installed
      2. Energy-based RMS threshold (no extra dependencies)
    """

    def __init__(self, cfg, wake_words: set[str] | None = None,
                 active_window_s: float = _ACTIVE_WINDOW_S):
        self.cfg             = cfg
        self.wake_words      = {w.lower().strip() for w in (wake_words or {"hey assistant", "ok computer", "hello there"})}
        self.active_window_s = active_window_s
        self.state           = WakeState.IDLE
        self.on_wake: Optional[Callable] = None

        self._ring: collections.deque  = collections.deque(maxlen=_PRE_ROLL_FRAMES)
        self._active_since: Optional[float] = None
        self._stats = {"detections": 0}
        self._oww   = None
        self._init_oww()

    def _init_oww(self):
        try:
            from openwakeword.model import Model
            self._oww = Model(wakeword_models=[], inference_framework="onnx")
            log.info("WakeWord: openWakeWord backend loaded")
        except ImportError:
            log.info("WakeWord: energy-based backend (pip install openwakeword for better accuracy)")
        except Exception as e:
            log.warning(f"WakeWord: openWakeWord failed ({e}) -- using energy fallback")

    # ── Public stream ──────────────────────────────────────────────────────────

    async def stream(self, audio_q: asyncio.Queue) -> AsyncIterator[bytes]:
        """Yield audio chunks only while in ACTIVE state."""
        while True:
            chunk = await audio_q.get()
            self._ring.append(chunk)

            if self.state == WakeState.IDLE:
                event = self._detect(chunk)
                if event:
                    await self._activate(event)
                    for buffered in list(self._ring):
                        yield buffered
                    self._ring.clear()
            else:
                yield chunk
                if self._active_since and (time.monotonic() - self._active_since) > self.active_window_s:
                    log.info(f"[WakeWord] active window expired -> IDLE")
                    self._deactivate()
                    yield None  # utterance boundary

    # ── Detection ──────────────────────────────────────────────────────────────

    def _detect(self, chunk: bytes) -> Optional[WakeEvent]:
        if self._oww:
            try:
                import numpy as np
                samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                for word, score in self._oww.predict(samples).items():
                    if score > 0.5:
                        self._stats["detections"] += 1
                        return WakeEvent(wake_word=word, confidence=float(score))
            except Exception as e:
                log.debug(f"[WakeWord] OWW predict: {e}")
        return self._energy_detect(chunk)

    def _energy_detect(self, chunk: bytes) -> Optional[WakeEvent]:
        samples = struct.unpack_from(f"<{len(chunk) // 2}h", chunk)
        rms = math.sqrt(sum(s * s for s in samples) / max(len(samples), 1))
        if rms > _ENERGY_THRESHOLD:
            self._stats["detections"] += 1
            return WakeEvent(wake_word="energy_trigger", confidence=rms / 32767)
        return None

    # ── State transitions ──────────────────────────────────────────────────────

    async def _activate(self, event: WakeEvent):
        self.state        = WakeState.ACTIVE
        self._active_since = time.monotonic()
        log.info(f"[WakeWord] ACTIVE -- {event.wake_word!r} (conf={event.confidence:.2f})")
        if self.on_wake:
            result = self.on_wake(event)
            if asyncio.iscoroutine(result):
                await result

    def _deactivate(self):
        self.state        = WakeState.IDLE
        self._active_since = None

    def force_activate(self):
        asyncio.create_task(self._activate(WakeEvent(wake_word="manual")))

    def force_idle(self):
        self._deactivate()

    def stats(self) -> dict:
        return {**self._stats, "state": self.state.name}

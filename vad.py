"""vad.py -- Voice Activity Detection for Windows 11."""

import array
import asyncio
import logging
import math
from collections import deque
from typing import AsyncIterator, Optional

log = logging.getLogger("vad")

_ENERGY_THRESHOLD    = 300
_HANGOVER_FRAMES     = 15
_SPEECH_START_FRAMES = 3
_SILENCE_END_FRAMES  = 20


def _rms(pcm: bytes) -> float:
    samples = array.array("h", pcm)
    return math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0


class VoiceActivityDetector:
    """
    Filters mic audio, yielding only speech chunks.
    Yields None as an utterance-boundary sentinel when silence is detected.

    Uses webrtcvad-wheels (better accuracy) if installed,
    otherwise falls back to energy-based RMS detection.
    Install better VAD: pip install webrtcvad-wheels
    """

    def __init__(self, cfg, threshold: float = _ENERGY_THRESHOLD):
        self.cfg       = cfg
        self.threshold = threshold
        self._in_speech    = False
        self._loud_count   = 0
        self._silent_count = 0
        self._prebuffer: deque = deque()
        self._stats = {"total_frames": 0, "speech_frames": 0, "utterances": 0}

        self._wvad = None
        try:
            import webrtcvad
            self._wvad = webrtcvad.Vad(2)
            log.info("VAD: webrtcvad (aggressiveness=2)")
        except ImportError:
            log.info(f"VAD: energy-based (threshold={threshold:.0f} RMS)")

    def _is_speech(self, chunk: bytes) -> bool:
        if self._wvad:
            try:
                return self._wvad.is_speech(chunk, self.cfg.sample_rate)
            except Exception:
                pass
        return _rms(chunk) > self.threshold

    async def filter(self, audio_q: asyncio.Queue) -> AsyncIterator[Optional[bytes]]:
        while True:
            chunk = await audio_q.get()
            self._stats["total_frames"] += 1
            speech = self._is_speech(chunk)

            if speech:
                self._loud_count   += 1
                self._silent_count  = 0
            else:
                self._silent_count += 1
                self._loud_count    = 0

            if not self._in_speech:
                self._prebuffer.append(chunk)
                if len(self._prebuffer) > _HANGOVER_FRAMES:
                    self._prebuffer.popleft()
                if self._loud_count >= _SPEECH_START_FRAMES:
                    self._in_speech = True
                    self._stats["utterances"] += 1
                    log.debug(f"[VAD] utterance #{self._stats['utterances']} opened")
                    for buffered in self._prebuffer:
                        yield buffered
                    self._prebuffer.clear()
            else:
                self._stats["speech_frames"] += 1
                yield chunk
                if self._silent_count >= _SILENCE_END_FRAMES:
                    self._in_speech    = False
                    self._loud_count   = 0
                    log.debug("[VAD] utterance closed")
                    yield None  # utterance-boundary sentinel

    async def calibrate(self, audio_q: asyncio.Queue, duration_s: float = 1.5):
        """Sample ambient noise and auto-tune threshold to 3x ambient RMS."""
        n = int(duration_s * 1000 / self.cfg.chunk_ms)
        log.info(f"[VAD] calibrating for {duration_s}s -- stay quiet...")
        energies = [_rms(await audio_q.get()) for _ in range(n)]
        ambient  = sum(energies) / len(energies)
        self.threshold = max(200, ambient * 3)
        log.info(f"[VAD] ambient={ambient:.0f}  threshold={self.threshold:.0f}")
        return self.threshold

    def adjust_threshold(self, value: float):
        log.info(f"[VAD] threshold {self.threshold:.0f} -> {value:.0f}")
        self.threshold = value

    def stats(self) -> dict:
        total  = self._stats["total_frames"]
        speech = self._stats["speech_frames"]
        return {**self._stats,
                "speech_ratio_pct": round(speech / total * 100, 1) if total else 0.0}

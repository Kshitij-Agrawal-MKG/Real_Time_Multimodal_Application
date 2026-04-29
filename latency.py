"""latency.py -- Per-utterance latency tracking."""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("latency")

# Sanity cap: any single-stage value above this is a stale timestamp, not real latency
_MAX_REASONABLE_MS = 30_000  # 30 seconds


@dataclass
class UtteranceRecord:
    utterance_id: int = 0

    # Raw perf_counter timestamps (seconds)
    t_asr_start:     Optional[float] = None
    t_asr_first:     Optional[float] = None
    t_llm_start:     Optional[float] = None
    t_llm_first:     Optional[float] = None
    t_tts_start:     Optional[float] = None
    t_tts_first:     Optional[float] = None
    t_audio_playing: Optional[float] = None

    # Derived latencies (milliseconds)
    asr_latency_ms: Optional[float] = None
    llm_ttft_ms:    Optional[float] = None
    tts_ttfb_ms:    Optional[float] = None
    overhead_ms:    Optional[float] = None
    total_ms:       Optional[float] = None

    def compute(self):
        def ms(a, b) -> Optional[float]:
            if a is None or b is None:
                return None
            v = round((b - a) * 1000, 1)
            # Guard against stale timestamps producing absurd values
            return v if 0 < v < _MAX_REASONABLE_MS else None

        self.asr_latency_ms = ms(self.t_asr_start, self.t_asr_first)
        self.llm_ttft_ms    = ms(self.t_llm_start, self.t_llm_first)
        self.tts_ttfb_ms    = ms(self.t_tts_start, self.t_tts_first)

        gaps = []
        if self.t_asr_first and self.t_llm_start:
            g = (self.t_llm_start - self.t_asr_first) * 1000
            if 0 <= g < _MAX_REASONABLE_MS:
                gaps.append(g)
        if self.t_llm_first and self.t_tts_start:
            g = (self.t_tts_start - self.t_llm_first) * 1000
            if 0 <= g < _MAX_REASONABLE_MS:
                gaps.append(g)
        self.overhead_ms = round(sum(gaps), 1) if gaps else 0.0
        self.total_ms    = ms(self.t_asr_start, self.t_tts_first)

    def summary(self) -> str:
        def fmt(v): return f"{v:.0f}" if v is not None else "--"
        return (
            f"#{self.utterance_id} | "
            f"ASR={fmt(self.asr_latency_ms)}ms  "
            f"LLM={fmt(self.llm_ttft_ms)}ms  "
            f"TTS={fmt(self.tts_ttfb_ms)}ms  "
            f"TOTAL={fmt(self.total_ms)}ms"
        )


class LatencyTracker:
    """High-resolution per-utterance latency tracker using time.perf_counter()."""

    def __init__(self):
        self._records: list[UtteranceRecord] = []
        self._current = UtteranceRecord(utterance_id=1)

    def mark_asr_start(self):
        """Call when the first PCM chunk of a new utterance is received."""
        self._current.t_asr_start = time.perf_counter()

    def mark_asr_first(self):
        """Call when Vosk emits the final transcript for this utterance."""
        t = time.perf_counter()
        self._current.t_asr_first  = t
        self._current.t_llm_start  = t   # LLM starts immediately after ASR

    def mark_llm_start(self):
        if not self._current.t_llm_start:
            self._current.t_llm_start = time.perf_counter()

    def mark_llm_first(self):
        """Call when the first token / full response arrives from Gemini."""
        t = time.perf_counter()
        self._current.t_llm_first  = t
        self._current.t_tts_start  = t   # TTS starts immediately after LLM

    def mark_tts_start(self):
        if not self._current.t_tts_start:
            self._current.t_tts_start = time.perf_counter()

    def mark_tts_first(self):
        """Call when the first PCM byte is written to the speaker."""
        self._current.t_tts_first = time.perf_counter()

    def mark_audio_playing(self):
        self._current.t_audio_playing = time.perf_counter()

    def log_summary(self):
        self._current.compute()
        log.info(self._current.summary())
        self._records.append(self._current)
        self._current = UtteranceRecord(utterance_id=self._current.utterance_id + 1)

    def export_json(self, path: Path):
        path.write_text(json.dumps([asdict(r) for r in self._records], indent=2))

    def latest_dict(self) -> dict:
        return asdict(self._records[-1]) if self._records else {}

    def all_dicts(self) -> list[dict]:
        return [asdict(r) for r in self._records]

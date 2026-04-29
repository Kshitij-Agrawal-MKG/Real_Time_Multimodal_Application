"""asr.py -- Vosk local offline ASR."""

import asyncio
import json
import logging
from pathlib import Path
from typing import AsyncIterator

from config import Config
from latency import LatencyTracker

log = logging.getLogger("asr")

# Sentinel object sent on vad_q to signal end-of-utterance
UTTERANCE_END = object()


class VoskASR:
    """
    Transcribes PCM audio using Vosk (fully local, no internet required).

    Protocol with vad_stage:
      - Receives PCM bytes chunks for processing
      - Receives UTTERANCE_END sentinel when VAD detects end of speech
      - On UTTERANCE_END: calls rec.FinalResult() to flush any buffered words
      - Yields the final transcript string for each utterance

    This design means Vosk sees speech audio + gets explicitly told when
    the utterance ended, instead of waiting for in-band silence detection.
    """

    def __init__(self, cfg: Config, tracker: LatencyTracker):
        self.cfg     = cfg
        self.tracker = tracker
        self._model  = None

    def _load_model(self):
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)
        path = Path(self.cfg.vosk_model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Vosk model not found: {path}\n"
                f"Download from https://alphacephei.com/vosk/models\n"
                f"Extract so that {path}\\conf\\ and {path}\\am\\ exist."
            )
        log.info(f"[ASR] loading model from {path}...")
        self._model = Model(str(path))
        log.info("[ASR] model ready")

    async def stream(self, audio_q: asyncio.Queue) -> AsyncIterator[str]:
        from vosk import KaldiRecognizer

        if self._model is None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._load_model)

        rec  = KaldiRecognizer(self._model, self.cfg.sample_rate)
        rec.SetWords(False)  # faster without word timestamps
        loop = asyncio.get_running_loop()
        utterance_started = False

        while True:
            item = await audio_q.get()

            if item is UTTERANCE_END:
                # VAD says utterance is over -- flush Vosk's buffer now
                if utterance_started:
                    result = await loop.run_in_executor(None, rec.FinalResult)
                    text = json.loads(result).get("text", "").strip()
                    utterance_started = False
                    if text:
                        self.tracker.mark_asr_first()
                        log.info(f"[ASR] {text!r}")
                        yield text
                continue

            chunk = item  # it's a PCM bytes chunk

            if not utterance_started:
                self.tracker.mark_asr_start()
                utterance_started = True

            # Run AcceptWaveform in executor -- it's CPU-bound
            accepted = await loop.run_in_executor(None, rec.AcceptWaveform, chunk)

            if accepted:
                # Vosk itself detected an internal boundary (rare with filtered audio)
                text = json.loads(rec.Result()).get("text", "").strip()
                if text:
                    self.tracker.mark_asr_first()
                    log.info(f"[ASR] {text!r}")
                    utterance_started = False
                    yield text
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "").strip()
                if partial:
                    log.debug(f"[ASR partial] {partial!r}")

    def reset(self):
        self._model = None
        log.info("[ASR] model unloaded")

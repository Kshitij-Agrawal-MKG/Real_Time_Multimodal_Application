"""tts.py -- AWS Polly streaming TTS with PyAudio playback on Windows 11."""

import asyncio
import logging
from pathlib import Path

from config import Config
from latency import LatencyTracker

log = logging.getLogger("tts")

_CANNED_DIR = Path(__file__).parent / "canned"
_CANNED_PHRASES = {
    "processing": "Just a moment, I'm still processing.",
    "asr_error":  "Sorry, I didn't catch that. Could you repeat?",
    "llm_error":  "Sorry, I'm having trouble thinking right now.",
    "tts_error":  "There was an audio issue. Please try again.",
}


class PollyTTS:
    """Synthesises text via AWS Polly and plays raw PCM through PyAudio."""

    def __init__(self, cfg: Config, tracker: LatencyTracker):
        self.cfg     = cfg
        self.tracker = tracker
        self._polly  = None
        self._pa     = None
        self._stream = None

    async def speak(self, text: str):
        """Synthesise and play audio. All blocking calls run in thread executor."""
        loop  = asyncio.get_running_loop()
        polly = self._get_polly()
        self.tracker.mark_tts_start()

        def _synthesise():
            return polly.synthesize_speech(
                Text=text,
                VoiceId=self.cfg.polly_voice,
                Engine=self.cfg.polly_engine,
                OutputFormat=self.cfg.polly_format,
                SampleRate="16000",
            )

        response = await loop.run_in_executor(None, _synthesise)

        def _play(audio_stream):
            out   = self._get_output_stream()
            first = True
            for chunk in audio_stream.iter_chunks(chunk_size=2048):
                if first:
                    self.tracker.mark_tts_first()
                    log.info("[TTS] first byte -> speaker")
                    first = False
                out.write(chunk)

        await loop.run_in_executor(None, _play, response["AudioStream"])
        log.debug("[TTS] playback done")

    async def speak_cached(self, key: str):
        """Play a pre-recorded PCM fallback clip, or synthesise on the fly."""
        path = _CANNED_DIR / f"{key}.pcm"
        if path.exists():
            loop = asyncio.get_running_loop()
            def _play_file():
                out  = self._get_output_stream()
                data = path.read_bytes()
                for i in range(0, len(data), 2048):
                    out.write(data[i:i + 2048])
            await loop.run_in_executor(None, _play_file)
        else:
            await self.speak(_CANNED_PHRASES.get(key, "Sorry, something went wrong."))

    def close(self):
        for obj, method in [(self._stream, "stop_stream"), (self._stream, "close"),
                            (self._pa, "terminate")]:
            if obj:
                try: getattr(obj, method)()
                except Exception: pass

    def _get_polly(self):
        if self._polly is None:
            import boto3
            self._polly = boto3.client(
                "polly",
                region_name=self.cfg.aws_region,
                aws_access_key_id=self.cfg.aws_access_key,
                aws_secret_access_key=self.cfg.aws_secret_key,
            )
        return self._polly

    def _get_output_stream(self):
        import pyaudio
        if self._pa is None:
            self._pa = pyaudio.PyAudio()
        if self._stream is None or not self._stream.is_active():
            if self._stream is not None:
                try: self._stream.close()
                except Exception: pass
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16_000,
                output=True,
                frames_per_buffer=1024,
            )
        return self._stream

"""mic.py -- Microphone capture via PyAudio on Windows 11."""

import asyncio
import logging
from typing import AsyncIterator

log = logging.getLogger("mic")


class MicrophoneStream:
    """Captures PCM audio from the default microphone as an async generator of chunks."""

    def __init__(self, cfg):
        self.cfg    = cfg
        self._loop:  asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None

    async def stream(self) -> AsyncIterator[bytes]:
        import pyaudio

        self._loop  = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=200)

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=self.cfg.channels,
            rate=self.cfg.sample_rate,
            input=True,
            frames_per_buffer=self.cfg.chunk_frames,
            stream_callback=self._callback,
        )

        log.info(f"Microphone open -- {self.cfg.sample_rate}Hz, {self.cfg.chunk_ms}ms chunks")
        stream.start_stream()
        try:
            while True:
                chunk = await self._queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            log.info("Microphone closed")

    def _callback(self, in_data, frame_count, time_info, status):
        import pyaudio
        if self._loop and self._queue:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, in_data)
        return (None, pyaudio.paContinue)

    def stop(self):
        if self._queue:
            self._queue.put_nowait(None)

"""config.py -- All pipeline parameters, loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Audio -- 20ms chunks give lower ASR latency than 30ms
    sample_rate: int  = 16_000
    channels: int     = 1
    chunk_ms: int     = 20

    # Vosk ASR (fully local, no API key needed)
    vosk_model_path: str = str(Path(__file__).parent / "vosk_model")

    # Gemini LLM -- 2.0-flash is significantly faster than 1.5-flash
    gemini_api_key: str = ""
    gemini_model:   str = "gemini-2.0-flash"
    system_prompt:  str = (
        "You are a concise voice assistant. "
        "Reply in 1-3 short spoken sentences. "
        "No markdown, no lists, no special characters."
    )

    # AWS Polly TTS
    aws_access_key: str = ""
    aws_secret_key: str = ""
    aws_region:     str = "us-east-1"
    polly_voice:    str = "Joanna"
    polly_engine:   str = "neural"
    polly_format:   str = "pcm"

    # Resilience timeouts (seconds)
    asr_timeout_s:      float = 60.0
    llm_timeout_s:      float = 30.0
    tts_timeout_s:      float = 15.0
    max_retries:        int   = 3
    retry_backoff_base: float = 0.5

    # Services
    dashboard_port: int = 8765
    metrics_port:   int = 9090

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            vosk_model_path = os.environ.get(
                "VOSK_MODEL_PATH",
                str(Path(__file__).parent / "vosk_model")
            ),
            gemini_api_key  = os.environ.get("GEMINI_API_KEY", ""),
            aws_access_key  = os.environ.get("AWS_ACCESS_KEY_ID", ""),
            aws_secret_key  = os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            aws_region      = os.environ.get("AWS_REGION", "us-east-1"),
        )

    @property
    def chunk_frames(self) -> int:
        return int(self.sample_rate * self.chunk_ms / 1000)

"""llm.py -- Gemini LLM with conversation memory.

Uses google-genai SDK:
    pip uninstall google-generativeai -y
    pip install google-genai
"""

import asyncio
import logging
from typing import AsyncIterator

from config import Config
from latency import LatencyTracker
from memory import ConversationMemory

log = logging.getLogger("llm")


class GeminiLLM:
    """Calls Gemini 2.0 Flash and yields response text. Maintains conversation memory."""

    def __init__(self, cfg: Config, tracker: LatencyTracker,
                 memory: ConversationMemory | None = None):
        self.cfg     = cfg
        self.tracker = tracker
        self.memory  = memory or ConversationMemory(max_turns=10, token_budget=2000)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.cfg.gemini_api_key)
        return self._client

    async def stream(self, transcript: str) -> AsyncIterator[str]:
        client = self._get_client()
        from google.genai import types

        log.info(f"[LLM] calling Gemini with: {transcript!r}")

        # Build clean conversation history from stored turns (no synthetic messages)
        contents = []
        for turn in self.memory._turns:
            role = "model" if turn.role == "assistant" else "user"
            contents.append(
                types.Content(role=role, parts=[types.Part(text=turn.text)])
            )
        # Append current user message
        contents.append(
            types.Content(role="user", parts=[types.Part(text=transcript)])
        )

        self.tracker.mark_llm_start()
        loop = asyncio.get_running_loop()

        def _call():
            return client.models.generate_content(
                model=self.cfg.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=self.cfg.system_prompt,
                    temperature=0.7,
                    max_output_tokens=150,
                ),
            )

        try:
            log.debug("[LLM] sending request to Gemini...")
            response = await loop.run_in_executor(None, _call)
            log.debug("[LLM] response received from Gemini")
        except Exception as exc:
            log.error(f"[LLM] API error: {exc}")
            yield "Sorry, I could not reach the language model."
            return

        # Extract text from response
        text = ""
        try:
            text = response.text or ""
        except Exception:
            for c in getattr(response, "candidates", []):
                for p in getattr(getattr(c, "content", None), "parts", []):
                    text += getattr(p, "text", "")

        text = text.strip()
        if not text:
            log.warning("[LLM] Gemini returned empty response")
            yield "I could not generate a response. Please try again."
            return

        self.tracker.mark_llm_first()
        log.info(f"[LLM] response: {text[:80]!r}{'...' if len(text) > 80 else ''}")

        # Save to memory (add both user turn and assistant response)
        self.memory.add_user(transcript)
        self.memory.add_assistant(text)

        # Yield full response -- TTS speaks it all at once
        yield text

    def reset_memory(self):
        self.memory.clear()
        log.info("[LLM] memory reset")

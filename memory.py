"""memory.py — Multi-turn conversation memory with token budgeting and persistence."""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("memory")

_CHARS_PER_TOKEN = 4  # rough estimate for budget checks


@dataclass
class Turn:
    role:      str
    text:      str
    timestamp: float = field(default_factory=time.time)

    @property
    def token_estimate(self) -> int:
        return max(1, len(self.text) // _CHARS_PER_TOKEN)


class ConversationMemory:
    """
    Sliding-window conversation history for multi-turn LLM context.

    Automatically trims old turns when max_turns or token_budget is exceeded.
    Compresses overflow into a rolling summary via a secondary LLM call.
    Persists to a JSON file between sessions when persist_path is set.
    """

    def __init__(
        self,
        max_turns:         int           = 20,
        token_budget:      int           = 4000,
        summary_threshold: int           = 3000,
        persist_path:      Optional[Path] = None,
    ):
        self.max_turns         = max_turns
        self.token_budget      = token_budget
        self.summary_threshold = summary_threshold
        self.persist_path      = persist_path

        self._turns:   list[Turn]    = []
        self._summary: Optional[str] = None

        if persist_path and persist_path.exists():
            self._load()

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_user(self, text: str):
        self._add(Turn(role="user", text=text.strip()))

    def add_assistant(self, text: str):
        self._add(Turn(role="assistant", text=text.strip()))

    def clear(self):
        self._turns.clear()
        self._summary = None
        if self.persist_path:
            self.persist_path.unlink(missing_ok=True)
        log.info("[Memory] cleared")

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def token_estimate(self) -> int:
        return sum(t.token_estimate for t in self._turns)

    def summary_str(self) -> str:
        base = f"{self.turn_count} turns, ~{self.token_estimate} tokens"
        return base + f", summary: {self._summary[:40]!r}..." if self._summary else base

    # ── LLM message format ─────────────────────────────────────────────────────

    def as_gemini_messages(self) -> list[dict]:
        """Return history in Gemini's role/parts format."""
        msgs = []
        if self._summary:
            msgs += [
                {"role": "user",  "parts": [f"[Context: {self._summary}]"]},
                {"role": "model", "parts": ["Understood."]},
            ]
        for t in self._turns:
            msgs.append({
                "role":  "model" if t.role == "assistant" else "user",
                "parts": [t.text],
            })
        return msgs

    # ── Summarisation ──────────────────────────────────────────────────────────

    async def maybe_summarise(self, llm) -> bool:
        if self.token_estimate < self.summary_threshold:
            return False
        half         = len(self._turns) // 2
        to_summarise = self._turns[:half]
        self._turns  = self._turns[half:]
        transcript   = "\n".join(f"{t.role.upper()}: {t.text}" for t in to_summarise)
        prompt       = (
            "Summarise this conversation in 2-3 sentences, preserving key facts:\n\n"
            + transcript
        )
        parts: list[str] = []
        async for token in llm.stream(prompt):
            parts.append(token)
        new_summary = "".join(parts).strip()
        self._summary = f"{self._summary} {new_summary}" if self._summary else new_summary
        log.info(f"[Memory] summarised: {self._summary[:60]!r}...")
        return True

    # ── Internal ───────────────────────────────────────────────────────────────

    def _add(self, turn: Turn):
        self._turns.append(turn)
        self._trim()
        if self.persist_path:
            self._save()

    def _trim(self):
        while len(self._turns) > self.max_turns:
            self._turns.pop(0)
        while self.token_estimate > self.token_budget and len(self._turns) > 2:
            self._turns.pop(0)

    def _save(self):
        if self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(
                json.dumps({"summary": self._summary,
                            "turns":   [asdict(t) for t in self._turns]}, indent=2)
            )

    def _load(self):
        data          = json.loads(self.persist_path.read_text())
        self._summary = data.get("summary")
        self._turns   = [Turn(**t) for t in data.get("turns", [])]
        log.info(f"[Memory] loaded {len(self._turns)} turns from {self.persist_path}")

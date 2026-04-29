"""hot_config.py — Live config reload by polling a JSON override file."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger("hot_config")

# Type coercions for known keys
_SCHEMA: dict[str, type] = {
    "vad_threshold":       float,
    "vad_hangover_frames": int,
    "asr_timeout_s":       float,
    "llm_timeout_s":       float,
    "tts_timeout_s":       float,
    "max_retries":         int,
    "polly_voice":         str,
    "polly_engine":        str,
    "gemini_model":        str,
    "llm_temperature":     float,
    "llm_max_tokens":      int,
    "active_window_s":     float,
    "debug_logging":       bool,
}

# These require a restart and cannot be hot-swapped
_IMMUTABLE = {"sample_rate", "channels", "chunk_ms",
              "gemini_api_key",
              "aws_access_key", "aws_secret_key"}


class HotConfig:
    """
    Polls a JSON file every poll_interval_s seconds.
    On change, fires per-key listeners and global listeners.
    Writes are atomic (tmp -> replace) to prevent partial reads.
    """

    def __init__(self, path: str | Path = "config_overrides.json",
                 poll_interval_s: float = 2.0):
        self.path             = Path(path)
        self.poll_interval_s  = poll_interval_s
        self._values: dict[str, Any]             = {}
        self._mtime:  float                      = 0.0
        self._listeners: dict[str, list[Callable]] = {}
        self._global: list[Callable]             = []
        self._load()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def set(self, key: str, value: Any, persist: bool = True):
        if key in _IMMUTABLE:
            log.warning(f"[HotConfig] {key!r} is immutable -- restart required")
            return
        old = self._values.get(key)
        self._values[key] = self._coerce(key, value)
        self._fire(key, old, self._values[key])
        if persist:
            self._save()

    def delete(self, key: str):
        if key in self._values:
            old = self._values.pop(key)
            self._fire(key, old, None)
            self._save()

    def on_change(self, key: str, callback: Callable):
        self._listeners.setdefault(key, []).append(callback)

    def on_any_change(self, callback: Callable):
        self._global.append(callback)

    async def watch(self):
        log.info(f"[HotConfig] watching {self.path} every {self.poll_interval_s}s")
        while True:
            await asyncio.sleep(self.poll_interval_s)
            self._reload_if_changed()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self._mtime  = self.path.stat().st_mtime
            self._values = {k: self._coerce(k, v) for k, v in data.items()
                            if k not in _IMMUTABLE}
            log.info(f"[HotConfig] loaded {len(self._values)} overrides")
        except Exception as e:
            log.error(f"[HotConfig] load error: {e}")

    def _reload_if_changed(self):
        if not self.path.exists():
            if self._values:
                old = dict(self._values)
                self._values.clear()
                for k, v in old.items():
                    self._fire(k, v, None)
            return
        if self.path.stat().st_mtime <= self._mtime:
            return
        log.info(f"[HotConfig] change detected")
        old_values = dict(self._values)
        self._load()
        for key in set(old_values) | set(self._values):
            old = old_values.get(key)
            new = self._values.get(key)
            if old != new:
                self._fire(key, old, new)

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._values, indent=2))
        tmp.replace(self.path)
        self._mtime = self.path.stat().st_mtime

    def _coerce(self, key: str, value: Any) -> Any:
        if key not in _SCHEMA:
            return value
        t = _SCHEMA[key]
        try:
            if t is bool:
                return str(value).lower() in ("true", "1", "yes") if isinstance(value, str) else bool(value)
            return t(value)
        except (ValueError, TypeError):
            return value

    def _fire(self, key: str, old: Any, new: Any):
        log.info(f"[HotConfig] {key}: {old!r} -> {new!r}")
        for cb in self._listeners.get(key, []):
            try:
                result = cb(old, new)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                log.error(f"[HotConfig] listener error: {e}")
        for cb in self._global:
            try:
                result = cb(key, old, new)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                log.error(f"[HotConfig] global listener error: {e}")


def parse_cli_overrides(args: list[str]) -> dict[str, str]:
    """Parse ["key=value", ...] from CLI --config arguments."""
    result = {}
    for arg in args:
        if "=" in arg:
            k, _, v = arg.partition("=")
            result[k.strip()] = v.strip()
    return result

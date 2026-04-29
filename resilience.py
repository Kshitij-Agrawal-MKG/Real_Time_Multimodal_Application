"""resilience.py — Timeouts, exponential-backoff retry, and circuit breaker."""

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional

from config import Config

log = logging.getLogger("resilience")


class ResilienceWrapper:
    """Wraps pipeline stage calls with timeout, retry, and circuit-breaker protection."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    async def run_with_timeout(
        self,
        stage: str,
        coro_fn: Callable[[], Awaitable],
        timeout_s: float,
        on_timeout: Optional[Callable] = None,
    ):
        """Run coro_fn() with a hard deadline. Calls on_timeout() if it expires."""
        try:
            await asyncio.wait_for(coro_fn(), timeout=timeout_s)
        except asyncio.TimeoutError:
            log.warning(f"[{stage}] timeout after {timeout_s}s")
            if on_timeout:
                result = on_timeout()
                if asyncio.iscoroutine(result):
                    await result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error(f"[{stage}] error: {exc!r}")
            raise

    async def run_with_retry(
        self,
        stage: str,
        coro_fn: Callable[[], Awaitable],
        fallback_queue: Optional[asyncio.Queue] = None,
        fallback: Optional[str] = None,
    ):
        """Retry coro_fn() with exponential backoff. On exhaustion, push fallback."""
        delay    = self.cfg.retry_backoff_base
        last_exc = None

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                await coro_fn()
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_exc = exc
                log.warning(f"[{stage}] attempt {attempt}/{self.cfg.max_retries} failed -- retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10.0)

        log.error(f"[{stage}] all retries exhausted: {last_exc!r}")
        if fallback and fallback_queue:
            await fallback_queue.put(fallback)

    async def circuit_breaker(
        self,
        stage: str,
        coro_fn: Callable[[], Awaitable],
        failure_threshold: int  = 5,
        reset_after_s: float    = 30.0,
    ):
        """Open-circuit protection: skip calls for reset_after_s after failure_threshold failures."""
        failures: int                    = 0
        open_since: Optional[float]      = None

        if open_since is not None:
            elapsed = time.monotonic() - open_since
            if elapsed < reset_after_s:
                log.warning(f"[{stage}] circuit open -- skipping ({reset_after_s - elapsed:.0f}s left)")
                return
            open_since = None
            log.info(f"[{stage}] circuit half-open")

        try:
            await coro_fn()
            failures = 0
        except Exception as exc:
            failures += 1
            log.error(f"[{stage}] failure {failures}/{failure_threshold}: {exc!r}")
            if failures >= failure_threshold:
                open_since = time.monotonic()
                log.error(f"[{stage}] circuit opened for {reset_after_s}s")
            raise

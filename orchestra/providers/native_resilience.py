"""Bounded retry for native (tool-using) provider calls on transient failures.

Native tool-use turns (Claude Code -> gateway -> Ollama/etc.) are single-shot today: a
transient network / 5xx / rate-limit error raises and crashes the client's tool loop mid-turn.
This wraps the native call in a bounded exponential-backoff retry for transient errors only —
non-transient errors (e.g. a 400 bad request) re-raise immediately. Pure/injectable for testing.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Substrings that mark a retryable transient failure. Providers surface HTTP errors as
# RuntimeError("ollama /api/chat 503: ..."), so we match on status codes and network phrases.
_TRANSIENT_MARKERS = (
    "timeout", "timed out", "connection", "reset by peer", "temporarily unavailable",
    "unavailable", "429", " 500", " 502", " 503", " 504", "bad gateway",
)


def is_transient_error(exc: BaseException) -> bool:
    """True for errors worth retrying (network blips, 5xx, rate limits); False otherwise."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def call_with_retry(
    call_fn: Callable[[], Any],
    *,
    max_attempts: int,
    backoff_ms: int,
    multiplier: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Any:
    """Call ``call_fn`` with bounded retry on transient errors.

    Re-raises immediately on a non-transient error or after the last attempt.
    """
    attempts = max(1, max_attempts)
    for attempt in range(attempts):
        try:
            return call_fn()
        except Exception as exc:
            if attempt >= attempts - 1 or not is_transient_error(exc):
                raise
            delay = (backoff_ms * (multiplier ** attempt)) / 1000.0
            logger.warning(
                "native call transient error (attempt %d/%d): %s; retrying in %.2fs",
                attempt + 1, attempts, exc, delay,
            )
            if delay > 0:
                sleep_fn(delay)
    raise RuntimeError("call_with_retry exhausted without returning")  # pragma: no cover

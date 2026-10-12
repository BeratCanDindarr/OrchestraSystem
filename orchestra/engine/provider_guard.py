"""Provider-level concurrency caps and temporary degradation guards."""
from __future__ import annotations

import threading
import time

from orchestra import config

_LOCK = threading.Lock()
_ACTIVE_COUNTS: dict[str, int] = {}
_CONSECUTIVE_TIMEOUTS: dict[str, int] = {}
_DEGRADED_UNTIL: dict[str, float] = {}

# Expanded failure keywords for circuit-breaker triggering
_FAILURE_KEYWORDS = [
    "timeout",
    "too many requests",
    "rate limit",
    "authentication",
    "auth",
    "connection",
    "connect",
    "internal server error",
    "bad gateway",
    "service unavailable",
    "empty output",
    "content filter",
    "safety",
]

_AUTH_KEYWORDS = [
    "401",
    "403",
    "unauthorized",
    "invalid key",
]


def _provider_limit(provider_name: str) -> int:
    rate_limits = config.rate_limit_config()
    per_provider = rate_limits.get("provider_concurrency", {})
    if isinstance(per_provider, dict) and provider_name in per_provider:
        return int(per_provider[provider_name])
    return int(rate_limits.get("default_concurrency", 2))


def can_use(provider_name: str) -> bool:
    now = time.time()
    with _LOCK:
        degraded_until = _DEGRADED_UNTIL.get(provider_name, 0.0)
        if degraded_until > now:
            return False
        if degraded_until:
            _DEGRADED_UNTIL.pop(provider_name, None)
        active = _ACTIVE_COUNTS.get(provider_name, 0)
        return active < _provider_limit(provider_name)


def start(provider_name: str) -> None:
    with _LOCK:
        _ACTIVE_COUNTS[provider_name] = _ACTIVE_COUNTS.get(provider_name, 0) + 1


def release(provider_name: str) -> None:
    """Give a slot back for a call that never returned (it raised, or its task was cancelled)."""
    with _LOCK:
        _ACTIVE_COUNTS[provider_name] = max(0, _ACTIVE_COUNTS.get(provider_name, 0) - 1)


def finish(provider_name: str, *, returncode: int, output: str = "", stderr: str = "") -> None:
    rate_limits = config.rate_limit_config()
    timeout_threshold = int(rate_limits.get("timeout_circuit_breaker_threshold", 3))
    degraded_seconds = int(rate_limits.get("degraded_seconds", 300))

    combined = f"{output}\n{stderr}".lower()
    stderr_lower = stderr.lower()

    # General failure detection (keywords in combined output). Only a call that failed is
    # read for them: an answer about "authentication" or a "connection timeout" is an answer.
    failure_detected = returncode == 124
    if not failure_detected and returncode != 0:
        failure_detected = any(kw in combined for kw in _FAILURE_KEYWORDS)

    # Auth-failure detection (specific to stderr)
    if not failure_detected:
        failure_detected = any(kw in stderr_lower for kw in _AUTH_KEYWORDS)

    # Empty-output detection: stdout empty but stderr contains errors
    if not failure_detected and not output.strip() and stderr.strip():
        failure_detected = True

    with _LOCK:
        _ACTIVE_COUNTS[provider_name] = max(0, _ACTIVE_COUNTS.get(provider_name, 0) - 1)

        if failure_detected:
            streak = _CONSECUTIVE_TIMEOUTS.get(provider_name, 0) + 1
            _CONSECUTIVE_TIMEOUTS[provider_name] = streak
            if streak >= timeout_threshold:
                _DEGRADED_UNTIL[provider_name] = time.time() + degraded_seconds
        else:
            _CONSECUTIVE_TIMEOUTS[provider_name] = 0

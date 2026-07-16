"""Unit tests for orchestra.providers.native_resilience (transient retry for native calls)."""
from __future__ import annotations

import pytest

from orchestra import config
from orchestra.providers import native_resilience as nr


def test_is_transient_error_classification():
    assert nr.is_transient_error(ConnectionError("boom")) is True
    assert nr.is_transient_error(TimeoutError("slow")) is True
    assert nr.is_transient_error(RuntimeError("ollama /api/chat 503: unavailable")) is True
    assert nr.is_transient_error(RuntimeError("ollama /api/chat 429: rate limit")) is True
    # non-transient client errors are NOT retried
    assert nr.is_transient_error(RuntimeError("ollama /api/chat 400: bad request")) is False
    assert nr.is_transient_error(ValueError("malformed tool schema")) is False


def test_call_with_retry_returns_on_first_success():
    calls = []
    out = nr.call_with_retry(
        lambda: calls.append(1) or "ok",
        max_attempts=3, backoff_ms=10, multiplier=2.0, sleep_fn=lambda s: None,
    )
    assert out == "ok" and len(calls) == 1


def test_call_with_retry_recovers_after_transient():
    seq = iter([RuntimeError("connection reset by peer"), "recovered"])

    def flaky():
        v = next(seq)
        if isinstance(v, Exception):
            raise v
        return v

    slept = []
    out = nr.call_with_retry(
        flaky, max_attempts=3, backoff_ms=10, multiplier=2.0, sleep_fn=slept.append,
    )
    assert out == "recovered" and len(slept) == 1


def test_call_with_retry_reraises_non_transient_immediately():
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("ollama /api/chat 400: bad request")

    with pytest.raises(RuntimeError, match="400"):
        nr.call_with_retry(boom, max_attempts=3, backoff_ms=10, multiplier=2.0, sleep_fn=lambda s: None)
    assert len(calls) == 1                      # no retry on non-transient


def test_call_with_retry_exhausts_attempts_on_persistent_transient():
    calls = []

    def always_503():
        calls.append(1)
        raise RuntimeError("service unavailable 503")

    with pytest.raises(RuntimeError, match="503"):
        nr.call_with_retry(always_503, max_attempts=3, backoff_ms=1, multiplier=2.0, sleep_fn=lambda s: None)
    assert len(calls) == 3                      # exactly max_attempts tries


def test_native_retry_disabled_by_default():
    assert config.retry_native_enabled() is False

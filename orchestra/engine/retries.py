"""Retry, backoff, and fallback orchestration with streaming support."""
from __future__ import annotations

import time
import os
from dataclasses import dataclass, field
from threading import Event
from typing import Callable

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from orchestra import config
from orchestra.engine.process_group import ProcessGroupManager, run_provider_process
from orchestra.providers.base import BaseProvider
from orchestra.providers.fallback import resolve_with_fallback

console = Console()

_RETRYABLE_OUTPUT_MARKERS = (
    "rate limit",
    "quota",
    "token limit",
    "credit balance is too low",
    "usage limit",
    "try again later",
    "too many requests",
    "overloaded",
    "temporarily unavailable",
    "authentication error",
    "invalid api key",
    "expired",
)

@dataclass
class RetryPolicy:
    max_attempts: int = 3
    backoff_seconds: list[float] = field(default_factory=lambda: [5.0, 10.0, 20.0])
    retry_on_exit_codes: list[int] = field(default_factory=lambda: [1])


def _should_retry_output(output: str, stderr: str) -> bool:
    combined = f"{output}\n{stderr}".lower()
    if not combined.strip():
        return True
    return any(marker in combined for marker in _RETRYABLE_OUTPUT_MARKERS)

def run_with_retry(
    provider: BaseProvider,
    effort: str,
    prompt: str,
    policy: RetryPolicy,
    *,
    alias: str | None = None,
    timeout: int | None = None,
    process_manager: ProcessGroupManager | None = None,
    pid_callback: Callable[[int | None], None] | None = None,
    attempt_callback: Callable[[BaseProvider, str, int], None] | None = None,
    cancel_event: Event | None = None,
    cancel_check: Callable[[], bool] | None = None,
    stream_callback: Callable[[str], None] | None = None, # 🚀 NEW
    cwd: str | os.PathLike | None = None,
) -> tuple[str, int]:
    effective_timeout = timeout if timeout is not None else config.timeout()
    last_output = ""
    last_returncode = 1

    for attempt in range(policy.max_attempts):
        if (cancel_event and cancel_event.is_set()) or (cancel_check and cancel_check()):
            return last_output or "[CANCELLED]", 130

        current_provider = provider
        current_effort = effort
        if alias is not None:
            current_provider, current_effort = resolve_with_fallback(alias, attempt=attempt)

        if attempt_callback:
            attempt_callback(current_provider, current_effort, attempt)

        output, returncode, stderr = run_provider_process(
            current_provider,
            prompt,
            current_effort,
            timeout=effective_timeout,
            process_manager=process_manager,
            pid_callback=pid_callback,
            stream_callback=stream_callback, # 🚀 PASS THROUGH
            cwd=cwd,
        )
        should_retry = returncode in policy.retry_on_exit_codes or _should_retry_output(output, stderr)
        if returncode == 0 and should_retry:
            last_output = output or stderr or last_output
            last_returncode = 1
        else:
            last_output, last_returncode = output, returncode

        if returncode == 0 and not should_retry:
            return output, returncode
        if attempt >= policy.max_attempts - 1: return last_output, last_returncode
        time.sleep(policy.backoff_seconds[min(attempt, len(policy.backoff_seconds)-1)])

    return last_output, last_returncode

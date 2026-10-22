"""Retry, backoff, and fallback orchestration with streaming support."""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Callable

from rich.console import Console

from orchestra import config
from orchestra.engine import provider_guard
from orchestra.engine.process_group import ProcessGroupManager
from orchestra.providers.base import BaseProvider, error_detail
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

def _default_backoff_seconds() -> list[float]:
    """Compute backoff sequence from config, preferring explicit backoff_seconds."""
    retry_cfg = config.retry_config()
    explicit = retry_cfg.get("backoff_seconds")
    if explicit is not None:
        return [float(v) for v in explicit]

    backoff_ms = retry_cfg.get("backoff_ms", 100)
    if isinstance(backoff_ms, list):
        return [float(v) / 1000.0 for v in backoff_ms]

    multiplier = float(retry_cfg.get("backoff_multiplier", 2.0))
    max_attempts = int(retry_cfg.get("max_attempts", 3))
    return [float(backoff_ms) * (multiplier ** i) / 1000.0 for i in range(max_attempts)]


@dataclass
class RetryPolicy:
    max_attempts: int = field(default_factory=lambda: int(config.retry_config().get("max_attempts", 3)))
    backoff_seconds: list[float] = field(default_factory=_default_backoff_seconds)
    retry_on_exit_codes: list[int] = field(default_factory=lambda: [int(v) for v in config.retry_config().get("retry_on_exit_codes", [1])])


ERROR_MESSAGE_CHARS = 400
FAILURE_REASON_CHARS = 300


def failure_reason(returncode: int, output: str) -> str:
    """'exit code N: <what the CLI said>': the code alone said nothing about why it failed."""
    said = error_detail((output or "").replace("[ERROR]", "")).replace("\n", " ").strip()
    return f"exit code {returncode}: {said[:FAILURE_REASON_CHARS]}" if said else f"exit code {returncode}"


def _should_retry_output(output: str, stderr: str, returncode: int = 1) -> bool:
    """Whether the output says the provider failed in a way another attempt may fix.

    A CLI sometimes prints "rate limit" or "credit balance is too low" and still exits 0, so
    the words are looked for after a success too, but only in a short output: a real answer
    that mentions an "overloaded method" or an "expired token" is an answer, not an error.
    """
    combined = f"{output}\n{stderr}".lower()
    if not combined.strip():
        return True
    if returncode == 0 and len(output) > ERROR_MESSAGE_CHARS:
        return False
    return any(marker in combined for marker in _RETRYABLE_OUTPUT_MARKERS)


async def run_with_retry_async(
    provider: BaseProvider,
    effort: str,
    prompt: str,
    policy: RetryPolicy,
    *,
    alias: str | None = None,
    timeout: int | None = None,
    process_manager: ProcessGroupManager | None = None,
    pid_callback: Callable[[int | None], None] | None = None,
    attempt_callback: Callable[[BaseProvider, str, int, str], None] | None = None,
    cancel_event: asyncio.Event | None = None,
    cancel_check: Callable[[], bool] | None = None,
    stream_callback: Callable[[str], None] | None = None,
    cwd: str | os.PathLike | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> tuple[str, int]:
    effective_timeout = timeout if timeout is not None else config.timeout()
    last_output = ""
    last_returncode = 1
    previous_failure = ""   # why the attempt before this one failed, for the log

    for attempt in range(policy.max_attempts):
        if (cancel_event and cancel_event.is_set()) or (cancel_check and cancel_check()):
            return last_output or "[CANCELLED]", 130

        current_provider = provider
        current_effort = effort
        if alias is not None:
            current_provider, current_effort = resolve_with_fallback(alias, attempt=attempt)

        if attempt_callback:
            attempt_callback(current_provider, current_effort, attempt, previous_failure)

        provider_guard.start(current_provider.name)
        current_provider.reset_usage()
        try:
            output, returncode = await current_provider.async_run(
                prompt,
                current_effort,
                timeout=effective_timeout,
                process_manager=process_manager,
                pid_callback=pid_callback,
                stream_callback=stream_callback,
                cwd=cwd,
                temperature=temperature,
                seed=seed,
            )
        except Exception as exc:
            # A provider that raises is a failed attempt like any other: the chain goes on, and
            # the other agents of the run keep their answers.
            output, returncode = f"[ERROR] {type(exc).__name__}: {exc}", 1
        except BaseException:  # CancelledError included: the slot must not stay taken
            provider_guard.release(current_provider.name)
            raise
        provider_guard.finish(current_provider.name, returncode=returncode, output=output, stderr="")
        if returncode == 0 and current_provider.reports_errors:
            should_retry = not output.strip()   # its errors never exit 0; an answer is an answer
        else:
            should_retry = returncode in policy.retry_on_exit_codes or _should_retry_output(output, "", returncode)
        if returncode == 0 and should_retry:
            last_output = output or last_output
            last_returncode = 1
        else:
            last_output, last_returncode = output, returncode

        if returncode == 0 and not should_retry:
            return output, returncode
        if attempt >= policy.max_attempts - 1:
            return last_output, last_returncode
        # The next attempt goes on, whatever the failure was (a timeout too): its log says why.
        previous_failure = failure_reason(returncode, output) if returncode else "no answer, or an error message"
        await asyncio.sleep(policy.backoff_seconds[min(attempt, len(policy.backoff_seconds) - 1)])

    return last_output, last_returncode

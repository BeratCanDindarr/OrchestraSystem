"""Parallel agent execution with live Rich status table.

This package splits the original ``parallel.py`` module by responsibility:

* :mod:`orchestra.engine.parallel.ui` - Rich status-table rendering.
* :mod:`orchestra.engine.parallel.budget` - cost/metrics/budget enforcement.
* :mod:`orchestra.engine.parallel.executor` - run cancellation helpers.

There is one agent runner, ``_run_one_agent_async``: ``run_parallel`` runs it with
``asyncio.run``, on a loop of its own in a worker thread when the caller is already inside
one. It lives in the package namespace because the test suite patches
``orchestra.engine.parallel.<name>`` (``resolve``, ``run_with_retry_async``, ``artifacts``).
"""
from __future__ import annotations

import asyncio
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Callable, Optional, Any

from rich.live import Live

from orchestra import config
from orchestra.engine import artifacts, costing
from orchestra.engine.process_group import ProcessGroupManager
from orchestra.engine.retries import RetryPolicy, failure_reason as _failure_reason, run_with_retry_async
from orchestra.engine.pii_scrubber import PiiScrubber
from orchestra.engine.validator import validate_agent_output
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus
from orchestra.normalize import infer_confidence, is_soft_failure, normalize
from orchestra.providers.fallback import resolve
from orchestra.state import FailureKind, FailureState, InterruptState
from orchestra.engine.trace import get_tracer

from .ui import console, _make_table
from .budget import (
    _now,
    _estimate_tokens,
    _estimate_cost_usd,
    _daily_spend_usd,
    _refresh_run_metrics,
    _enforce_budget_guard_async,
)
from .executor import _mark_run_cancelled_async

__all__ = [
    "run_parallel",
    "run_parallel_async",
    "_run_one_agent_async",
    "_mark_run_cancelled_async",
    "_make_table",
    "console",
    "_now",
    "_estimate_tokens",
    "_estimate_cost_usd",
    "_daily_spend_usd",
    "_refresh_run_metrics",
    "_enforce_budget_guard_async",
    "resolve",
    "run_with_retry_async",
    "artifacts",
    "get_tracer",
]


def _own_provider(alias: str) -> Optional[Any]:
    """The provider the alias itself names, whatever is answering for it right now."""
    try:
        from orchestra.providers.fallback import ALIAS_TABLE, _canonical_alias, _is_backend_ref
        if _is_backend_ref(alias):
            return None
        return ALIAS_TABLE[_canonical_alias(alias.split("#", 1)[0])][0]
    except Exception:
        return None


async def _run_one_agent_async(
    run: OrchestraRun,
    alias: str,
    prompt: str,
    lock: threading.Lock,
    process_manager: ProcessGroupManager,
    cancel_event: asyncio.Event,
    tracer: Optional[Any] = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    temperature: float | None = None,
    seed: int | None = None,
    cwd: str | None = None,
) -> AgentRun | None:
    """Run a single agent asynchronously, update run.agents, write artifacts."""
    provider, effort = resolve(alias)
    span = tracer.start_span(f"agent:{alias}", metadata={"model": provider.model_label(effort)}) if tracer else None
    max_tokens_per_agent = int(config.token_budget_config().get("max_tokens_per_agent", 12000))
    prompt_tokens = _estimate_tokens(prompt)

    with lock:
        agent = AgentRun(
            alias=alias,
            provider=provider.name,
            model=provider.model_label(effort),
            status=AgentStatus.STARTED,
            start_time=_now(),
        )
        run.agents.append(agent)
        artifacts.append_event(
            run.run_id,
            {
                "event": "agent_started",
                "alias": alias,
                "model": agent.model,
            },
        )
        artifacts.write_manifest(run)
        from orchestra.engine.replay import get_recorder
        get_recorder(run.run_id).record_agent_dispatch(
            alias, prompt, agent.model, provider.name
        )

    if max_tokens_per_agent > 0 and prompt_tokens > max_tokens_per_agent:
        with lock:
            agent.status = AgentStatus.FAILED
            agent.end_time = _now()
            agent.error = f"context_limit_exceeded: {prompt_tokens} > {max_tokens_per_agent}"
            agent.failure_kind = FailureKind.BUDGET_EXCEEDED.value
            run.failure = FailureState(
                kind=FailureKind.BUDGET_EXCEEDED.value,
                message=agent.error,
                retryable=False,
                source="context_guard",
                agent_alias=alias,
            )
            artifacts.append_event(
                run.run_id,
                {"event": "agent_failed", "alias": alias, "error": agent.error},
            )
            artifacts.write_manifest(run)
        if span:
            span.finish(status="failed")
        return agent

    def _set_pid(pid: int | None) -> None:
        with lock:
            agent.pid = pid
            artifacts.write_manifest(run)

    last_provider = provider  # Track actual provider used (may change on fallback)
    requested_provider = _own_provider(alias) or provider

    def _on_attempt(current_provider, current_effort: str, attempt: int, after: str = "") -> None:
        nonlocal last_provider
        last_provider = current_provider
        with lock:
            agent.provider = current_provider.name
            agent.model = current_provider.model_label(current_effort)
            if current_provider is not requested_provider:
                # Another model is answering under this alias: the log must say so.
                artifacts.append_event(
                    run.run_id,
                    {"event": "agent_fallback", "alias": alias, "requested": requested_provider.name,
                     "model": agent.model},
                )
            if attempt > 0:
                artifacts.append_event(
                    run.run_id,
                    {
                        "event": "agent_retry",
                        "alias": alias,
                        "attempt": attempt + 1,
                        "model": agent.model,
                        "after": after,   # why the attempt before failed
                    },
                )
            artifacts.write_manifest(run)

    def _is_externally_cancelled() -> bool:
        manifest = artifacts.load_manifest(run.run_id)
        return bool(manifest and manifest.get("status") == RunStatus.CANCELLED.value)

    def _scrubbed_stream_callback(chunk: str) -> None:
        scrubbed = PiiScrubber.scrub_text(chunk)
        if stream_callback:
            stream_callback(scrubbed)
        try:
            from orchestra import live_status
            live_status.note_stream(run.run_id, alias, scrubbed)
        except Exception as _exc:
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "live_status.note_stream failed for run=%s alias=%s: %s",
                run.run_id, alias, _exc,
            )

    try:
        stdout, returncode = await run_with_retry_async(
            provider,
            effort,
            prompt,
            RetryPolicy(),
            alias=alias,
            timeout=config.timeout_for_alias(alias),
            process_manager=process_manager,
            pid_callback=_set_pid,
            attempt_callback=_on_attempt,
            cancel_event=cancel_event,
            cancel_check=_is_externally_cancelled,
            stream_callback=_scrubbed_stream_callback,
            temperature=temperature,
            seed=seed,
            cwd=cwd,
        )
    except asyncio.CancelledError:
        with lock:
            agent.status = AgentStatus.CANCELLED
            agent.error = "cancelled by user"
            agent.pid = None
            agent.end_time = _now()
            artifacts.append_event(run.run_id, {"event": "agent_cancelled", "alias": alias})
            artifacts.write_manifest(run)
        if span:
            span.finish(status="cancelled")
        raise

    externally_cancelled = _is_externally_cancelled()

    with lock:
        if externally_cancelled:
            run.status = RunStatus.CANCELLED
            run.interrupt_state = InterruptState.CANCELLED
            run.failure = FailureState(
                kind=FailureKind.UNKNOWN_RUNTIME_ERROR.value,
                message="Run cancelled externally",
                retryable=False,
                source="parallel",
                agent_alias=alias,
            )
        agent.pid = None
        agent.end_time = _now()
        
        status_label = "completed"
        if cancel_event.is_set() or run.status == RunStatus.CANCELLED or externally_cancelled:
            agent.status = AgentStatus.CANCELLED
            agent.error = "cancelled by user"
            artifacts.append_event(run.run_id, {"event": "agent_cancelled", "alias": alias})
            status_label = "cancelled"
        elif returncode != 0:
            agent.status = AgentStatus.FAILED
            agent.error = _failure_reason(returncode, stdout)
            agent.failure_kind = FailureKind.MODEL_TIMEOUT.value if returncode == 124 else FailureKind.TOOL_EXECUTION_FAILED.value
            agent.stdout_log = stdout
            reported = last_provider.last_usage()
            if isinstance(reported, dict) and reported:
                # A call that failed after the model worked was still paid for.
                costing.charge(agent, agent.model, reported, prompt=prompt, output=stdout)
            run.failure = FailureState(
                kind=FailureKind.MODEL_TIMEOUT.value if returncode == 124 else FailureKind.TOOL_EXECUTION_FAILED.value,
                message=agent.error,
                retryable=returncode != 124,
                source=agent.provider,
                agent_alias=alias,
            )
            if stdout:
                artifacts.write_agent_log(run.run_id, alias, stdout)
                artifacts.write_normalized(run.run_id, alias, normalize(stdout, agent.provider))
            artifacts.append_event(
                run.run_id,
                {"event": "agent_failed", "alias": alias, "error": agent.error},
            )
            status_label = "failed"
        else:
            agent.status = AgentStatus.COMPLETED
            agent.stdout_log = stdout
            reported = last_provider.last_usage()
            costing.charge(agent, agent.model, reported if isinstance(reported, dict) else None,
                           prompt=prompt, output=stdout)
            agent.confidence = infer_confidence(stdout)
            agent.soft_failed = is_soft_failure(stdout)
            validation = validate_agent_output(stdout, soft_failed=agent.soft_failed)
            agent.validation_status = validation.status
            agent.validation_reason = validation.reason
            if agent.soft_failed:
                agent.failure_kind = FailureKind.MODEL_SOFT_FAILURE.value
                run.failure = FailureState(
                    kind=FailureKind.MODEL_SOFT_FAILURE.value,
                    message="Agent output matched soft-failure heuristics",
                    retryable=True,
                    source=agent.provider,
                    agent_alias=alias,
                )
            artifacts.write_agent_log(run.run_id, alias, stdout)
            artifacts.write_normalized(run.run_id, alias, normalize(stdout, agent.provider))
            artifacts.append_event(
                run.run_id,
                {
                    "event": "agent_completed",
                    "alias": alias,
                    "elapsed": agent.elapsed,
                    "confidence": agent.confidence,
                    "soft_failed": agent.soft_failed,
                    "validation_status": agent.validation_status,
                    "validation_reason": agent.validation_reason,
                    "estimated_completion_tokens": agent.estimated_completion_tokens,
                    "estimated_cost_usd": agent.estimated_cost_usd,
                },
            )
        
        if span:
            span.finish(status=status_label)

        _refresh_run_metrics(run)

    await _enforce_budget_guard_async(run, lock, process_manager, cancel_event)

    with lock:
        from orchestra.engine.replay import get_recorder
        get_recorder(run.run_id).record_agent_completed(
            alias, agent.stdout_log or "", returncode
        )
        artifacts.write_manifest(run)

    return agent


def _patience_seconds(run: OrchestraRun, agent_count: int) -> float:
    """How long a dual run waits for the other agents once one has answered (`[dual]
    patience_seconds`, default 180; 0 waits for all). A dual took 120 s on average, 318 s at most."""
    if run.mode != "dual" or agent_count < 2:
        return 0.0
    return float(config.section("dual").get("patience_seconds", 180) or 0)


async def _collect(tasks, results: list, cancel_event: asyncio.Event, patience: float, *, on_result, lock) -> list:
    """Gather the agents' results; returns the tasks still running when the patience ran out."""
    loop = asyncio.get_running_loop()
    pending, deadline = set(tasks), None
    while pending:
        timeout = None if deadline is None else max(0.0, deadline - loop.time())
        done, pending = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if not done:
            return list(pending)
        for task in done:
            if task.cancelled() or task.exception() is not None:
                continue
            if task.result() is not None:
                results.append(task.result())
        if on_result is not None:
            with lock:
                on_result()
        if cancel_event.is_set():   # the budget guard stopped the run
            for task in pending:
                task.cancel()
            return []
        if deadline is None and patience > 0:
            deadline = loop.time() + patience
    return []


def _mark_late(run: OrchestraRun, lock: threading.Lock, aliases: list[str], patience: float) -> None:
    """Record the agents cut for lateness: their run goes on with the answers it has."""
    with lock:
        for agent in run.agents:
            if agent.alias in aliases and agent.status != AgentStatus.COMPLETED:
                agent.status = AgentStatus.CANCELLED
                agent.failure_kind = "LATE"
                agent.error = f"late: no answer {patience:g} s after the first agent's ([dual] patience_seconds)"
                agent.pid = None
                agent.end_time = agent.end_time or _now()
        for alias in aliases:
            artifacts.append_event(run.run_id, {"event": "agent_cancelled", "alias": alias, "reason": "late"})
        artifacts.write_manifest(run)


async def run_parallel_async(
    run: OrchestraRun,
    agents: list[tuple[str, str]],
    show_live: bool = True,
    install_signal_handlers: bool = True,
    stream_callback: Optional[Callable[[str], None]] = None,
    emit_console: bool = True,
    **_kwargs: Any,
) -> list[AgentRun]:
    """
    Asynchronous implementation using asyncio.gather().
    """
    lock = threading.Lock()
    cancel_event = asyncio.Event()
    process_manager = ProcessGroupManager()
    results: list[AgentRun] = []
    tracer = get_tracer(run.run_id)
    live_refresh_per_second = int(config.execution_config().get("live_refresh_per_second", 4))
    previous_handler = signal.getsignal(signal.SIGINT) if install_signal_handlers else None

    def _handle_sigint(signum, frame) -> None:
        del signum, frame
        if cancel_event.is_set():
            return
        cancel_event.set()
        console.print("\n[yellow]Ctrl+C received — cancelling active agent processes...[/yellow]")
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(lambda: asyncio.create_task(process_manager.kill_all_async()))
        loop.call_soon_threadsafe(lambda: asyncio.create_task(_mark_run_cancelled_async(run, lock, "cancelled by user")))

    if install_signal_handlers:
        signal.signal(signal.SIGINT, _handle_sigint)

    tasks = [
        asyncio.create_task(
            _run_one_agent_async(
                run,
                alias,
                prompt,
                lock,
                process_manager,
                cancel_event,
                tracer=tracer,
                stream_callback=stream_callback,
                temperature=_kwargs.get("temperature"),
                seed=_kwargs.get("seed"),
                cwd=_kwargs.get("cwd"),
            )
        )
        for alias, prompt in agents
    ]

    alias_of = {task: alias for task, (alias, _) in zip(tasks, agents)}
    patience = _patience_seconds(run, len(tasks))

    try:
        if show_live:
            with Live(_make_table(run), console=console, refresh_per_second=live_refresh_per_second) as live:
                late = await _collect(tasks, results, cancel_event, patience,
                                      on_result=lambda: live.update(_make_table(run)), lock=lock)
        else:
            late = await _collect(tasks, results, cancel_event, patience, on_result=None, lock=lock)
        if late:
            # Cancel first, then kill: killed first, a late agent's retry loop took the killed CLI
            # for a failed attempt and started a new one, which then ran on after the cancel.
            for task in late:
                task.cancel()
            await process_manager.kill_all_async()   # only the late ones are still running
            await asyncio.gather(*late, return_exceptions=True)
            _mark_late(run, lock, [alias_of[task] for task in late], patience)
    except KeyboardInterrupt:
        cancel_event.set()
        await process_manager.kill_all_async()
        await _mark_run_cancelled_async(run, lock, "cancelled by user")
    finally:
        if install_signal_handlers and previous_handler is not None:
            signal.signal(signal.SIGINT, previous_handler)
        # Ensure any remaining tasks are cancelled and awaited
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    return results


def run_parallel(
    run: OrchestraRun,
    agents: list[tuple[str, str]],   # [(alias, prompt), ...]
    show_live: bool = True,
    install_signal_handlers: bool = True,
    stream_callback: Optional[Callable[[str], None]] = None,
    emit_console: bool = True,
    **_kwargs: object,
) -> list[AgentRun]:
    """
    Run multiple agents in parallel.
    agents: list of (alias, prompt) tuples.
    Returns list of completed AgentRun objects.
    """
    def run_async_core(handle_signals: bool) -> list[AgentRun]:
        return asyncio.run(run_parallel_async(
            run, agents,
            show_live=show_live,
            install_signal_handlers=handle_signals,
            stream_callback=stream_callback,
            emit_console=emit_console,
            **_kwargs,
        ))

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return run_async_core(install_signal_handlers)
    # Called from inside an event loop: the same async core, on a loop of its own in a worker
    # thread (signals belong to the main thread). There is one agent runner, not two.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(copy_context().run, run_async_core, False).result()

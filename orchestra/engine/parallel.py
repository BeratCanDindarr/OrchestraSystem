"""Parallel agent execution with live Rich status table."""
from __future__ import annotations

import asyncio
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from datetime import datetime, timezone
from typing import Callable, Optional, Any

from rich.console import Console
from rich.live import Live
from rich.table import Table

from orchestra import config
from orchestra.engine import artifacts
from orchestra.engine.process_group import ProcessGroupManager
from orchestra.engine.retries import RetryPolicy, run_with_retry, run_with_retry_async
from orchestra.engine.pii_scrubber import PiiScrubber
from orchestra.engine.validator import validate_agent_output
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus
from orchestra.normalize import infer_confidence, is_soft_failure, normalize
from orchestra.providers.fallback import resolve
from orchestra.state import FailureKind, FailureState, InterruptState
from orchestra.engine.trace import get_tracer

console = Console()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_table(run: OrchestraRun) -> Table:
    table = Table(
        title=f"[bold cyan]run {run.run_id}[/bold cyan]  [dim]{run.mode}[/dim]",
        box=None,
        pad_edge=False,
        show_header=True,
    )
    table.add_column("AGENT", style="cyan", width=12)
    table.add_column("MODEL", style="white", width=22)
    table.add_column("STATE", width=11)
    table.add_column("ELAPSED", width=8)
    table.add_column("LAST", style="dim", no_wrap=True)

    colors = {
        AgentStatus.QUEUED: "yellow",
        AgentStatus.STARTED: "blue",
        AgentStatus.COMPLETED: "green",
        AgentStatus.FAILED: "red",
        AgentStatus.CANCELLED: "dim",
    }
    for agent in run.agents:
        color = colors.get(agent.status, "white")
        table.add_row(
            agent.alias,
            agent.model,
            f"[{color}]{agent.status.value}[/{color}]",
            agent.elapsed,
            agent.last_line,
        )
    return table


def _mark_run_cancelled(run: OrchestraRun, lock: threading.Lock, reason: str) -> None:
    with lock:
        already_cancelled = run.status == RunStatus.CANCELLED
        run.status = RunStatus.CANCELLED
        run.interrupt_state = InterruptState.CANCEL_REQUESTED
        run.updated_at = _now()
        for agent in run.agents:
            if agent.status in (AgentStatus.QUEUED, AgentStatus.STARTED):
                agent.status = AgentStatus.CANCELLED
                agent.end_time = agent.end_time or run.updated_at
                agent.error = reason
                agent.pid = None
        artifacts.write_manifest(run)
        if not already_cancelled:
            artifacts.append_event(run.run_id, {"event": "run_cancelled", "reason": reason})


async def _mark_run_cancelled_async(run: OrchestraRun, lock: threading.Lock, reason: str) -> None:
    with lock:
        already_cancelled = run.status == RunStatus.CANCELLED
        run.status = RunStatus.CANCELLED
        run.interrupt_state = InterruptState.CANCEL_REQUESTED
        run.updated_at = _now()
        for agent in run.agents:
            if agent.status in (AgentStatus.QUEUED, AgentStatus.STARTED):
                agent.status = AgentStatus.CANCELLED
                agent.end_time = agent.end_time or run.updated_at
                agent.error = reason
                agent.pid = None
        artifacts.write_manifest(run)
        if not already_cancelled:
            artifacts.append_event(run.run_id, {"event": "run_cancelled", "reason": reason})


def _estimate_tokens(text: str) -> int:
    from orchestra.engine.context_compression import estimate_tokens
    return estimate_tokens(text)


def _estimate_cost_usd(model: str, estimated_tokens: int) -> float:
    if estimated_tokens <= 0:
        return 0.0
    return round((estimated_tokens / 1000.0) * config.price_per_1k_tokens(model), 6)


def _daily_spend_usd(exclude_run_id: str) -> float:
    from orchestra.storage.db import get_db

    today = datetime.now(timezone.utc).date().isoformat()
    connection = get_db()
    try:
        row = connection.execute(
            """
            SELECT COALESCE(SUM(total_cost_usd), 0.0)
            FROM runs
            WHERE substr(created_at, 1, 10) = ? AND run_id != ?
            """,
            (today, exclude_run_id),
        ).fetchone()
        return float(row[0] or 0.0) if row else 0.0
    finally:
        connection.close()


def _refresh_run_metrics(run: OrchestraRun) -> None:
    completed = [agent for agent in run.agents if agent.status == AgentStatus.COMPLETED]
    run.total_cost_usd = round(sum(agent.estimated_cost_usd for agent in completed), 6)
    run.avg_confidence = round(
        sum(agent.confidence for agent in completed) / len(completed),
        3,
    ) if completed else 0.0


def _enforce_budget_guard(run: OrchestraRun, lock: threading.Lock, process_manager: ProcessGroupManager, cancel_event: threading.Event) -> None:
    from orchestra.monitoring.cost_monitor import _check_budget
    ok, reason = _check_budget(run)
    max_budget = float(config.max_budget_usd())
    daily_budget = float(config.token_budget_config().get("daily_budget_usd", 0.0))
    historical_daily_spend = _daily_spend_usd(run.run_id) if daily_budget > 0 else 0.0
    exceeds_daily_budget = daily_budget > 0 and (historical_daily_spend + run.total_cost_usd) >= daily_budget
    if ok and not exceeds_daily_budget:
        return

    cancel_event.set()
    process_manager.kill_all()

    if not ok:
        logger.warning(f"Budget enforcement: {reason}")
        message = reason
    else:
        message = f"Daily budget exceeded: ${historical_daily_spend + run.total_cost_usd:.4f} >= ${daily_budget:.4f}"
        logger.warning(f"Budget enforcement: {message}")

    run.status = RunStatus.FAILED
    run.interrupt_state = InterruptState.CANCEL_REQUESTED
    run.failure = FailureState(
        kind=FailureKind.BUDGET_EXCEEDED.value,
        message=message,
        retryable=False,
        source="budget_guard",
    )
    run.updated_at = _now()

    for agent in run.agents:
        if agent.status in (AgentStatus.QUEUED, AgentStatus.STARTED):
            agent.status = AgentStatus.CANCELLED
            agent.end_time = agent.end_time or run.updated_at
            agent.error = "budget_exceeded"
            agent.pid = None

    artifacts.append_event(
        run.run_id,
        {
            "event": "budget_exceeded",
            "reason": message,
            "total_cost_usd": run.total_cost_usd,
            "max_budget_usd": max_budget,
            "daily_budget_usd": daily_budget,
            "historical_daily_spend_usd": round(historical_daily_spend, 6),
        },
    )


async def _enforce_budget_guard_async(run: OrchestraRun, lock: threading.Lock, process_manager: ProcessGroupManager, cancel_event: asyncio.Event) -> None:
    from orchestra.monitoring.cost_monitor import _check_budget
    ok, reason = _check_budget(run)
    max_budget = float(config.max_budget_usd())
    daily_budget = float(config.token_budget_config().get("daily_budget_usd", 0.0))
    historical_daily_spend = _daily_spend_usd(run.run_id) if daily_budget > 0 else 0.0
    exceeds_daily_budget = daily_budget > 0 and (historical_daily_spend + run.total_cost_usd) >= daily_budget
    if ok and not exceeds_daily_budget:
        return

    cancel_event.set()
    await process_manager.kill_all_async()

    if not ok:
        logger.warning(f"Budget enforcement: {reason}")
        message = reason
    else:
        message = f"Daily budget exceeded: ${historical_daily_spend + run.total_cost_usd:.4f} >= ${daily_budget:.4f}"
        logger.warning(f"Budget enforcement: {message}")

    run.status = RunStatus.FAILED
    run.interrupt_state = InterruptState.CANCEL_REQUESTED
    run.failure = FailureState(
        kind=FailureKind.BUDGET_EXCEEDED.value,
        message=message,
        retryable=False,
        source="budget_guard",
    )
    run.updated_at = _now()

    for agent in run.agents:
        if agent.status in (AgentStatus.QUEUED, AgentStatus.STARTED):
            agent.status = AgentStatus.CANCELLED
            agent.end_time = agent.end_time or run.updated_at
            agent.error = "budget_exceeded"
            agent.pid = None

    artifacts.append_event(
        run.run_id,
        {
            "event": "budget_exceeded",
            "reason": message,
            "total_cost_usd": run.total_cost_usd,
            "max_budget_usd": max_budget,
            "daily_budget_usd": daily_budget,
            "historical_daily_spend_usd": round(historical_daily_spend, 6),
        },
    )


def _run_one_agent(
    run: OrchestraRun,
    alias: str,
    prompt: str,
    lock: threading.Lock,
    process_manager: ProcessGroupManager,
    cancel_event: threading.Event,
    tracer: Optional[Any] = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> AgentRun:
    """Run a single agent, update run.agents, write artifacts."""
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

    def _on_attempt(current_provider, current_effort: str, attempt: int) -> None:
        nonlocal last_provider
        last_provider = current_provider
        with lock:
            agent.provider = current_provider.name
            agent.model = current_provider.model_label(current_effort)
            if attempt > 0:
                artifacts.append_event(
                    run.run_id,
                    {
                        "event": "agent_retry",
                        "alias": alias,
                        "attempt": attempt + 1,
                        "model": agent.model,
                    },
                )
            artifacts.write_manifest(run)

    def _is_externally_cancelled() -> bool:
        manifest = artifacts.load_manifest(run.run_id)
        return bool(manifest and manifest.get("status") == RunStatus.CANCELLED.value)

    def _scrubbed_stream_callback(chunk: str) -> None:
        if stream_callback:
            stream_callback(PiiScrubber.scrub_text(chunk))

    stdout, returncode = run_with_retry(
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
    )

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
            agent.error = f"exit code {returncode}"
            agent.failure_kind = FailureKind.MODEL_TIMEOUT.value if returncode == 124 else FailureKind.TOOL_EXECUTION_FAILED.value
            agent.stdout_log = stdout
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
            usage = last_provider.last_usage()
            if usage is not None:
                agent.estimated_completion_tokens = usage.get("completion_tokens", _estimate_tokens(stdout))
            else:
                agent.estimated_completion_tokens = _estimate_tokens(stdout)
            agent.estimated_cost_usd = _estimate_cost_usd(agent.model, agent.estimated_completion_tokens)
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
        _enforce_budget_guard(run, lock, process_manager, cancel_event)
        from orchestra.engine.replay import get_recorder
        get_recorder(run.run_id).record_agent_completed(
            alias, agent.stdout_log or "", returncode
        )
        artifacts.write_manifest(run)

    return agent


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

    def _on_attempt(current_provider, current_effort: str, attempt: int) -> None:
        nonlocal last_provider
        last_provider = current_provider
        with lock:
            agent.provider = current_provider.name
            agent.model = current_provider.model_label(current_effort)
            if attempt > 0:
                artifacts.append_event(
                    run.run_id,
                    {
                        "event": "agent_retry",
                        "alias": alias,
                        "attempt": attempt + 1,
                        "model": agent.model,
                    },
                )
            artifacts.write_manifest(run)

    def _is_externally_cancelled() -> bool:
        manifest = artifacts.load_manifest(run.run_id)
        return bool(manifest and manifest.get("status") == RunStatus.CANCELLED.value)

    def _scrubbed_stream_callback(chunk: str) -> None:
        if stream_callback:
            stream_callback(PiiScrubber.scrub_text(chunk))

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
            agent.error = f"exit code {returncode}"
            agent.failure_kind = FailureKind.MODEL_TIMEOUT.value if returncode == 124 else FailureKind.TOOL_EXECUTION_FAILED.value
            agent.stdout_log = stdout
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
            usage = last_provider.last_usage()
            if usage is not None:
                agent.estimated_completion_tokens = usage.get("completion_tokens", _estimate_tokens(stdout))
            else:
                agent.estimated_completion_tokens = _estimate_tokens(stdout)
            agent.estimated_cost_usd = _estimate_cost_usd(agent.model, agent.estimated_completion_tokens)
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
        from orchestra.engine.replay import get_recorder
        get_recorder(run.run_id).record_agent_completed(
            alias, agent.stdout_log or "", returncode
        )
        artifacts.write_manifest(run)

    return agent


def _run_parallel_sync(
    run: OrchestraRun,
    agents: list[tuple[str, str]],   # [(alias, prompt), ...]
    show_live: bool = True,
    install_signal_handlers: bool = True,
    stream_callback: Optional[Callable[[str], None]] = None,
    emit_console: bool = True,
    **_kwargs: object,
) -> list[AgentRun]:
    """
    Synchronous implementation using ThreadPoolExecutor (preserved for fallback).
    """
    lock = threading.Lock()
    cancel_event = threading.Event()
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
        process_manager.kill_all()
        _mark_run_cancelled(run, lock, "cancelled by user")

    if install_signal_handlers:
        signal.signal(signal.SIGINT, _handle_sigint)

    try:
        with ThreadPoolExecutor(max_workers=max(1, len(agents))) as pool:
            futures = {
                pool.submit(
                    copy_context().run,
                    _run_one_agent,
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
                ): alias
                for alias, prompt in agents
            }

            if show_live:
                with Live(_make_table(run), console=console, refresh_per_second=live_refresh_per_second) as live:
                    for future in as_completed(futures):
                        results.append(future.result())
                        with lock:
                            live.update(_make_table(run))
            else:
                for future in as_completed(futures):
                    results.append(future.result())
    except KeyboardInterrupt:
        cancel_event.set()
        process_manager.kill_all()
        _mark_run_cancelled(run, lock, "cancelled by user")
    finally:
        if install_signal_handlers and previous_handler is not None:
            signal.signal(signal.SIGINT, previous_handler)

    return results


async def run_parallel_async(
    run: OrchestraRun,
    agents: list[tuple[str, str]],
    show_live: bool = True,
    install_signal_handlers: bool = True,
    stream_callback: Optional[Callable[[str], None]] = None,
    emit_console: bool = True,
    **_kwargs: object,
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
            )
        )
        for alias, prompt in agents
    ]

    try:
        if show_live:
            with Live(_make_table(run), console=console, refresh_per_second=live_refresh_per_second) as live:
                for coro in asyncio.as_completed(tasks):
                    try:
                        agent = await coro
                    except asyncio.CancelledError:
                        continue
                    if agent is not None:
                        results.append(agent)
                    with lock:
                        live.update(_make_table(run))
                    # If budget guard triggered, cancel remaining tasks
                    if cancel_event.is_set():
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        break
        else:
            for coro in asyncio.as_completed(tasks):
                try:
                    agent = await coro
                except asyncio.CancelledError:
                    continue
                if agent is not None:
                    results.append(agent)
                if cancel_event.is_set():
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    break
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
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_parallel_async(
            run, agents,
            show_live=show_live,
            install_signal_handlers=install_signal_handlers,
            stream_callback=stream_callback,
            emit_console=emit_console,
            **_kwargs,
        ))
    # Already inside an event loop — fall back to the sync ThreadPoolExecutor implementation
    return _run_parallel_sync(
        run, agents,
        show_live=show_live,
        install_signal_handlers=install_signal_handlers,
        stream_callback=stream_callback,
        emit_console=emit_console,
        **_kwargs,
    )

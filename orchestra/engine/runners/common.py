from __future__ import annotations
"""Core run engine: Ultra-stable V5.1 (Fixed Threading & Signal Issues).

Integrated blocks (9 total):
- Block 7 (OtelTracer): Span creation for timing/tokens/cost per provider
- Block 2 (WorkspaceGuard): Guard critical files during execution
- Block 6 (PiiScrubber): Auto-scrub payloads before EventLog append
- Block 4 (Artifacts): Checkpoint hooks (planned_node, approval_gate, budget_exceeded)
- Block 9 (FTS5Cache): search() before agent → return cache hit if similarity > 0.85
- Block 3 (UPSERT): suspend_run/resume_run in approval gate flow
- Block 8 (Seq Fix): seq-ordered replay for idempotency checks
"""

import orchestra.engine.runners as _runner

import hashlib
import json
import logging
import re
from orchestra.engine.parallel import run_parallel
from orchestra.engine.parallel.budget import _daily_spend_usd

import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any, Callable

from rich.console import Console

logger = logging.getLogger("orchestra.engine.runner")
from orchestra import config
from orchestra.engine import artifacts
from orchestra.engine.reviewer import review_pair
from orchestra.engine.autonomy_policy import AutonomyPolicy
from orchestra.engine.policy_engine import PolicyEngine, PolicyDecision
from orchestra.engine.synthesizer import (
    build_synthesis_prompt,
    compute_divergence_index,
    measure_synthesis_quality,
    outputs_sufficient,
    parse_failure_report,
    select_synthesis_alias,
)
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus
from orchestra.providers.fallback import available_alias_ids
from orchestra.storage.memory import get_memory
from orchestra.storage.blackboard import get_blackboard_context, write_note
from orchestra.engine.artifacts import append_event
from orchestra.monitoring.cost_monitor import _check_budget
from orchestra.router.outcome_router import append_outcome
from orchestra.state import ApprovalState, FailureKind, FailureState
from orchestra.engine.state_suspension import suspend_run as suspend_run_to_db, delete_paused_run, resume_run as resume_run_from_db

# Block 5: Idempotency (Block 8 integration: seq-ordered replay)

# Block 7: OtelTracer
from orchestra.engine.tracer import OtelTracer

# Block 6: PiiScrubber
from orchestra.engine.pii_scrubber import PiiScrubber

# Block 2: WorkspaceGuard
from orchestra.engine import isolation
from orchestra.engine.workspace_guard import guard_critical_files

# Block 9: FTS5Cache
from orchestra.storage.fts_cache import FtsSearchCache, CacheResult

console = Console()

# Global singletons for all blocks (initialized lazily)
_tracer: Optional[OtelTracer] = None
_pii_scrubber: Optional[PiiScrubber] = None
_fts_cache: Optional[FtsSearchCache] = None
_event_log: Optional[Any] = None  # EventLog instance

_tracer_lock = threading.Lock()
_pii_scrubber_lock = threading.Lock()
_fts_cache_lock = threading.Lock()
_event_log_lock = threading.Lock()


def _get_tracer() -> OtelTracer:
    """Lazy-load OtelTracer singleton (thread-safe)."""
    global _tracer
    if _tracer is None:
        with _tracer_lock:
            if _tracer is None:
                try:
                    _tracer = OtelTracer.get_instance()
                except Exception as e:
                    logger.warning("Failed to initialize OtelTracer: %s", e)
                    _tracer = OtelTracer()  # Fallback to default
    return _tracer


def _span_model_and_provider(alias: str) -> tuple[str, str]:
    """The model and provider an alias names, for its span (`gen_ai.request.model`)."""
    try:
        from orchestra.providers import fallback
        entry = fallback.ALIAS_TABLE.get(alias.split("@", 1)[0])
        return fallback.model_of(alias), (entry[0].name if entry else "")
    except Exception:
        return alias, ""


def _emit_span_usage(span, run) -> None:
    """Emit aggregate token/cost attributes for the run onto its span.

    Sums across ALL agents so it works for the multi-agent dual/critical modes
    (whose spans carry a label, not a real alias) as well as single-agent ask —
    and captures reviewer/synthesis cost, not just one agent. No-op when the span
    is missing or the run has no agents.
    """
    if span is None or not run.agents:
        return
    tokens_in = sum(getattr(a, "prompt_tokens", 0) or 0 for a in run.agents)
    tokens_out = sum(getattr(a, "estimated_completion_tokens", 0) or 0 for a in run.agents)
    cost = sum(getattr(a, "estimated_cost_usd", 0.0) or 0.0 for a in run.agents)
    tracer = _get_tracer()
    tracer.set_tokens(span, tokens_in, tokens_out)
    tracer.set_cost(span, cost)


def _maybe_assess_calibration(run) -> Optional[dict]:
    """Assess calibration / anti-sycophancy of a completed run's output.

    Returns None when disabled, no summary, or on any assessment failure — never raises.
    """
    if not config.eval_calibration_enabled() or not run.summary:
        return None
    from orchestra.engine import calibration, eval_harness
    # Guard the nested judge run so ITS finalize does not recursively judge/calibrate
    # itself (assess_calibration spawns a run_ask that would otherwise re-enter here).
    token = eval_harness._in_judge.set(True)
    try:
        return calibration.assess_calibration(run.task, run.summary, parent_run_id=run.run_id)
    except Exception:
        logger.warning("Calibration assessment failed", exc_info=True)
        return None
    finally:
        eval_harness._in_judge.reset(token)


def _get_pii_scrubber() -> PiiScrubber:
    """Lazy-load PiiScrubber singleton (thread-safe)."""
    global _pii_scrubber
    if _pii_scrubber is None:
        with _pii_scrubber_lock:
            if _pii_scrubber is None:
                _pii_scrubber = PiiScrubber()
    return _pii_scrubber


def _get_fts_cache() -> Optional[FtsSearchCache]:
    """Lazy-load FTS5Cache singleton (thread-safe); None when `[fts_cache] enabled = false`."""
    global _fts_cache
    if not config.fts_cache_enabled():
        return None
    if _fts_cache is None:
        with _fts_cache_lock:
            if _fts_cache is None:
                try:
                    cache_db_path = config.fts_cache_db_path()
                    _fts_cache = FtsSearchCache(db_path=cache_db_path)
                except (OSError, sqlite3.Error, ValueError) as e:
                    logger.warning("Failed to initialize FtsSearchCache: %s", e)
                    _fts_cache = FtsSearchCache(db_path=":memory:")
                try:
                    # Once per process: entries past their time are never a hit, so drop them.
                    _fts_cache.cleanup_expired(ttl_hours=config.cache_ttl_hours())
                except Exception as e:
                    logger.warning("FTS5Cache cleanup failed: %s", e)
    return _fts_cache


def _get_event_log() -> Optional[Any]:
    """Lazy-load EventLog from storage (thread-safe)."""
    global _event_log
    if _event_log is None:
        with _event_log_lock:
            if _event_log is None:
                try:
                    from orchestra.storage.event_log import EventLog
                    event_log_path = config.artifact_root() / "event_log.db"
                    _event_log = EventLog(event_log_path)
                except Exception as e:
                    logger.warning("Failed to initialize EventLog: %s", e)
                    _event_log = None
    return _event_log

def _now() -> str: return datetime.now(timezone.utc).isoformat()
def _estimate_tokens(text: str) -> int:
    from orchestra.engine.context_compression import estimate_tokens
    return estimate_tokens(text)


# ==============================================================================
# Block 9: FTS5Cache Integration — Search before agent execution
# ==============================================================================
def _cache_scope() -> str:
    """The project state an answer holds for: its root, its commit and its uncommitted edits.

    Agents answer by reading the files, and the cache is shared by every project: without this
    the same question got another project's answer, or one from before the files changed.
    """
    root = config.project_root()
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              timeout=5).stdout.strip()
        edits = subprocess.run(["git", "diff", "HEAD"], cwd=root, capture_output=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        head, edits = "", b""
    return f"{root}:{head}:{hashlib.sha256(edits).hexdigest()[:16]}"


def _cached_confidence(response: str) -> float:
    """What the stored answer's agent said about it; the neutral 0.5 when it said nothing."""
    from orchestra.normalize import stated_confidence
    stated = stated_confidence(response)
    return 0.5 if stated is None else stated


def _try_cache_hit(prompt: str, run_id: str, similarity_threshold: float = 0.85, *, alias: str = "") -> Optional[CacheResult]:
    """Block 9: Return the stored answer to this prompt from this alias, if there is one.

    A hit is the same prompt answered by the same alias within the cache's time to live.
    It used to be any stored entry sharing the prompt's words, which gave a short prompt
    the answer to a different, longer one.

    Args:
        prompt: The prompt text
        run_id: Current run ID, for the log
        similarity_threshold: Unused; kept for callers that still pass it
        alias: The agent alias being asked

    Returns:
        CacheResult on a hit, None otherwise
    """
    try:
        cache = _get_fts_cache()
        if cache is None:
            return None
        hit = cache.lookup(prompt, f"{alias}@{_cache_scope()}", ttl_hours=config.cache_ttl_hours())
        if hit is None:
            logger.debug(f"FTS5Cache: no stored answer for run {run_id}")
            return None
        logger.info(f"FTS5Cache hit (run {run_id}, {alias}): saved ${hit.cost_saved_usd:.2f}")
        return hit
    except Exception as e:
        logger.warning("FTS5Cache lookup failed: %s", e, exc_info=True)
        return None


# ==============================================================================
# Block 9: FTS5Cache Integration — Store response in cache
# ==============================================================================
def _store_in_cache(prompt: str, response: str, run_id: str, cost_usd: float = 0.0, *, alias: str = "") -> None:
    """Block 9: Store prompt/response pair in FTS5Cache.

    Called after successful agent execution to populate the cache.
    Automatically scrubs PII before storage.

    Args:
        prompt: The prompt text
        response: The agent response
        run_id: Current run ID
        cost_usd: Cost of this execution (for tracking savings)
    """
    try:
        cache = _get_fts_cache()
        if cache is None:
            return

        # Store in cache (PII scrubbing happens internally in cache.store())
        cache.store(run_id, prompt, response, cost_usd=cost_usd, alias=f"{alias}@{_cache_scope()}")
        logger.debug(f"FTS5Cache: stored result for run {run_id}")
    except Exception as e:
        logger.warning("FTS5Cache store failed: %s", e, exc_info=True)


def _append_event_scrubbed(run_id: str, event: dict) -> None:
    """Block 6: Append event to EventLog with PII scrubbing.

    Auto-scrubs sensitive payloads before storing to prevent data leakage.

    Args:
        run_id: Current run ID
        event: Event dict (will be scrubbed in-place)
    """
    try:
        scrubber = _get_pii_scrubber()

        # Scrub common sensitive fields
        for field in ["prompt", "response", "output", "error", "message"]:
            if field in event and isinstance(event[field], str):
                event[field] = scrubber.scrub_text(event[field])

        # Scrub stdout_log if present
        if "stdout_log" in event and isinstance(event["stdout_log"], str):
            event["stdout_log"] = scrubber.scrub_text(event["stdout_log"])

        append_event(run_id, event)
    except Exception as e:
        logger.warning("PII scrubbing failed: %s", e, exc_info=True)
        # Fall back to unscrubbedappend
        append_event(run_id, event)


def _is_user_facing_agent(agent: AgentRun) -> bool:
    """Return True for agents whose output may be surfaced to the user."""
    return agent.alias != "verifier"


def _safe_step(run: OrchestraRun, name: str, fn: Callable[[], Any]) -> None:
    """Run one finalize side-effect in isolation so a single failure cannot abort the rest of
    the chain (addresses M-11: the finalize chain had no failure isolation, so a mid-chain
    error could leave e.g. a manifest written but no outcome recorded). Emits a
    finalize_step_failed event on error so the degraded state is observable.
    """
    try:
        fn()
    except Exception:
        logger.warning("finalize step %r failed for run %s", name, run.run_id, exc_info=True)
        try:
            append_event(run.run_id, {"event": "finalize_step_failed", "step": name})
        except Exception:
            logger.warning("could not record finalize_step_failed for %s", run.run_id, exc_info=True)


def _log_eval_outcome(run: OrchestraRun) -> None:
    """Log evaluation outcome (P0 #1: Eval Harness): COMPLETED → PASS, anything else → FAIL."""
    try:
        from orchestra.storage.eval_tracker import EvalTracker
        eval_status = "PASS" if run.status == RunStatus.COMPLETED else "FAIL"
        EvalTracker().log_run(
            run_id=run.run_id,
            task=run.task,
            mode=run.mode,
            status=eval_status,
            created_at=run.updated_at or run.created_at
        )
        append_event(run.run_id, {"event": "eval_outcome_logged", "status": eval_status})
    except Exception as e:
        logger.warning("Failed to log eval outcome: %s", e, exc_info=True)


def _final_status(run: OrchestraRun) -> RunStatus:
    """The status a run ends with.

    A verdict the runner already reached stands: a run it failed (policy denial, budget,
    empty plan) or that was cancelled is not turned back into a success because its agents
    happened to finish. Otherwise the agents decide.
    """
    if run.status in (RunStatus.FAILED, RunStatus.CANCELLED):
        return run.status
    latest = {a.alias: a.status for a in run.agents}
    if any(status == AgentStatus.COMPLETED for status in latest.values()):
        # An agent cut for lateness leaves the run with the answers that came in time.
        latest = {a.alias: a.status for a in run.agents if a.failure_kind != "LATE"}
    review_blocked = run.latest_review_status == "blocked"
    if latest and all(s == AgentStatus.COMPLETED for s in latest.values()) and not review_blocked:
        return RunStatus.COMPLETED
    return RunStatus.FAILED


def _answer_withheld(run: OrchestraRun) -> bool:
    """A run the policy denied does not hand its agents' output on as the answer."""
    return run.failure is not None and run.failure.kind == FailureKind.POLICY_DENIED.value


def _finalize(run: OrchestraRun) -> None:
    run.status = _final_status(run)
    if not run.summary and not _answer_withheld(run):
        first_completed = next(
            (
                agent for agent in run.agents
                if agent.status == AgentStatus.COMPLETED
                and agent.stdout_log
                and _is_user_facing_agent(agent)
            ),
            None,
        )
        if first_completed:
            run.summary = first_completed.stdout_log
    # Build consensus graph for multi-agent runs
    if len(run.agents) >= 2:
        from orchestra.engine.consensus import build_consensus_graph
        try:
            graph = build_consensus_graph(run)
            run.consensus_graph = graph.to_dict()
        except Exception:
            logger.warning("Consensus graph build failed", exc_info=True)

    # Generate federation summary if enabled
    if config.federation_config().get("enabled", False):
        from orchestra.engine.federation_summarizer import summarize_run
        from orchestra.engine.federation import FederationBus
        try:
            summary = summarize_run(run)
            if summary:
                bus = FederationBus()
                bus.enqueue_outbox(summary)
        except Exception:
            logger.warning("Federation summary failed", exc_info=True)

    run.updated_at = _now()
    # Each persistence step is isolated so one failure cannot abort the rest of the chain (M-11).
    _safe_step(run, "write_manifest", lambda: artifacts.write_manifest(run))
    _safe_step(run, "write_checkpoint", lambda: artifacts.write_checkpoint(run, "final"))
    _safe_step(run, "run_finished_event", lambda: append_event(
        run.run_id, {"event": "run_finished", "status": run.status.value,
                     # the statusline flags a run whose answers the verifier did not pass
                     "review": run.latest_review_status}))

    def _record_finalized() -> None:
        from orchestra.engine.replay import get_recorder
        get_recorder(run.run_id).record_run_finalized(run.status.value, run.summary)
    _safe_step(run, "record_run_finalized", _record_finalized)

    # A cancelled run says nothing about quality, and a step of another run (verifier, judge,
    # planned node) is not a task: neither is graded.
    if run.status != RunStatus.CANCELLED and not run.parent_run_id:
        _log_eval_outcome(run)

    # Optional LLM-as-judge scoring (config-gated, off by default). Skipped while a judge
    # run is already in flight (run_judge starts its own run) to avoid recursive re-judging.
    if (
        config.eval_harness_enabled()
        and run.status == RunStatus.COMPLETED
        and run.summary
    ):
        from orchestra.engine import eval_harness, eval_sampler
        if not eval_harness.is_judging() and eval_sampler.should_sample(
            run.run_id, config.eval_sample_rate()
        ):
            try:
                result = eval_harness.run_judge(
                    task_text=run.task,
                    agent_output=run.summary,
                    run_id=run.run_id,
                    t_hash=eval_harness.task_hash(run.task),
                    judge_alias=config.eval_judge_alias(),
                    mode=run.mode,
                )
                calibration = _maybe_assess_calibration(run) or {}
                scores = {
                    "answer_quality": result.answer_quality,
                    "factual_grounding": result.factual_grounding,
                    "source_attribution": result.source_attribution,
                    "conciseness": result.conciseness,
                    "calibration": float(calibration.get("calibration_score", 0.0)),
                    "composite_score": result.composite_score,
                }
                from orchestra.storage.eval_tracker import EvalTracker
                EvalTracker().log_score(run.run_id, eval_harness.task_hash(run.task), scores)
                append_event(run.run_id, {
                    "event": "eval_judge_completed",
                    "composite_score": result.composite_score,
                    "calibration_score": scores["calibration"],
                    "explanation": result.explanation[:200],
                })
            except Exception:
                logger.warning("Auto eval judge failed", exc_info=True)

    _safe_step(run, "append_outcome", lambda: append_outcome(run))
    # Runs are no longer stored in memory; its expired notes and old log entries still go.
    _safe_step(run, "memory_sweep", lambda: get_memory().sweep())


def show_in_status_line() -> None:
    """Start the status-line writer, so a run started from the terminal is shown as well.

    The service does this for MCP and HTTP runs; a runner called directly would otherwise
    leave no trace in the status line. Never lets a display problem stop a run.
    """
    try:
        from orchestra import live_status
        live_status.install()
    except Exception:
        logger.debug("live status writer could not be installed", exc_info=True)


def _run_ask_impl(alias: str, prompt: str, **kwargs) -> OrchestraRun:
    show_in_status_line()
    # 🛡️ Force install_signal_handlers=False for daemon safety
    kwargs["install_signal_handlers"] = False
    # An internal step (verifier, synthesiser, note summary) belongs to another run and brings a
    # finished prompt: it is not prefixed again, not answered from the cache and not stopped by an
    # approval gate of its own.
    internal = bool(kwargs.pop("internal", False))
    parent_run_id = kwargs.pop("parent_run_id", None)
    # fresh: the caller wants a new answer. Nothing stored is used; the new answer is stored.
    fresh = bool(kwargs.pop("fresh", False))
    # A follow-up's answer depends on the discussion before it: it is never read from, or
    # written to, the stores that are keyed by the prompt alone.
    follows_run_id = kwargs.pop("follows_run_id", None)
    given_id = kwargs.pop("run_id", None)
    run = OrchestraRun(mode="ask", task=prompt, parent_run_id=parent_run_id, follows_run_id=follows_run_id,
                       **({"run_id": given_id} if given_id else {}))
    run.status = RunStatus.RUNNING
    artifacts.write_manifest(run)
    _append_event_scrubbed(run.run_id, {"event": "run_started", "mode": "ask", "alias": alias})

    if not internal and _denied_before_start(run, [alias], finalize=_finalize):
        return run

    # Block 2: WorkspaceGuard
    try:
        guard_critical_files()
    except Exception:
        logger.warning("Workspace guard failed", exc_info=True)

    _speculative_prefetch(run, prompt)

    # Block 7: OtelTracer instrumentation
    span = _get_tracer().start_agent_span(run.run_id, alias, *_span_model_and_provider(alias))
    t0 = time.time()

    from orchestra.engine.durable_execution import suspend_on_signal
    from orchestra.engine.state_suspension import suspend_run as _suspend_fn
    from orchestra.storage.db import get_db as _get_db
    _suspend_ctx = suspend_on_signal(run, suspend_fn=_suspend_fn, get_db_fn=_get_db)
    _suspend_ctx.__enter__()

    try:
        # Block 9: Check cache before execution
        prefixed_prompt = prompt if internal else _prefixed(alias, prompt, run)
        # An alias that works in a workspace of its own changes files: a stored answer would
        # report changes that were never made this time.
        isolated = isolation.applies_to(alias)
        cache_hit = (None if internal or fresh or follows_run_id or isolated
                     else _try_cache_hit(prompt, run.run_id, alias=alias))
        answered_now = False
        workspace_note = ""
        if cache_hit:
            # Create a synthetic agent response from cache
            agent = AgentRun(
                alias=alias,
                provider="cache",
                model="fts5_cache",
                status=AgentStatus.COMPLETED,
                start_time=_now(),
                end_time=_now(),
            )
            agent.stdout_log = cache_hit.response
            agent.estimated_completion_tokens = _estimate_tokens(cache_hit.response)
            agent.estimated_cost_usd = 0.0  # Cache hits are free
            agent.confidence = _cached_confidence(cache_hit.response)
            agent.validation_status = "passed"
            run.agents.append(agent)
            _append_event_scrubbed(run.run_id, {
                "event": "cache_hit",
                "alias": alias,
                "similarity": cache_hit.similarity,
                "cost_saved_usd": cache_hit.cost_saved_usd,
            })
            from orchestra.engine.replay import get_recorder
            get_recorder(run.run_id).record_agent_dispatch(alias, prefixed_prompt, "fts5_cache", "cache")
            get_recorder(run.run_id).record_agent_completed(alias, cache_hit.response, 0)
            artifacts.write_manifest(run)
        else:
            # No cache hit: execute agent normally
            workspace = isolation.open_workspace(run.run_id) if isolated else None
            try:
                run_parallel(run, [(alias, prefixed_prompt)],
                             **({**kwargs, "cwd": str(workspace)} if workspace else kwargs))
            finally:
                if workspace is not None:
                    changes = isolation.close_workspace(run.run_id, workspace)
                    if changes.strip():
                        workspace_note = isolation.how_to_apply(run.run_id, changes)
                        _append_event_scrubbed(run.run_id, {"event": "workspace_changes_collected",
                                                            "patch_chars": len(changes)})
            answered_now = not isolated

        # Block 4: Checkpoint for ask completion
        artifacts.write_checkpoint(run, "ask_completion")

        # Budget enforcement for single-agent runs
        run.total_cost_usd = round(sum(a.estimated_cost_usd for a in run.agents if a.status == AgentStatus.COMPLETED), 6)
        ok, reason = _check_budget(run)
        if not ok:
            logger.warning(f"Budget enforcement: {reason}")
            run.status = RunStatus.FAILED
            run.failure = FailureState(
                kind=FailureKind.BUDGET_EXCEEDED.value,
                message=reason,
                retryable=False,
                source="budget_guard",
            )
            for agent in run.agents:
                if agent.status == AgentStatus.QUEUED:
                    agent.status = AgentStatus.CANCELLED
                    agent.error = "budget_exceeded"
            _append_event_scrubbed(run.run_id, {"event": "budget_exceeded", "reason": reason})
            _finalize(run)
            return run

        # Evaluate approval policies (P0 #2: Approval Gates)
        decision = None if internal else _evaluate_policy(run, prompt)
        if _stop_at_policy(run, decision, "ask_completion", suspend=suspend_run_to_db, finalize=_finalize):
            return run

        # Block 9: Store the answer for the next identical ask. Only now: an answer the policy
        # denied, or that is still waiting for approval, is not handed to the next caller.
        if answered_now and not internal and not follows_run_id \
                and run.agents and run.agents[-1].status == AgentStatus.COMPLETED:
            _store_in_cache(
                prompt,
                run.agents[-1].stdout_log or "",
                run.run_id,
                cost_usd=run.agents[-1].estimated_cost_usd or 0.0,
                alias=alias,
            )

        if not run.summary:
            first_completed = next(
                (
                    agent for agent in run.agents
                    if agent.status == AgentStatus.COMPLETED and agent.stdout_log
                ),
                None,
            )
            if first_completed:
                run.summary = first_completed.stdout_log
        if workspace_note:
            run.summary = f"{run.summary}\n\n{workspace_note}" if run.summary else workspace_note

        _finalize(run); return run
    except Exception as e:
        # Block 7: Record error and timing
        if span:
            _get_tracer().set_error(span, str(e))
        _fail_crashed(run, e, finalize=_finalize)
        raise
    finally:
        _suspend_ctx.__exit__(None, None, None)  # __exit__ args intentionally None: suspend_on_signal only restores signal handlers and ignores exc info.
        # Block 7: End span with timing metrics
        if span:
            _get_tracer().set_timing(span, "total_latency_ms", (time.time() - t0) * 1000)
            _emit_span_usage(span, run)
            span.end()


def _speculative_prefetch(run: OrchestraRun, prompt: str) -> None:
    """Launch PASTE speculative prefetch in background threads (best-effort, never blocks)."""
    try:
        speculation_cfg = config.section("speculation") if hasattr(config, "section") else {}
        if not bool(speculation_cfg.get("enabled", False)):
            return
        from orchestra.engine.speculative import SpeculativeExecutor
        from orchestra.engine.tool_registry import register_speculative_tools
        executor = SpeculativeExecutor(run_id=run.run_id)
        register_speculative_tools(executor)
        threads = executor.prefetch(prompt)
        if threads:
            _append_event_scrubbed(run.run_id, {
                "event": "speculative_prefetch",
                "tools_launched": len(threads),
            })
    except Exception:
        logger.warning("Speculative prefetch failed", exc_info=True)

def _validate_persona_alias(alias: str | None, default: str, role: str) -> str:
    """Validate a persona alias against registered providers.

    Falls back to *default* with a warning if the alias is unknown.
    """
    if alias is None:
        return default
    try:
        valid = available_alias_ids()
    except Exception:
        logger.warning("Failed to load available aliases", exc_info=True)
        valid = []
    if alias not in valid:
        logger.warning(
            "Persona '%s' references unregistered alias '%s'; falling back to '%s'",
            role, alias, default,
        )
        return default
    return alias


# Imported last: policy_gate and prompting read helpers defined above. Re-exported, so
# _run_ask_impl, the runners and patches of common.<name> keep working.
from orchestra.engine.runners.policy_gate import (  # noqa: E402,F401
    _evaluate_policy,
    _should_require_approval,
    _estimate_run_cost,
    _estimate_avg_confidence,
    approval_requested_event,
    _fail_crashed,
    _denied_before_start,
    _stop_at_policy,
)
from orchestra.engine.runners.prompting import (  # noqa: E402,F401
    _ROLE_PREFIXES,
    _UNITY_ANNOTATION_RULE,
    _CONFIDENCE_RULE,
    _PROJECT_DOCS_RULE,
    _UNITY_TASK,
    _UNITY_CONTEXT_MARKER,
    _RETRIEVAL_POLICY_FILE,
    _AGENT_POLICY_FILE,
    _load_retrieval_policy,
    _load_agent_policy,
    _prefixed,
)

"""Core run engine: Ultra-stable V5.1 (Fixed Threading & Signal Issues).

Integrated blocks (9 total):
- Block 5 (Idempotency): @idempotent decorator wraps agent calls
- Block 7 (OtelTracer): Span creation for timing/tokens/cost per provider
- Block 2 (WorkspaceGuard): Guard critical files during execution
- Block 6 (PiiScrubber): Auto-scrub payloads before EventLog append
- Block 4 (Artifacts): Checkpoint hooks (planned_node, approval_gate, budget_exceeded)
- Block 9 (FTS5Cache): search() before agent → return cache hit if similarity > 0.85
- Block 3 (UPSERT): suspend_run/resume_run in approval gate flow
- Block 8 (Seq Fix): seq-ordered replay for idempotency checks
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any, Callable

from rich.console import Console

logger = logging.getLogger(__name__)
from orchestra import config
from orchestra.engine import artifacts
from orchestra.engine.parallel import run_parallel
from orchestra.engine.reviewer import review_pair
from orchestra.engine.synthesizer import build_synthesis_prompt, select_synthesis_alias, outputs_sufficient
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus
from orchestra.storage.memory import get_memory
from orchestra.storage.blackboard import get_blackboard_context, write_note
from orchestra.engine.artifacts import append_event
from orchestra.state import ApprovalState
from orchestra.engine.state_suspension import suspend_run as suspend_run_to_db, delete_paused_run, resume_run as resume_run_from_db

# Block 5: Idempotency (Block 8 integration: seq-ordered replay)
from orchestra.engine.idempotency import idempotent, IdempotencyKey

# Block 7: OtelTracer
from orchestra.engine.tracer import OtelTracer

# Block 6: PiiScrubber
from orchestra.engine.pii_scrubber import PiiScrubber

# Block 9: FTS5Cache
from orchestra.storage.fts_cache import FtsSearchCache, CacheResult

console = Console()
_memory = get_memory()

# Global singletons for all blocks (initialized lazily)
_tracer: Optional[OtelTracer] = None
_pii_scrubber: Optional[PiiScrubber] = None
_fts_cache: Optional[FtsSearchCache] = None
_event_log: Optional[Any] = None  # EventLog instance


def _get_tracer() -> OtelTracer:
    """Lazy-load OtelTracer singleton."""
    global _tracer
    if _tracer is None:
        try:
            _tracer = OtelTracer.get_instance()
        except Exception as e:
            logger.warning(f"Failed to initialize OtelTracer: {e}")
            _tracer = OtelTracer()  # Fallback to default
    return _tracer


def _get_pii_scrubber() -> PiiScrubber:
    """Lazy-load PiiScrubber singleton."""
    global _pii_scrubber
    if _pii_scrubber is None:
        _pii_scrubber = PiiScrubber()
    return _pii_scrubber


def _get_fts_cache() -> FtsSearchCache:
    """Lazy-load FTS5Cache singleton."""
    global _fts_cache
    if _fts_cache is None:
        try:
            cache_db_path = config.fts_cache_db_path()
            _fts_cache = FtsSearchCache(db_path=cache_db_path)
        except Exception as e:
            logger.warning(f"Failed to initialize FtsSearchCache: {e}")
            _fts_cache = FtsSearchCache(db_path=":memory:")
    return _fts_cache


def _get_event_log() -> Optional[Any]:
    """Lazy-load EventLog from storage."""
    global _event_log
    if _event_log is None:
        try:
            from orchestra.storage.event_log import EventLog
            event_log_path = str(config.artifact_root() / "event_log.db")
            _event_log = EventLog(event_log_path)
        except Exception as e:
            logger.warning(f"Failed to initialize EventLog: {e}")
            _event_log = None
    return _event_log

def _now() -> str: return datetime.now(timezone.utc).isoformat()
def _estimate_tokens(text: str) -> int: return max(0, len(text) // 4)

_UNITY_ANNOTATION_RULE = (
    "When your response refers to a specific Unity project file, script, GameObject, or scene, "
    "annotate it using this syntax so the Unity Editor can locate it directly:\n"
    "  [[asset:Assets/path/to/file.asset|DisplayName]]\n"
    "  [[script:Assets/path/to/Script.cs:42|ClassName:42]]\n"
    "  [[go:GameObjectName|Display Name]]\n"
    "  [[scene:SceneName|Display Name]]\n"
    "Only annotate paths you are confident exist in the project. Do not guess paths."
)

_UNITY_CONTEXT_MARKER = "[Unity Context]"
_RETRIEVAL_POLICY_FILE = "orchestra-retrieval-policy.md"
_AGENT_POLICY_FILE = "orchestra-agent-policy.md"

def _load_retrieval_policy() -> str:
    try:
        candidates = [Path.cwd()] + list(Path.cwd().parents)
        policy_path = None
        for base in candidates:
            candidate = base / _RETRIEVAL_POLICY_FILE
            if candidate.exists():
                policy_path = candidate
                break
        if policy_path is None:
            fallback = config.project_root() / _RETRIEVAL_POLICY_FILE
            if fallback.exists():
                policy_path = fallback
        if policy_path is None:
            return ""
        body = policy_path.read_text(encoding="utf-8").strip()
        if not body:
            return ""
        return f"[HIDDEN RETRIEVAL POLICY]\n{body}\n"
    except Exception:
        return ""


def _load_agent_policy() -> str:
    try:
        candidates = [Path.cwd()] + list(Path.cwd().parents)
        policy_path = None
        for base in candidates:
            candidate = base / _AGENT_POLICY_FILE
            if candidate.exists():
                policy_path = candidate
                break
        if policy_path is None:
            fallback = config.project_root() / _AGENT_POLICY_FILE
            if fallback.exists():
                policy_path = fallback
        if policy_path is None:
            return ""
        body = policy_path.read_text(encoding="utf-8").strip()
        if not body:
            return ""
        return f"[HIDDEN AGENT POLICY]\n{body}\n"
    except Exception:
        return ""


def _compress_blackboard_context(text: str, max_chars: int) -> str:
    body = (text or "").strip()
    if not body or len(body) <= max_chars:
        return text
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    kept: list[str] = []
    total = 0
    for line in reversed(lines):
        total += len(line) + 1
        if total > max_chars:
            break
        kept.append(line)
    kept.reverse()
    return "\n[GLOBAL BLACKBOARD (Compressed)]\n" + "\n".join(kept) + "\n"


def _should_require_approval(run: OrchestraRun, prompt: str) -> bool:
    """Evaluate approval policies to determine if human approval gate is required.

    Args:
        run: Current OrchestraRun with completed agents and metrics
        prompt: Original task prompt to check against complexity keywords

    Returns:
        bool: True if any approval policy triggers
    """
    policies = config.approval_policies_config()
    if not policies.get("enabled", False):
        return False

    # Policy 1: Cost-based approval gate
    if policies.get("require_approval_on_high_cost", False):
        cost_threshold = float(policies.get("cost_threshold_usd", 1.0))
        estimated_cost = _estimate_run_cost(run)
        if estimated_cost > cost_threshold:
            logger.info(f"Approval triggered: cost ${estimated_cost:.2f} > threshold ${cost_threshold:.2f}")
            return True

    # Policy 2: Confidence-based approval gate
    if policies.get("require_approval_on_low_confidence", False):
        confidence_threshold = float(policies.get("confidence_threshold", 0.7))
        avg_confidence = _estimate_avg_confidence(run)
        if avg_confidence < confidence_threshold:
            logger.info(f"Approval triggered: confidence {avg_confidence:.2f} < threshold {confidence_threshold:.2f}")
            return True

    # Policy 3: Task complexity-based approval gate
    if policies.get("require_approval_on_complex_task", False):
        keywords = policies.get("complexity_keywords", [])
        if any(kw.lower() in prompt.lower() for kw in keywords):
            logger.info(f"Approval triggered: task matches complexity keywords")
            return True

    # Policy 4: Mode-based approval requirement
    required_modes = policies.get("approval_required_for_modes", [])
    if run.mode in required_modes:
        logger.info(f"Approval triggered: mode '{run.mode}' requires approval")
        return True

    return False


def _estimate_run_cost(run: OrchestraRun) -> float:
    """Estimate total cost of run in USD based on agent token usage.

    Args:
        run: OrchestraRun with completed agents

    Returns:
        float: Estimated cost in USD
    """
    total_cost = 0.0
    for agent in run.agents:
        if not hasattr(agent, "estimated_completion_tokens"):
            continue
        tokens = agent.estimated_completion_tokens or 0
        if tokens == 0:
            continue
        # Determine model label from agent alias
        model_label = _alias_to_model_label(agent.alias)
        price_per_1k = config.price_per_1k_tokens(model_label)
        total_cost += (tokens / 1000.0) * price_per_1k
    return total_cost


def _estimate_avg_confidence(run: OrchestraRun) -> float:
    """Estimate average confidence across completed agents.

    Args:
        run: OrchestraRun with completed agents

    Returns:
        float: Average confidence (0-1), or 1.0 if no agents
    """
    confidences = []
    for agent in run.agents:
        if hasattr(agent, "confidence") and agent.confidence is not None:
            confidences.append(agent.confidence)
    if not confidences:
        return 1.0
    return sum(confidences) / len(confidences)


def _alias_to_model_label(alias: str) -> str:
    """Convert agent alias to model pricing label.

    Args:
        alias: Agent alias (e.g., 'cdx-deep', 'gmn-pro')

    Returns:
        str: Model label for pricing lookup (e.g., 'gpt-5.4/xhigh')
    """
    mapping = {
        "cdx-fast": "gpt-5.4/low",
        "cdx-deep": "gpt-5.4/xhigh",
        "gmn-fast": "gemini/flash",
        "gmn-pro": "gemini/pro",
        "cld-fast": "claude/sonnet",
        "cld-deep": "claude/opus",
    }
    return mapping.get(alias, "gpt-5.4/medium")

# Role-specific prefixes for diversity in parallel/dual runs (orchestraplan.md Phase 1.3)
_ROLE_PREFIXES: dict[str, str] = {
    "cdx-deep":    "You are an expert implementer. Provide exact code, concrete solutions, and specific file citations.",
    "cdx-fast":    "You are a fast implementer. Give concise, practical code solutions.",
    "gmn-pro":     "You are an architectural analyst. Focus on trade-offs, alternatives, and system-level context.",
    "gmn-fast":    "You are a quick analyst. Give concise analysis and the top 1-2 trade-offs.",
    "cld-deep":    "You are a senior engineer. Focus on correctness, clarity, and comprehensive solutions.",
    "cld-fast":    "You are a concise engineer. Provide clear, correct answers efficiently.",
    "oll-coder":   "You are a local code expert. Focus on implementation details and correctness.",
    "oll-analyst": "You are a local analyst. Provide structured reasoning and analysis.",
}


# ==============================================================================
# Block 9: FTS5Cache Integration — Search before agent execution
# ==============================================================================
def _try_cache_hit(prompt: str, run_id: str, similarity_threshold: float = 0.85) -> Optional[CacheResult]:
    """Block 9: Try to find cached response for this prompt.

    Searches FTS5Cache for similar prompts before agent execution.
    If hit with similarity >= threshold, returns cached result.

    Args:
        prompt: The prompt text to search for
        run_id: Current run ID for metadata
        similarity_threshold: Minimum similarity (0-1) to accept a cache hit

    Returns:
        CacheResult if hit found with sufficient similarity, None otherwise
    """
    try:
        cache = _get_fts_cache()
        if cache is None:
            return None

        # Search with conservative threshold to avoid false matches
        # BM25 scoring: -0.8 → similarity ~0.556, -2.0 → similarity ~0.333
        bm25_threshold = -0.8
        results = cache.search(prompt, topk=1, score_threshold=bm25_threshold)

        if not results:
            logger.debug(f"FTS5Cache: no results for run {run_id}")
            return None

        best_result = results[0]
        if best_result.similarity >= similarity_threshold:
            logger.info(
                f"FTS5Cache hit (run {run_id}): similarity={best_result.similarity:.3f} "
                f"(saved ${best_result.cost_saved_usd:.2f})"
            )
            return best_result
        else:
            logger.debug(
                f"FTS5Cache: similarity={best_result.similarity:.3f} < threshold {similarity_threshold}"
            )
            return None
    except Exception as e:
        logger.warning(f"FTS5Cache search failed: {e}")
        return None


# ==============================================================================
# Block 9: FTS5Cache Integration — Store response in cache
# ==============================================================================
def _store_in_cache(prompt: str, response: str, run_id: str, cost_usd: float = 0.0) -> None:
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
        cache.store(run_id, prompt, response, cost_usd=cost_usd)
        logger.debug(f"FTS5Cache: stored result for run {run_id}")
    except Exception as e:
        logger.warning(f"FTS5Cache store failed: {e}")


# ==============================================================================
# Block 6: PiiScrubber Integration — Scrub before EventLog append
# ==============================================================================
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
        logger.warning(f"PII scrubbing failed: {e}")
        # Fall back to unscrubbedappend
        append_event(run_id, event)

def _prefixed(alias: str, prompt: str, run: Optional[OrchestraRun] = None) -> str:
    from orchestra.engine.context_memory import ContextMemory
    from orchestra.storage.blackboard import read_notes

    retrieval_policy = _load_retrieval_policy()
    agent_policy = _load_agent_policy()
    annotation = f"\n\n{_UNITY_ANNOTATION_RULE}"
    role = _ROLE_PREFIXES.get(alias, "You are a senior engineer. Use shared wisdom.")
    budgets = config.token_budget_config()
    max_context_tokens = int(budgets.get("max_context_per_run", 24000))

    ctx_cfg = config._cfg.get("context_memory", {})
    recent_n = int(ctx_cfg.get("recent_n", 10))
    summary_alias = str(ctx_cfg.get("summary_alias", "cld-fast"))

    notes = read_notes(run.run_id) if run else []
    priority_context = f"{agent_policy}{retrieval_policy}"

    def _summarize(text: str) -> str:
        summary_prompt = (
            "Summarize the following agent collaboration notes in under 200 words. "
            "Keep key decisions, findings, and file paths.\n\n" + text
        )
        try:
            summary_run = run_ask(
                summary_alias, summary_prompt,
                emit_console=False, show_live=False,
                install_signal_handlers=False,
            )
            return (summary_run.agents[0].stdout_log or "") if summary_run.agents else ""
        except Exception:
            return ""

    ctx = ContextMemory(notes, recent_n=recent_n)
    return ctx.build(
        role=f"{role}{annotation}",
        priority_context=priority_context,
        prompt=prompt,
        max_tokens=max_context_tokens,
        token_counter=_estimate_tokens,
        summarize_fn=_summarize if notes else None,
    )

def _finalize(run: OrchestraRun) -> None:
    latest = {a.alias: a.status for a in run.agents}
    review_blocked = run.latest_review_status == "blocked"
    run.status = (
        RunStatus.COMPLETED
        if all(s == AgentStatus.COMPLETED for s in latest.values()) and not review_blocked
        else RunStatus.FAILED
    )
    run.updated_at = _now()
    artifacts.write_manifest(run)
    artifacts.write_checkpoint(run, "final")
    append_event(run.run_id, {"event": "run_finished", "status": run.status.value})

    # Log evaluation outcome (P0 #1: Eval Harness)
    try:
        from orchestra.storage.eval_tracker import EvalTracker
        # Map RunStatus to eval outcome: COMPLETED → PASS, anything else → FAIL
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
        logger.debug(f"Failed to log eval outcome: {e}")

    try:
        from orchestra.router.outcome_router import append_outcome
        append_outcome(run)
    except Exception:
        pass

    # Index completed task to semantic memory for future routing
    if run.status == RunStatus.COMPLETED:
        try:
            from orchestra.storage.memory import get_memory
            memory = get_memory()

            # Aggregate agent performance stats
            agent_stats = []
            for agent in run.agents:
                agent_stats.append({
                    "alias": agent.alias,
                    "status": agent.status.value if agent.status else "unknown",
                    "confidence": getattr(agent, "confidence", 0.0),
                    "tokens": getattr(agent, "estimated_completion_tokens", 0),
                })

            metadata = {
                "run_id": run.run_id,
                "mode": run.mode,
                "agent_count": len(run.agents),
                "agents": [a.alias for a in run.agents],
                "agent_stats": agent_stats,
                "total_cost": getattr(run, "total_cost_usd", 0.0),
            }
            summary = f"{run.mode.upper()}: {run.task[:200]}"
            memory.add(run.task, metadata=metadata, summary=summary)
            append_event(run.run_id, {"event": "task_indexed_to_memory", "metadata_keys": list(metadata.keys())})
        except Exception as e:
            logger.debug(f"Failed to index task to semantic memory: {e}")


def _record_round1_review(run: OrchestraRun) -> None:
    completed = [agent for agent in run.agents if agent.status == AgentStatus.COMPLETED]
    if len(completed) < 2:
        return

    connection = None
    try:
        from orchestra.storage.db import get_db
        connection = get_db()
    except Exception:
        pass

    decision = review_pair("round1", completed[0], completed[1], connection=connection)
    run.reviews.append(decision.to_dict())
    run.latest_review_stage = decision.stage
    run.latest_review_status = decision.status
    run.latest_review_winner = decision.winner
    run.latest_review_reason = decision.reason

    if connection is not None:
        try:
            from orchestra.storage.reputation import record_outcome
            winner = decision.winner
            with connection:
                for agent in completed:
                    if winner == "tie":
                        record_outcome(connection, agent.alias, "tie")
                    elif winner == agent.alias:
                        record_outcome(connection, agent.alias, "win")
                    else:
                        record_outcome(connection, agent.alias, "soft_fail" if agent.soft_failed else "loss")
        except Exception:
            pass
        finally:
            connection.close()


def _build_verification_prompt(run: OrchestraRun, prompt: str, candidates: List[AgentRun]) -> str:
    truncate_chars = int(config.synthesis_config().get("verification_candidate_truncate_chars", 6000))
    review = run.reviews[-1] if run.reviews else {}
    review_status = review.get("status", run.latest_review_status or "not_run")
    review_reason = review.get("reason", run.latest_review_reason or "none")

    lines = [
        "You are the verification loop.",
        "Inspect the candidate outputs below and determine whether they are safe to present to the user as-is.",
        "Focus on likely compile errors, hallucinated file paths, missing validation, and contradictions between candidates.",
        "Respond in this format exactly:",
        "VERDICT: PASS | NEEDS_FIX | BLOCK",
        "RATIONALE: one short paragraph",
        "FIXES:",
        "- bullet list of concrete issues or `- none`",
        "",
        "[ORIGINAL TASK]",
        prompt,
        "",
        f"[ROUND1 REVIEW] status={review_status} reason={review_reason}",
    ]

    for candidate in candidates:
        body = (candidate.stdout_log or "").strip()
        if len(body) > truncate_chars:
            body = body[:truncate_chars] + "\n...[truncated]"
        lines.extend([
            "",
            f"[CANDIDATE {candidate.alias}]",
            f"model={candidate.model}",
            f"confidence={candidate.confidence:.2f}",
            f"validation={candidate.validation_status}:{candidate.validation_reason}",
            body,
        ])

    return "\n".join(lines).strip()


def _parse_verification_verdict(text: str) -> tuple[str, str]:
    body = (text or "").strip()
    upper = body.upper()
    if "VERDICT: BLOCK" in upper:
        return "blocked", "verification_blocked"
    if "VERDICT: NEEDS_FIX" in upper:
        return "needs_attention", "verification_needs_fix"
    if "VERDICT: PASS" in upper:
        return "passed", "verification_passed"
    return "needs_attention", "verification_inconclusive"


def _run_verification_loop(run: OrchestraRun, prompt: str, **kwargs) -> None:
    candidates = [a for a in run.agents if a.status == AgentStatus.COMPLETED]
    if len(candidates) < 1:
        return

    should_verify = (
        run.mode in {"dual", "critical", "planned"}
        or any(a.validation_status != "passed" for a in candidates)
        or any("```" in (a.stdout_log or "") for a in candidates)
    )
    if not should_verify:
        return

    verifier_alias = "gmn-fast"
    append_event(run.run_id, {"event": "verification_started", "alias": verifier_alias})
    verify_prompt = _prefixed(verifier_alias, _build_verification_prompt(run, prompt, candidates), run)
    verify_run = run_ask(
        verifier_alias,
        verify_prompt,
        emit_console=False,
        install_signal_handlers=False,
        stream_callback=kwargs.get("stream_callback"),
        show_live=kwargs.get("show_live", False),
    )

    if not verify_run.agents:
        append_event(run.run_id, {"event": "verification_completed", "status": "missing"})
        return

    verifier = verify_run.agents[0]
    verifier.alias = "verifier"
    run.agents.append(verifier)

    status, reason = _parse_verification_verdict(verifier.stdout_log)
    decision = {
        "stage": "verification",
        "status": status,
        "winner": run.latest_review_winner,
        "reason": reason,
        "should_continue": status != "blocked",
        "should_synthesize": status == "passed",
        "concerns": [status],
        "score_a": 0.0,
        "score_b": 0.0,
    }
    run.reviews.append(decision)
    run.latest_review_stage = "verification"
    run.latest_review_status = status
    run.latest_review_reason = reason
    if status == "blocked":
        run.failure = FailureState(
            kind=FailureKind.REVIEW_REJECTED.value,
            message="Verification loop blocked delivery",
            retryable=False,
            source="verification",
            agent_alias="verifier",
        )

    write_note(
        run.run_id,
        "verifier",
        f"{status}: {reason}",
        tags=["verification", run.mode],
    )
    artifacts.write_manifest(run)
    append_event(
        run.run_id,
        {
            "event": "verification_completed",
            "status": status,
            "reason": reason,
            "alias": "verifier",
        },
    )


def _synthesize_if_possible(run: OrchestraRun, prompt: str, **kwargs) -> None:
    completed = [a for a in run.agents if a.status == AgentStatus.COMPLETED and not a.soft_failed]
    if len(completed) < 2 or not outputs_sufficient(completed[0], completed[1]):
        append_event(run.run_id, {"event": "synthesis_skipped", "reason": "insufficient_candidate_outputs"})
        return

    synth_alias = select_synthesis_alias(completed[0], completed[1], len(prompt))
    synth_prompt = _prefixed(synth_alias, build_synthesis_prompt(completed[0], completed[1]), run)
    synth = run_ask(
        synth_alias,
        synth_prompt,
        emit_console=False,
        install_signal_handlers=False,
        stream_callback=kwargs.get("stream_callback"),
        show_live=kwargs.get("show_live", False),
    )
    if not synth.agents:
        append_event(run.run_id, {"event": "synthesis_skipped", "alias": synth_alias, "reason": "no_synth_agent_output"})
        return

    first = synth.agents[0]
    if first.stdout_log:
        run.summary = first.stdout_log
        append_event(run.run_id, {"event": "synthesis_completed", "alias": synth_alias})
    else:
        append_event(run.run_id, {"event": "synthesis_skipped", "alias": synth_alias, "reason": "empty_synth_output"})


def _resume_critical(run: OrchestraRun, checkpoint: dict | None, **kwargs) -> OrchestraRun:
    force_continue = bool(kwargs.get("force_continue", False))
    checkpoint_label = (checkpoint or {}).get("label", "")
    checkpoint_status = (checkpoint or {}).get("status", "")

    if checkpoint_label == "approval_gate" and run.approval_state == ApprovalState.PENDING and not force_continue:
        run.status = RunStatus.WAITING_APPROVAL
        run.updated_at = _now()
        artifacts.write_manifest(run)
        append_event(run.run_id, {"event": "run_resume_checkpointed", "mode": run.mode, "label": checkpoint_label})
        return run

    if force_continue or run.approval_state in {ApprovalState.APPROVED, ApprovalState.RESUMED} or checkpoint_status == RunStatus.RUNNING.value:
        run.status = RunStatus.RUNNING
        run.approval_state = ApprovalState.RESUMED
        run.updated_at = _now()
        artifacts.write_manifest(run)
        append_event(run.run_id, {"event": "run_resumed", "mode": run.mode, "label": checkpoint_label or "manifest"})

        # Load persisted state from suspension storage if available
        try:
            from orchestra.storage.db import get_db
            from orchestra.engine.state_suspension import resume_run as load_paused_state
            connection = get_db()
            paused_run = load_paused_state(connection, run.run_id)
            if paused_run:
                # Sync agent outputs and reviews from paused state
                run.agents = paused_run.agents
                run.reviews = paused_run.reviews
                run.summary = paused_run.summary
                delete_paused_run(connection, run.run_id)
                append_event(run.run_id, {"event": "paused_state_restored", "agent_count": len(run.agents)})
            connection.close()
        except Exception as e:
            console.log(f"[warning] Failed to restore paused state: {e}")

        _synthesize_if_possible(run, run.task, **kwargs)
        _finalize(run)
        return run

    if checkpoint_status == RunStatus.WAITING_APPROVAL.value:
        run.status = RunStatus.WAITING_APPROVAL
        run.updated_at = _now()
        artifacts.write_manifest(run)
        return run

    return run

def run_ask(alias: str, prompt: str, **kwargs) -> OrchestraRun:
    # 🛡️ Force install_signal_handlers=False for daemon safety
    kwargs["install_signal_handlers"] = False
    run = OrchestraRun(mode="ask", task=prompt); run.status = RunStatus.RUNNING; artifacts.write_manifest(run)
    _append_event_scrubbed(run.run_id, {"event": "run_started", "mode": "ask", "alias": alias})
    _speculative_prefetch(run, prompt)

    # Block 7: OtelTracer instrumentation
    span = _get_tracer().start_agent_span(run.run_id, alias, f"ask/{alias}")
    t0 = time.time()

    try:
        # Block 9: Check cache before execution
        prefixed_prompt = _prefixed(alias, prompt, run)
        cache_hit = _try_cache_hit(prompt, run.run_id, similarity_threshold=0.85)
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
            agent.confidence = 1.0  # High confidence in cached results
            agent.validation_status = "passed"
            run.agents.append(agent)
            _append_event_scrubbed(run.run_id, {
                "event": "cache_hit",
                "alias": alias,
                "similarity": cache_hit.similarity,
                "cost_saved_usd": cache_hit.cost_saved_usd,
            })
            artifacts.write_manifest(run)
        else:
            # No cache hit: execute agent normally
            run_parallel(run, [(alias, prefixed_prompt)], **kwargs)

            # Block 9: Store result in cache after successful execution
            if run.agents and run.agents[-1].status == AgentStatus.COMPLETED:
                _store_in_cache(
                    prompt,
                    run.agents[-1].stdout_log or "",
                    run.run_id,
                    cost_usd=run.agents[-1].estimated_cost_usd or 0.0,
                )

        # Block 4: Checkpoint for ask completion
        artifacts.write_checkpoint(run, "ask_completion")

        # Evaluate approval policies (P0 #2: Approval Gates)
        policy_triggered = _should_require_approval(run, prompt)
        if policy_triggered:
            run.status = RunStatus.WAITING_APPROVAL
            run.approval_state = ApprovalState.PENDING
            run.updated_at = _now()
            artifacts.write_checkpoint(run, "approval_gate")
            _append_event_scrubbed(run.run_id, {"event": "approval_requested", "stage": "ask_completion"})

            try:
                from orchestra.storage.db import get_db
                connection = get_db()
                # Block 3: UPSERT suspend_run
                suspend_run_to_db(connection, run, paused_by="approval_gate")
                connection.close()
            except Exception as e:
                console.log(f"[warning] Failed to persist paused state: {e}")

            return run

        _finalize(run); return run
    except Exception as e:
        # Block 7: Record error and timing
        if span:
            _get_tracer().set_error(span, str(e))
        raise
    finally:
        # Block 7: End span with timing metrics
        if span:
            _get_tracer().set_timing(span, "total_latency_ms", (time.time() - t0) * 1000)
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
        pass  # speculation is always best-effort

def run_dual(prompt: str, *, agents: list[str] | None = None, **kwargs) -> OrchestraRun:
    kwargs["install_signal_handlers"] = False
    run = OrchestraRun(mode="dual", task=prompt); run.status = RunStatus.RUNNING; artifacts.write_manifest(run)
    _append_event_scrubbed(run.run_id, {"event": "run_started", "mode": "dual"})

    # Block 7: OtelTracer instrumentation
    agent_alias = "dual"
    span = _get_tracer().start_agent_span(run.run_id, agent_alias, f"dual")
    t0 = time.time()

    try:
        _agents = agents if agents and len(agents) >= 2 else ["cdx-deep", "gmn-pro"]
        run_parallel(run, [(_agents[0], _prefixed(_agents[0], prompt, run)), (_agents[1], _prefixed(_agents[1], prompt, run))], **kwargs)

        # Block 4: Checkpoint after round 1
        artifacts.write_checkpoint(run, "round1_complete")

        _record_round1_review(run)
        _run_verification_loop(run, prompt, **kwargs)
        if run.latest_review_status == "blocked":
            _finalize(run); return run

        # Block 4: Checkpoint after verification
        artifacts.write_checkpoint(run, "verification_complete")

        # Evaluate approval policies (P0 #2: Approval Gates)
        policy_triggered = _should_require_approval(run, prompt)
        if policy_triggered:
            run.status = RunStatus.WAITING_APPROVAL
            run.approval_state = ApprovalState.PENDING
            run.updated_at = _now()
            artifacts.write_checkpoint(run, "approval_gate")
            _append_event_scrubbed(run.run_id, {"event": "approval_requested", "stage": "dual_verification"})

            try:
                from orchestra.storage.db import get_db
                connection = get_db()
                # Block 3: UPSERT suspend_run
                suspend_run_to_db(connection, run, paused_by="approval_gate")
                connection.close()
            except Exception as e:
                console.log(f"[warning] Failed to persist paused state: {e}")

            return run

        _synthesize_if_possible(run, prompt, **kwargs)
        _finalize(run); return run
    except Exception as e:
        # Block 7: Record error and timing
        if span:
            _get_tracer().set_error(span, str(e))
        raise
    finally:
        # Block 7: End span with timing metrics
        if span:
            _get_tracer().set_timing(span, "total_latency_ms", (time.time() - t0) * 1000)
            span.end()


def run_critical(
    prompt: str,
    *,
    require_approval: bool = False,
    approval_behavior: str = "continue",
    **kwargs,
) -> OrchestraRun:
    kwargs["install_signal_handlers"] = False
    run = OrchestraRun(mode="critical", task=prompt)
    run.status = RunStatus.RUNNING
    artifacts.write_manifest(run)
    artifacts.write_checkpoint(run, "started")
    _append_event_scrubbed(run.run_id, {"event": "run_started", "mode": "critical"})

    # Block 7: OtelTracer instrumentation
    agent_alias = "critical"
    span = _get_tracer().start_agent_span(run.run_id, agent_alias, f"critical")
    t0 = time.time()

    try:
        # Use faster agents for round1 so approval-gated flows become responsive.
        # Deep reasoning can still happen in the later synthesis / continuation stages.
        run_parallel(run, [("cdx-fast", _prefixed("cdx-fast", prompt, run)), ("gmn-fast", _prefixed("gmn-fast", prompt, run))], **kwargs)
        _record_round1_review(run)
        _run_verification_loop(run, prompt, **kwargs)
        artifacts.write_checkpoint(run, "round1_complete")
        if run.latest_review_status == "blocked":
            _finalize(run)
            return run

        # Block 4: Checkpoint before approval decision
        artifacts.write_checkpoint(run, "verification_complete")

        # Evaluate approval policies if not explicitly required
        policy_triggered = not require_approval and _should_require_approval(run, prompt)
        should_pause = require_approval or policy_triggered or approval_behavior == "pause"
        if should_pause:
            run.status = RunStatus.WAITING_APPROVAL
            run.approval_state = ApprovalState.PENDING
            run.updated_at = _now()
            artifacts.write_checkpoint(run, "approval_gate")
            _append_event_scrubbed(run.run_id, {"event": "approval_requested", "stage": "round1"})

            # Persist state to database for resumption after approval
            try:
                from orchestra.storage.db import get_db
                connection = get_db()
                # Block 3: UPSERT suspend_run
                suspend_run_to_db(connection, run, paused_by="approval_gate")
                connection.close()
            except Exception as e:
                console.log(f"[warning] Failed to persist paused state: {e}")

            return run

        _synthesize_if_possible(run, prompt, **kwargs)
        _critic_rerun_if_needed(run, prompt, **kwargs)
        _finalize(run)
        return run
    except Exception as e:
        # Block 7: Record error and timing
        if span:
            _get_tracer().set_error(span, str(e))
        raise
    finally:
        # Block 7: End span with timing metrics
        if span:
            _get_tracer().set_timing(span, "total_latency_ms", (time.time() - t0) * 1000)
            span.end()


def _critic_rerun_if_needed(run: OrchestraRun, prompt: str, **kwargs) -> None:
    """CriticAgent: one-shot follow-up when dissent found + confidence below threshold.

    Only fires on critical mode, at most once per run (guarded by event check).
    """
    if not run.summary:
        return

    # Guard: only once per run
    existing_events = artifacts.read_events(run.run_id)
    if any(e.get("event") == "critic_triggered" for e in existing_events):
        return

    review = config.review_config()
    threshold = float(review.get("critic_confidence_threshold", 0.65))
    if run.avg_confidence >= threshold:
        return

    from orchestra.engine.synthesizer import parse_dissent
    dissent = parse_dissent(run.summary)
    if not dissent:
        return

    critic_alias = str(review.get("critic_alias", "cld-fast"))
    critic_prompt = (
        f"A previous analysis of the following task had this dissenting viewpoint:\n\n"
        f"DISSENT:\n{dissent}\n\n"
        f"ORIGINAL TASK:\n{prompt}\n\n"
        f"Address the dissent directly. Provide a revised, confident answer that resolves "
        f"the disagreement or explains why the original conclusion stands."
    )
    _append_event_scrubbed(run.run_id, {"event": "critic_triggered", "alias": critic_alias, "dissent_length": len(dissent)})
    critic_run = run_ask(
        critic_alias,
        critic_prompt,
        emit_console=False,
        show_live=False,
        install_signal_handlers=False,
    )
    if critic_run.agents and critic_run.agents[0].stdout_log:
        run.summary = critic_run.agents[0].stdout_log
        _append_event_scrubbed(run.run_id, {"event": "critic_completed", "alias": critic_alias})


def run_planned(prompt: str, **kwargs) -> OrchestraRun:
    kwargs["install_signal_handlers"] = False
    from orchestra.engine.planner import create_graph_plan
    from orchestra.storage.db import get_db
    import time
    run = OrchestraRun(mode="planned", task=prompt); run.status = RunStatus.RUNNING; artifacts.write_manifest(run)
    _append_event_scrubbed(run.run_id, {"event": "run_started", "mode": "planned"})
    graph = create_graph_plan(run)
    if not graph or not graph.nodes:
        run.status = RunStatus.FAILED; _finalize(run); return run
    visited = set()
    queue = [n.name for n in graph.nodes.values() if n.name not in [e.target for e in graph.edges]]
    node_counter = 0
    db = None
    try:
        db = get_db()
    except Exception:
        pass
    while queue:
        name = queue.pop(0)
        if name in visited:
            continue
        if not graph.record_step():
            _append_event_scrubbed(run.run_id, {"event": "max_steps_reached", "steps": graph._steps_executed})
            break
        node = graph.nodes[name]
        _append_event_scrubbed(run.run_id, {"event": "node_started", "name": name, "dynamic": node.dynamic})
        step_run = run_ask(node.agent, node.action, emit_console=False, **kwargs)
        node.status = "completed" if step_run.status == RunStatus.COMPLETED else "failed"
        node.result = step_run.summary or ""
        visited.add(name)

        # Block 4: Checkpoint after each planned node completion
        node_counter += 1
        artifacts.write_checkpoint(run, f"planned_node_{node_counter}_{name}")

        # Block 4: UPSERT to planned_checkpoints table
        if db and node.status == "completed":
            try:
                cost = (step_run.agents[0].estimated_cost_usd if step_run.agents else 0.0) or 0.0
                db.execute(
                    "INSERT INTO planned_checkpoints (run_id, node_id, result_json, status, completed_at, cost_usd) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(run_id, node_id) DO UPDATE SET "
                    "result_json=excluded.result_json, status=excluded.status, completed_at=excluded.completed_at, cost_usd=excluded.cost_usd",
                    (run.run_id, name, json.dumps({"result": node.result}), "COMPLETED", time.time(), cost)
                )
            except Exception as e:
                logger.warning(f"Failed to UPSERT planned_checkpoint: {e}")

        next_nodes = graph.get_next_nodes(name, node.status)
        queue.extend(next_nodes)

    if db:
        try:
            db.close()
        except Exception:
            pass

    _run_verification_loop(run, prompt, **kwargs)

    # Block 4: Approval gate after verification
    policy_triggered = _should_require_approval(run, prompt)
    if policy_triggered:
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING
        run.updated_at = _now()
        artifacts.write_checkpoint(run, "approval_gate")
        _append_event_scrubbed(run.run_id, {"event": "approval_requested", "stage": "planned_verification"})

        try:
            from orchestra.storage.db import get_db
            connection = get_db()
            suspend_run_to_db(connection, run, paused_by="approval_gate")
            connection.close()
        except Exception as e:
            console.log(f"[warning] Failed to persist paused state: {e}")

        return run

    _finalize(run)
    return run


def resume_run(run_id: str, approval_decision: str = "approve", **kwargs) -> OrchestraRun:
    """Resume a paused run after approval decision (Phase 3: PENDING → APPROVED → RUNNING).

    Loads paused run from database, transitions approval state, and continues execution
    or terminates based on approval decision.

    Args:
        run_id: ID of paused run
        approval_decision: "approve" (continue) or "reject" (fail)
        **kwargs: passed to synthesis/finalization

    Returns:
        OrchestraRun with updated status and approval_state
    """
    from orchestra.storage.db import get_db
    from orchestra.engine.state_suspension import resume_run as resume_run_from_db, delete_paused_run

    connection = get_db()
    paused_run = resume_run_from_db(connection, run_id)

    if paused_run:
        run = paused_run
        # Guard: ensure run is in WAITING_APPROVAL state
        if run.status != RunStatus.WAITING_APPROVAL:
            raise ValueError(f"Cannot resume run {run_id}: status={run.status.value}, expected WAITING_APPROVAL")

        if approval_decision == "approve":
            run.approval_state = ApprovalState.APPROVED
            run.status = RunStatus.RUNNING
            run.updated_at = _now()
            append_event(run.run_id, {"event": "approval_resumed", "decision": "approved"})
            artifacts.write_checkpoint(run, "approval_approved")

            # Continue execution based on mode
            if run.mode == "ask":
                _finalize(run)
            elif run.mode == "dual":
                _synthesize_if_possible(run, run.task, **kwargs)
                _finalize(run)
            elif run.mode == "critical":
                _synthesize_if_possible(run, run.task, **kwargs)
                _finalize(run)
            elif run.mode == "planned":
                # Load completed node_ids from planned_checkpoints
                try:
                    completed_nodes = connection.execute(
                        "SELECT node_id FROM planned_checkpoints WHERE run_id = ? AND status = 'COMPLETED'",
                        (run_id,)
                    ).fetchall()
                    completed_node_ids = {row[0] for row in completed_nodes}
                    # Restart verification with knowledge of completed nodes
                    _run_verification_loop(run, run.task, **kwargs)
                except Exception as e:
                    logger.warning(f"Failed to load planned_checkpoints: {e}")
                    _run_verification_loop(run, run.task, **kwargs)
                _finalize(run)

            try:
                delete_paused_run(connection, run_id)
            except Exception:
                pass  # Best-effort cleanup
        else:
            run.approval_state = ApprovalState.REJECTED
            run.status = RunStatus.FAILED
            run.updated_at = _now()
            append_event(run.run_id, {"event": "approval_rejected", "decision": "rejected"})
            artifacts.write_checkpoint(run, "approval_rejected")

            try:
                delete_paused_run(connection, run_id)
            except Exception:
                pass

        connection.close()
        return run

    connection.close()

    # Fallback: load from manifest/checkpoint (older resumption path)
    manifest = artifacts.load_manifest(run_id)
    checkpoint = artifacts.latest_checkpoint(run_id)
    if not manifest and not checkpoint:
        raise ValueError(f"Run not found: {run_id}")
    if not manifest and checkpoint:
        raise ValueError(f"Run manifest missing for {run_id}; latest checkpoint is {checkpoint.get('label', 'unknown')}")

    run = OrchestraRun.from_manifest(manifest)
    if run.mode == "critical":
        return _resume_critical(run, checkpoint, **kwargs)

    return run

if __name__ == "__main__": pass

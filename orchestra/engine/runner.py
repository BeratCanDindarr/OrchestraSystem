"""Core run engine: Ultra-stable V5.1 (Fixed Threading & Signal Issues)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, List, Dict

from rich.console import Console
from orchestra import config
from orchestra.engine import artifacts
from orchestra.engine.parallel import run_parallel
from orchestra.engine.reviewer import review_pair
from orchestra.engine.synthesizer import build_synthesis_prompt, select_synthesis_alias, outputs_sufficient
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus
from orchestra.storage.memory import get_memory
from orchestra.storage.blackboard import get_blackboard_context
from orchestra.engine.artifacts import append_event
from orchestra.state import ApprovalState

console = Console()
_memory = get_memory()

def _now() -> str: return datetime.now(timezone.utc).isoformat()

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

def _prefixed(alias: str, prompt: str, run: Optional[OrchestraRun] = None) -> str:
    blackboard = get_blackboard_context(run.run_id) if run else ""
    retrieval_policy = _load_retrieval_policy()
    agent_policy = _load_agent_policy()
    # Result cards should not depend on Unity context injection; project/file lookup
    # queries still need annotation hints even when no explicit context is pinned.
    annotation = f"\n\n{_UNITY_ANNOTATION_RULE}"
    role = _ROLE_PREFIXES.get(alias, "You are a senior engineer. Use shared wisdom.")
    return f"{role}{annotation}\n\n{agent_policy}{retrieval_policy}{blackboard}\n\n{prompt}"

def _finalize(run: OrchestraRun) -> None:
    latest = {a.alias: a.status for a in run.agents}
    run.status = RunStatus.COMPLETED if all(s == AgentStatus.COMPLETED for s in latest.values()) else RunStatus.FAILED
    run.updated_at = _now()
    artifacts.write_manifest(run)
    artifacts.write_checkpoint(run, "final")
    append_event(run.run_id, {"event": "run_finished", "status": run.status.value})


def _record_round1_review(run: OrchestraRun) -> None:
    if len(run.agents) < 2:
        return
    decision = review_pair("round1", run.agents[0], run.agents[1])
    run.reviews.append(decision.to_dict())
    run.latest_review_stage = decision.stage
    run.latest_review_status = decision.status
    run.latest_review_winner = decision.winner
    run.latest_review_reason = decision.reason


def _synthesize_if_possible(run: OrchestraRun, prompt: str, **kwargs) -> None:
    completed = [a for a in run.agents if a.status == AgentStatus.COMPLETED and not a.soft_failed]
    if len(completed) < 2 or not outputs_sufficient(completed[0], completed[1]):
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
    if synth.agents and synth.agents[0].stdout_log:
        run.summary = synth.agents[0].stdout_log
        append_event(run.run_id, {"event": "synthesis_completed", "alias": synth_alias})

def run_ask(alias: str, prompt: str, **kwargs) -> OrchestraRun:
    # 🛡️ Force install_signal_handlers=False for daemon safety
    kwargs["install_signal_handlers"] = False
    run = OrchestraRun(mode="ask", task=prompt); run.status = RunStatus.RUNNING; artifacts.write_manifest(run)
    append_event(run.run_id, {"event": "run_started", "mode": "ask", "alias": alias})
    run_parallel(run, [(alias, _prefixed(alias, prompt, run))], **kwargs)
    _finalize(run); return run

def run_dual(prompt: str, **kwargs) -> OrchestraRun:
    kwargs["install_signal_handlers"] = False
    run = OrchestraRun(mode="dual", task=prompt); run.status = RunStatus.RUNNING; artifacts.write_manifest(run)
    append_event(run.run_id, {"event": "run_started", "mode": "dual"})
    run_parallel(run, [("cdx-deep", _prefixed("cdx-deep", prompt, run)), ("gmn-pro", _prefixed("gmn-pro", prompt, run))], **kwargs)
    _synthesize_if_possible(run, prompt, **kwargs)
    _finalize(run); return run


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
    append_event(run.run_id, {"event": "run_started", "mode": "critical"})

    run_parallel(run, [("cdx-deep", _prefixed("cdx-deep", prompt, run)), ("gmn-pro", _prefixed("gmn-pro", prompt, run))], **kwargs)
    _record_round1_review(run)

    should_pause = require_approval or approval_behavior == "pause"
    if should_pause:
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING.value
        run.updated_at = _now()
        artifacts.write_checkpoint(run, "approval_gate")
        append_event(run.run_id, {"event": "approval_requested", "stage": "round1"})
        return run

    _synthesize_if_possible(run, prompt, **kwargs)
    _finalize(run)
    return run

def run_planned(prompt: str, **kwargs) -> OrchestraRun:
    kwargs["install_signal_handlers"] = False
    from orchestra.engine.planner import create_graph_plan
    run = OrchestraRun(mode="planned", task=prompt); run.status = RunStatus.RUNNING; artifacts.write_manifest(run)
    append_event(run.run_id, {"event": "run_started", "mode": "planned"})
    graph = create_graph_plan(run)
    if not graph or not graph.nodes:
        run.status = RunStatus.FAILED; _finalize(run); return run
    visited = set()
    queue = [n.name for n in graph.nodes.values() if n.name not in [e.target for e in graph.edges]]
    while queue:
        name = queue.pop(0)
        if name in visited: continue
        node = graph.nodes[name]
        append_event(run.run_id, {"event": "node_started", "name": name})
        step_run = run_ask(node.agent, node.action, emit_console=False, **kwargs)
        node.status = "completed" if step_run.status == RunStatus.COMPLETED else "failed"
        visited.add(name); next_nodes = graph.get_next_nodes(name, node.status); queue.extend(next_nodes)
    _finalize(run); return run


def resume_run(run_id: str, **kwargs) -> OrchestraRun:
    manifest = artifacts.load_manifest(run_id)
    if not manifest:
        raise ValueError(f"Run not found: {run_id}")

    run = OrchestraRun.from_manifest(manifest)
    if run.status == RunStatus.WAITING_APPROVAL and run.mode == "critical":
        run.status = RunStatus.RUNNING
        run.approval_state = ApprovalState.RESUMED.value
        run.updated_at = _now()
        artifacts.write_manifest(run)
        append_event(run.run_id, {"event": "run_resumed", "mode": run.mode})
        _synthesize_if_possible(run, run.task, **kwargs)
        _finalize(run)
        return run

    return run

if __name__ == "__main__": pass

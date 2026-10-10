"""Planned mode runner: graph-based execution."""
from __future__ import annotations

import orchestra.engine.runners as _runner

import json
import logging
import time

from orchestra import config
from orchestra.engine import artifacts
from orchestra.engine.runners.common import show_in_status_line
from orchestra.engine.runners.common import (
    _append_event_scrubbed,
    approval_requested_event,
    _evaluate_policy,
    _finalize,
    _now,
    _run_verification_loop,
)
from orchestra.engine.state_suspension import suspend_run as suspend_run_to_db
from orchestra.engine.workspace_guard import guard_critical_files
from orchestra.models import OrchestraRun, RunStatus
from orchestra.state import ApprovalState, FailureKind, FailureState

logger = logging.getLogger("orchestra.engine.runner")


STEP_RESULT_CHARS = 4000


def _node_alias(agent: str) -> str:
    """The alias a node runs on; the planner sometimes invents a name instead of an alias."""
    from orchestra.providers.fallback import known_aliases
    if agent in known_aliases():
        return agent
    return str(config.personas_config().get("implementer") or config.default_light_alias())


def _node_prompt(graph, node) -> str:
    """A node's instruction, followed by the results of the nodes that lead to it."""
    earlier = [graph.nodes[edge.source] for edge in graph.edges
               if edge.target == node.name and edge.source in graph.nodes]
    blocks = [f"## {source.name}\n{source.result[:STEP_RESULT_CHARS]}" for source in earlier if source.result]
    if not blocks:
        return node.action
    return f"{node.action}\n\n[RESULTS OF EARLIER STEPS]\n" + "\n\n".join(blocks)


def _attach_step(run: OrchestraRun, name: str, step_run: OrchestraRun, result: str) -> None:
    """Record a node's answer on the planned run itself, where the verifier and the logs look."""
    from orchestra.models import AgentRun, AgentStatus
    source = step_run.agents[0] if step_run.agents else None
    agent = AgentRun(
        alias=name,
        provider=source.provider if source else "",
        model=source.model if source else "",
        status=source.status if source else AgentStatus.FAILED,
        start_time=source.start_time if source else None,
        end_time=source.end_time if source else None,
    )
    agent.stdout_log = result
    if source:
        agent.estimated_cost_usd = source.estimated_cost_usd
        agent.estimated_completion_tokens = source.estimated_completion_tokens
        agent.confidence = source.confidence
        agent.validation_status = source.validation_status
    run.agents.append(agent)
    artifacts.write_agent_log(run.run_id, name, result)
    artifacts.write_manifest(run)


def run_planned(prompt: str, **kwargs) -> OrchestraRun:
    kwargs["install_signal_handlers"] = False
    from orchestra.engine.planner import create_graph_plan
    from orchestra.storage.db import get_db
    show_in_status_line()
    run = OrchestraRun(mode="planned", task=prompt); run.status = RunStatus.RUNNING; artifacts.write_manifest(run)
    _append_event_scrubbed(run.run_id, {"event": "run_started", "mode": "planned"})

    # Block 2: WorkspaceGuard
    try:
        guard_critical_files()
    except Exception:
        logger.warning("Workspace guard failed", exc_info=True)

    graph = create_graph_plan(run)
    if not graph or not graph.nodes:
        run.status = RunStatus.FAILED; _finalize(run); return run
    visited = set()
    queue = [n.name for n in graph.nodes.values() if n.name not in [e.target for e in graph.edges]]
    node_counter = 0
    started = time.monotonic()
    max_seconds = config.planned_max_seconds()
    db = None
    try:
        db = get_db()
    except Exception:
        logger.warning("Failed to get DB connection for planned run", exc_info=True)
    try:
        while queue:
            name = queue.pop(0)
            if name in visited:
                continue
            if not graph.record_step():
                _append_event_scrubbed(run.run_id, {"event": "max_steps_reached", "steps": graph._steps_executed})
                break
            elapsed = time.monotonic() - started
            if visited and elapsed > max_seconds:
                # A node that times out is retried on its fallback models, so one node can take
                # many minutes. Past the budget no new node starts; the run ends with what it has.
                skipped = [name, *[n for n in queue if n not in visited]]
                _append_event_scrubbed(run.run_id, {"event": "planned_deadline_reached", "seconds": round(elapsed),
                                                    "limit": max_seconds, "skipped": skipped})
                logger.warning("Planned run %s passed its %ss budget; skipping %s", run.run_id, max_seconds, skipped)
                break
            node = graph.nodes[name]
            _append_event_scrubbed(run.run_id, {"event": "node_started", "name": name, "dynamic": node.dynamic})
            from orchestra.engine.runners.ask_runner import run_ask
            step_kwargs = {k: v for k, v in kwargs.items() if k not in ("emit_console", "parent_run_id")}
            step_run = run_ask(
                _node_alias(node.agent), _node_prompt(graph, node), emit_console=False,
                parent_run_id=run.run_id, **step_kwargs,
            )
            node.status = "completed" if step_run.status == RunStatus.COMPLETED else "failed"
            node.result = step_run.summary or (step_run.agents[0].stdout_log if step_run.agents else "") or ""
            _attach_step(run, name, step_run, node.result)
            if node.status == "completed" and node.result:
                run.summary = node.result
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
                    # Without this the write lock is held until the run ends and every other
                    # writer fails with "database is locked".
                    db.commit()
                except Exception as e:
                    logger.warning("Failed to UPSERT planned_checkpoint: %s", e, exc_info=True)

            next_nodes = graph.get_next_nodes(name, node.status)
            queue.extend(next_nodes)
    finally:
        if db:
            try:
                db.close()
            except Exception:
                logger.warning("Failed to close DB connection", exc_info=True)

    _run_verification_loop(run, prompt, **kwargs)

    # Block 4: Approval gate after verification
    decision = _evaluate_policy(run, prompt)
    if decision.action == "deny":
        run.status = RunStatus.FAILED
        run.failure = FailureState(
            kind=FailureKind.POLICY_DENIED.value,
            message=decision.reason,
            retryable=False,
            source="policy_guard",
        )
        _finalize(run)
        return run
    elif decision.action == "gate":
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING
        run.updated_at = _now()
        artifacts.write_checkpoint(run, "approval_gate")
        _append_event_scrubbed(run.run_id, approval_requested_event("planned_verification", decision))

        try:
            from orchestra.storage.db import get_db
            connection = get_db()
            suspend_run_to_db(connection, run, paused_by="approval_gate")
            connection.close()
        except Exception as e:
            logger.warning("Failed to persist paused state: %s", e, exc_info=True)

        return run

    _finalize(run)
    return run

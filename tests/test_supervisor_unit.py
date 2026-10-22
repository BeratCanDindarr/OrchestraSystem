"""Unit tests for orchestra.engine.supervisor (hierarchical subagent spawning)."""
from __future__ import annotations

import json

from orchestra import config
from orchestra.engine import supervisor
from orchestra.models import AgentRun, AgentStatus, OrchestraRun


def _mk_run(output: str) -> OrchestraRun:
    run = OrchestraRun(mode="ask", task="t")
    run.agents = [
        AgentRun(
            alias="a", provider="p", model="m",
            status=AgentStatus.COMPLETED, stdout_log=output,
        )
    ]
    return run


def _router(subtasks_by_task):
    """Build a fake run_ask that decomposes via a lookup table and echoes leaf/synth work.

    ``subtasks_by_task`` maps a task string -> list of subtasks (empty => atomic leaf).
    Records every (alias, task) call in ``.calls``.
    """
    calls = []

    def fake_ask(alias, prompt, **kwargs):
        calls.append((alias, prompt))
        if alias == "planner":
            # Decompose: extract the task from the prompt tail after "TASK:\n"
            task = prompt.split("TASK:\n", 1)[-1].strip()
            subs = subtasks_by_task.get(task, [])
            return _mk_run(json.dumps({"subtasks": subs}))
        if alias == "synth":
            return _mk_run(f"SYNTH({prompt.count('[Subresult')})")
        return _mk_run(f"WORK:{prompt}")   # worker leaf

    fake_ask.calls = calls
    return fake_ask


def _kw(fake_ask, **over):
    base = dict(
        run_ask_fn=fake_ask, worker_alias="worker",
        planner_alias="planner", synthesizer_alias="synth",
    )
    base.update(over)
    return base


def test_depth_limit_runs_worker_directly_without_planner():
    fake = _router({"root": ["a", "b"]})
    out = supervisor.run_supervised("root", max_depth=0, **_kw(fake))
    assert supervisor._first_output(out) == "WORK:root"
    # No planner call at all: depth limit short-circuits to the worker.
    assert [c[0] for c in fake.calls] == ["worker"]


def test_atomic_task_runs_as_leaf_no_synthesis():
    fake = _router({"root": []})              # planner returns no subtasks
    out = supervisor.run_supervised("root", max_depth=2, **_kw(fake))
    assert supervisor._first_output(out) == "WORK:root"
    aliases = [c[0] for c in fake.calls]
    assert aliases == ["planner", "worker"]   # decompose then leaf, no synth


def test_multi_subtask_spawns_and_synthesizes():
    fake = _router({"root": ["s1", "s2"], "s1": [], "s2": []})
    out = supervisor.run_supervised("root", max_depth=2, **_kw(fake))
    aliases = [c[0] for c in fake.calls]
    # root decomposed -> each subtask decomposed (atomic) -> worker leaf -> final synth
    assert aliases == ["planner", "planner", "worker", "planner", "worker", "synth"]
    assert supervisor._first_output(out).startswith("SYNTH(")


def test_nested_decomposition_respects_depth():
    # root -> [s1, s2]; s1 -> [g1, g2] (deeper). Depth limit 1 => s1's children are NOT planned,
    # they run as workers directly at depth 1.
    fake = _router({"root": ["s1", "s2"], "s1": ["g1", "g2"], "s2": []})
    out = supervisor.run_supervised("root", max_depth=1, **_kw(fake))
    aliases = [c[0] for c in fake.calls]
    # depth 0: plan root. depth 1: s1 and s2 hit depth limit -> worker (no planner).
    assert aliases == ["planner", "worker", "worker", "synth"]
    assert supervisor._first_output(out).startswith("SYNTH(")


def test_supervisor_config_defaults():
    assert config.supervisor_max_depth() == 2
    assert config.supervisor_worker_alias() == "cdx-fast"
    assert config.supervisor_planner_alias() == "gmn-pro"
    assert config.supervisor_synthesizer_alias() == "gmn-pro"

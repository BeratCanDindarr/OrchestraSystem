"""The steps of a planned run belong to it: they see earlier results, their answers are the
run's answers, and they do not show up as runs of their own."""
from __future__ import annotations

from unittest.mock import patch

from orchestra.engine import artifacts
from orchestra.engine.graph import OrchestraGraph
from orchestra.engine.runners import planned_runner
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus
from orchestra.storage import db as db_module


def _graph(run: OrchestraRun) -> OrchestraGraph:
    graph = OrchestraGraph(run.run_id)
    graph.add_node("research", "Find where the HUD is built.", "kimi-deep")
    graph.add_node("outline", "Write the three bullets.", "explore")  # not an alias
    graph.add_edge("research", "outline", "on_success")
    return graph


def _run_planned(task: str):
    calls = []

    def fake_ask(alias, prompt, **kwargs):
        calls.append({"alias": alias, "prompt": prompt, "kwargs": kwargs})
        step = OrchestraRun(mode="ask", task=prompt, parent_run_id=kwargs.get("parent_run_id"))
        agent = AgentRun(alias=alias, provider="test", model="test/model", status=AgentStatus.COMPLETED)
        agent.stdout_log = f"result of step {len(calls)}"
        step.agents.append(agent)
        step.summary = agent.stdout_log
        step.status = RunStatus.COMPLETED
        return step

    with patch("orchestra.engine.planner.create_graph_plan", _graph), \
            patch("orchestra.engine.runners.ask_runner.run_ask", fake_ask), \
            patch.object(planned_runner, "_run_verification_loop"), \
            patch("orchestra.config.personas_config", return_value={"implementer": "cld-deep"}):
        run = planned_runner.run_planned(task)
    return run, calls


def test_a_step_gets_the_results_of_the_steps_before_it():
    _, calls = _run_planned("planned steps: outline a hint button (context)")

    assert "result of step 1" not in calls[0]["prompt"]
    assert "Write the three bullets." in calls[1]["prompt"]
    assert "result of step 1" in calls[1]["prompt"]


def test_step_answers_become_the_runs_answers():
    run, _ = _run_planned("planned steps: outline a hint button (answers)")

    outputs = {agent.alias: agent.stdout_log for agent in run.agents}
    assert outputs["research"] == "result of step 1"
    assert outputs["outline"] == "result of step 2"
    assert run.summary == "result of step 2"
    assert artifacts.read_agent_log(run.run_id, "outline") == "result of step 2"


def test_steps_are_children_and_an_unknown_agent_is_replaced():
    run, calls = _run_planned("planned steps: outline a hint button (children)")

    assert [call["kwargs"].get("parent_run_id") for call in calls] == [run.run_id, run.run_id]
    assert [call["alias"] for call in calls] == ["kimi-deep", "cld-deep"]


def test_the_run_list_leaves_steps_out_unless_asked():
    parent = OrchestraRun(mode="dual", task="run list parent")
    child = OrchestraRun(mode="ask", task="run list child", parent_run_id=parent.run_id)
    for run in (parent, child):
        run.status = RunStatus.COMPLETED
        db_module.upsert_run(run.to_manifest())

    listed = {row["run_id"] for row in db_module.list_runs_db(500)}
    everything = {row["run_id"] for row in db_module.list_runs_db(500, include_steps=True)}

    assert parent.run_id in listed and child.run_id not in listed
    assert {parent.run_id, child.run_id} <= everything


def test_a_connection_waits_for_a_lock_instead_of_failing():
    connection = db_module.get_db()
    try:
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000
    finally:
        connection.close()


def test_past_its_time_budget_a_planned_run_starts_no_more_steps(monkeypatch):
    monkeypatch.setattr("orchestra.config.planned_max_seconds", lambda: -1)

    run, calls = _run_planned("planned steps: outline a hint button (deadline)")

    assert len(calls) == 1
    assert run.summary == "result of step 1"
    events = [event["event"] for event in artifacts.read_events(run.run_id)]
    assert "planned_deadline_reached" in events

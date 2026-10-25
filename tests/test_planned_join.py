"""A planned run's join node waits for every branch that leads to it."""
from __future__ import annotations

from unittest.mock import patch

from orchestra.engine.graph import OrchestraGraph
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus


def _uneven_diamond() -> OrchestraGraph:
    graph = OrchestraGraph(run_id="plan-1")
    for name in "ABCDE":
        graph.add_node(name, f"do {name}", "cld-fast")
    for source, target in (("A", "B"), ("A", "C"), ("C", "E"), ("B", "D"), ("E", "D")):
        graph.add_edge(source, target)
    return graph


def test_a_node_waits_while_an_ancestor_is_still_pending():
    graph = _uneven_diamond()

    assert graph.has_pending_ancestor("D", {"E"}) is True
    assert graph.has_pending_ancestor("D", {"C"}) is True     # C leads to D through E
    assert graph.has_pending_ancestor("D", set()) is False
    assert graph.has_pending_ancestor("E", {"B"}) is False    # B is on the other branch


def _run_plan_in_order(graph: OrchestraGraph, fail: str = "") -> list[str]:
    from orchestra.engine.runners import planned_runner

    order: list[str] = []

    def fake_ask(alias, prompt, **kwargs):
        name = prompt.split("do ", 1)[1][0]
        order.append(name)
        step = OrchestraRun(mode="ask", task=prompt, created_at="2026-10-21T10:00:00Z")
        step.status = RunStatus.FAILED if name == fail else RunStatus.COMPLETED
        step.agents = [AgentRun(alias=alias, provider="p", model="m", status=AgentStatus.COMPLETED, stdout_log=name)]
        return step

    with patch("orchestra.engine.planner.create_graph_plan", return_value=graph), \
         patch.object(planned_runner, "_node_prompt", side_effect=lambda g, node: node.action), \
         patch("orchestra.engine.runners.ask_runner.run_ask", side_effect=fake_ask), \
         patch.object(planned_runner, "_run_verification_loop"), \
         patch.object(planned_runner, "_synthesize_if_possible", create=True):
        planned_runner.run_planned("plan it", emit_console=False, install_signal_handlers=False)
    return order


def test_the_join_runs_once_after_both_branches():
    order = _run_plan_in_order(_uneven_diamond())

    assert order.count("D") == 1
    assert order.index("D") > order.index("B") and order.index("D") > order.index("E")


def test_a_join_still_runs_when_a_branch_will_never_reach_it():
    graph = OrchestraGraph(run_id="plan-2")
    for name in "ABCDE":
        graph.add_node(name, f"do {name}", "cld-fast")
    for source, target in (("A", "B"), ("A", "C"), ("C", "E"), ("B", "D"), ("E", "D")):
        graph.add_edge(source, target, condition="on_success")

    order = _run_plan_in_order(graph, fail="C")   # E needs C to succeed: it never runs

    assert "E" not in order and order.count("D") == 1

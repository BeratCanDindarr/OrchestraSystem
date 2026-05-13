"""Tests for Dynamic DAG Conditional Edges + max_steps hard cap."""
from __future__ import annotations

import pytest

from orchestra.engine.graph import OrchestraGraph, Node, Edge, build_graph_from_plan


# ---------------------------------------------------------------------------
# Static graph — existing behavior unchanged
# ---------------------------------------------------------------------------

def test_add_node_and_edge():
    g = OrchestraGraph("run-001")
    g.add_node("n1", "Do task 1", "cdx-fast")
    g.add_node("n2", "Do task 2", "cdx-fast")
    g.add_edge("n1", "n2", "default")
    assert "n1" in g.nodes
    assert "n2" in g.nodes
    assert len(g.edges) == 1


def test_get_next_nodes_default():
    g = OrchestraGraph("run-001")
    g.add_node("n1", "task", "cdx-fast")
    g.add_node("n2", "task2", "cdx-fast")
    g.add_edge("n1", "n2", "default")
    assert g.get_next_nodes("n1", "completed") == ["n2"]
    assert g.get_next_nodes("n1", "failed") == ["n2"]


def test_get_next_nodes_on_success_only():
    g = OrchestraGraph("run-001")
    g.add_node("n1", "task", "cdx-fast")
    g.add_node("n2", "success path", "cdx-fast")
    g.add_node("n3", "failure path", "cdx-fast")
    g.add_edge("n1", "n2", "on_success")
    g.add_edge("n1", "n3", "on_failure")
    assert g.get_next_nodes("n1", "completed") == ["n2"]
    assert g.get_next_nodes("n1", "failed") == ["n3"]


# ---------------------------------------------------------------------------
# Dynamic node insertion — add_conditional_node
# ---------------------------------------------------------------------------

def test_add_conditional_node_success():
    g = OrchestraGraph("run-001")
    g.add_node("n1", "initial task", "cdx-fast")
    ok = g.add_conditional_node(
        after="n1",
        name="n2_dynamic",
        action="follow-up task",
        agent="cld-fast",
        condition="on_success",
    )
    assert ok is True
    assert "n2_dynamic" in g.nodes
    assert g.nodes["n2_dynamic"].dynamic is True
    assert any(e.source == "n1" and e.target == "n2_dynamic" for e in g.edges)


def test_add_conditional_node_duplicate_name_rejected():
    g = OrchestraGraph("run-001")
    g.add_node("n1", "task", "cdx-fast")
    g.add_node("n2", "task2", "cdx-fast")
    ok = g.add_conditional_node(after="n1", name="n2", action="dup", agent="cdx-fast")
    assert ok is False
    assert g.nodes["n2"].dynamic is False  # original unchanged


def test_add_conditional_node_unknown_source_rejected():
    g = OrchestraGraph("run-001")
    ok = g.add_conditional_node(after="nonexistent", name="new_node", action="task", agent="cdx-fast")
    assert ok is False
    assert "new_node" not in g.nodes


def test_conditional_node_appears_in_traversal():
    g = OrchestraGraph("run-001")
    g.add_node("n1", "task", "cdx-fast")
    g.add_conditional_node(after="n1", name="n2", action="dynamic task", agent="cld-fast", condition="on_success")
    next_nodes = g.get_next_nodes("n1", "completed")
    assert "n2" in next_nodes


def test_conditional_node_not_triggered_on_wrong_condition():
    g = OrchestraGraph("run-001")
    g.add_node("n1", "task", "cdx-fast")
    g.add_conditional_node(after="n1", name="n2", action="success only", agent="cld-fast", condition="on_success")
    next_nodes = g.get_next_nodes("n1", "failed")
    assert "n2" not in next_nodes


# ---------------------------------------------------------------------------
# max_steps hard cap
# ---------------------------------------------------------------------------

def test_record_step_increments():
    g = OrchestraGraph("run-001", max_steps=5)
    assert g._steps_executed == 0
    assert g.record_step() is True
    assert g._steps_executed == 1


def test_record_step_blocks_at_cap():
    g = OrchestraGraph("run-001", max_steps=3)
    for _ in range(3):
        assert g.record_step() is True
    assert g.record_step() is False
    assert g._steps_executed == 3  # not incremented past cap


def test_steps_remaining_property():
    g = OrchestraGraph("run-001", max_steps=10)
    g.record_step()
    g.record_step()
    assert g.steps_remaining == 8


def test_add_conditional_node_blocked_when_at_cap():
    g = OrchestraGraph("run-001", max_steps=2)
    g.add_node("n1", "task", "cdx-fast")
    g._steps_executed = 2  # exhaust the cap
    ok = g.add_conditional_node(after="n1", name="n2", action="blocked", agent="cdx-fast")
    assert ok is False
    assert "n2" not in g.nodes


# ---------------------------------------------------------------------------
# to_dict — includes new fields
# ---------------------------------------------------------------------------

def test_to_dict_includes_max_steps_and_steps_executed():
    g = OrchestraGraph("run-42", max_steps=15)
    g.add_node("n1", "task", "cdx-fast")
    g.record_step()
    d = g.to_dict()
    assert d["max_steps"] == 15
    assert d["steps_executed"] == 1
    assert "n1" in d["nodes"]


def test_to_dict_marks_dynamic_nodes():
    g = OrchestraGraph("run-42")
    g.add_node("n1", "static", "cdx-fast")
    g.add_conditional_node(after="n1", name="n2", action="dynamic", agent="cld-fast")
    d = g.to_dict()
    assert d["nodes"]["n1"]["dynamic"] is False
    assert d["nodes"]["n2"]["dynamic"] is True


# ---------------------------------------------------------------------------
# build_graph_from_plan — backward compat (no max_steps in old JSON)
# ---------------------------------------------------------------------------

def test_build_graph_from_plan_backward_compat():
    plan = '{"nodes": [{"name": "n1", "action": "t", "agent": "cdx-fast"}], "edges": []}'
    g = build_graph_from_plan("run-999", plan)
    assert g is not None
    assert "n1" in g.nodes
    assert g.max_steps == 20  # default


def test_build_graph_from_plan_invalid_json_returns_none():
    g = build_graph_from_plan("run-999", "not valid json{{{")
    assert g is None

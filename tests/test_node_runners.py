"""Tests for multi-agent node runners (Discussion / Consensus / Judge / Condition)."""
from orchestra.monitoring import node_runners as nr


def _stub():
    """A run_model stub that echoes which model saw what, so we can assert the flow."""
    calls = []

    def run(alias, prompt):
        calls.append((alias, prompt))
        return f"{alias}:reply"
    return run, calls


def test_run_discussion_debates_then_synthesises():
    run, calls = _stub()
    out = nr.run_discussion(["a", "b"], "solve X", rounds=2, run_model=run)
    # round1: a,b ; round2: a,b ; synthesis: a  → 5 calls, panel[0] synthesises
    assert len(calls) == 5
    assert calls[-1][0] == "a" and "Synthesise ONE consensus" in calls[-1][1]
    assert "solve X" in calls[-1][1]
    assert out == "a:reply"


def test_run_consensus_parallel_then_merge():
    run, calls = _stub()
    out = nr.run_consensus(["a", "b", "c"], "pick one", run_model=run)
    assert len(calls) == 4                       # 3 independent answers + 1 merge
    assert "Merge into ONE best answer" in calls[-1][1] and out == "a:reply"


def test_run_judge_reviews_with_criteria():
    run, calls = _stub()
    out = nr.run_judge("j", "no globals", "some code", run_model=run)
    assert len(calls) == 1 and "no globals" in calls[0][1] and "VERDICT" in calls[0][1]
    assert out == "j:reply"


def test_run_condition_emits_route_over_options():
    run, calls = _stub()
    nr.run_condition("c", "if urgent", "the input", ["Fast Path", "Slow Path"], run_model=run)
    assert "Fast Path, Slow Path" in calls[0][1] and "ROUTE:" in calls[0][1]


class _Node:
    def __init__(self, nid, ntype, panel=(), default_model="", rules="", content="", name="N"):
        self.id, self.node_type, self.panel = nid, ntype, tuple(panel)
        self.default_model, self.rules, self.content, self.name = default_model, rules, content, name


def test_dispatch_node_streams_and_completes(monkeypatch):
    from orchestra.monitoring.dispatch_inbox import DispatchInbox
    inbox = DispatchInbox()
    run, _calls = _stub()
    done = {}
    node = _Node("n1", "consensus", panel=["a", "b"], name="Panel")
    nr.dispatch_node(node, "task text", on_complete=lambda did: done.update(did=did),
                     inbox=inbox, roster=[], run_model=run)
    import time
    time.sleep(0.3)                              # runner is threaded
    d = inbox.get(done.get("did"))
    assert d is not None and d.status == "completed" and d.room_id == "n1"
    assert d.output == "a:reply"                # consensus merge result

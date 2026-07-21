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


class _RC:
    def __init__(self, cid):
        self.id = cid


def test_resolve_panel_uses_saved_then_falls_back():
    roster = [(_RC("m1"), "available"), (_RC("m2"), "idle"), (_RC("m3"), "offline")]
    assert nr._resolve_panel(_Node("n", "discussion", panel=["x", "y"]), roster) == ["x", "y"]
    # empty panel → up to 2 free (non-offline) models
    assert nr._resolve_panel(_Node("n", "discussion"), roster) == ["m1", "m2"]
    assert nr._resolve_panel(_Node("n", "discussion"), []) == []


def test_rounds_parsing_clamped():
    assert nr._rounds(_Node("n", "discussion", content="3")) == 3
    assert nr._rounds(_Node("n", "discussion", content="")) == 2      # default
    assert nr._rounds(_Node("n", "discussion", content="99")) == 5    # clamped high
    assert nr._rounds(_Node("n", "discussion", content="xx")) == 2    # non-numeric → default


def test_run_discussion_single_round_no_debate():
    run, calls = _stub()
    nr.run_discussion(["a", "b"], "task", rounds=1, run_model=run)
    # rounds=1 → only initial views (a,b) + synthesis (a) = 3 calls, no critique round
    assert len(calls) == 3


def test_discussion_empty_panel_returns_empty():
    run, calls = _stub()
    assert nr.run_discussion([], "task", run_model=run) == "" and calls == []


def test_dispatch_node_judge_picks_default_model(monkeypatch):
    from orchestra.monitoring.dispatch_inbox import DispatchInbox
    inbox = DispatchInbox()
    seen = {}

    def run(alias, prompt):
        seen["alias"] = alias
        return "ok"
    node = _Node("j", "judge", default_model="jm", rules="be strict", name="J")
    nr.dispatch_node(node, "review this", inbox=inbox, roster=[], run_model=run)
    import time
    time.sleep(0.2)
    assert seen.get("alias") == "jm"            # judge used its pinned model


def test_dispatch_node_failure_marks_failed_and_still_routes(monkeypatch):
    from orchestra.monitoring.dispatch_inbox import DispatchInbox
    inbox = DispatchInbox()
    routed = {}

    def boom(alias, prompt):
        raise RuntimeError("model down")
    node = _Node("n", "consensus", panel=["a"], name="P")
    nr.dispatch_node(node, "t", on_complete=lambda did: routed.update(did=did),
                     inbox=inbox, roster=[], run_model=boom)
    import time
    time.sleep(0.2)
    d = inbox.get(routed.get("did"))
    assert d is not None and d.status == "failed"    # error surfaced, on_complete still fired

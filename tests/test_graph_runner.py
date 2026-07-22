"""Tests for blueprint graph execution (Start → agents, prompt injection, output)."""
from orchestra.monitoring import graph_runner as gr
from orchestra.monitoring.room_store import RoomStore


def _capture(sink):
    def _fn(cid, task, *, context="", room_id=None, on_complete=None, **kw):
        sink.append({"cid": cid, "task": task, "context": context, "room_id": room_id})
        return {"ok": True, "dispatch_id": f"d{len(sink)}"}
    return _fn


class _C:
    def __init__(self, cid):
        self.id = cid


def test_fire_targets_routes_by_node_type(tmp_path, monkeypatch):
    store = RoomStore(path=tmp_path / "r.json")
    start = store.create("Start", workspace_id="w", node_type="start", content="go", now=1.0)
    disc = store.create("Debate", workspace_id="w", node_type="discussion", now=2.0)
    out = store.create("Out", workspace_id="w", node_type="output", now=3.0)
    store.update(start.id, connections=[disc.id])
    store.update(disc.id, connections=[out.id])

    seen = {}
    from orchestra.monitoring import node_runners
    monkeypatch.setattr(node_runners, "dispatch_node",
                        lambda node, text, **kw: seen.update(nid=node.id, ntype=node.node_type, text=text) or "d1")
    res = gr.play("w", store=store, roster_fn=lambda: [(_C("m1"), "idle")], dispatch_fn=_capture([]))
    # Start → discussion node dispatched via node_runners (not the single-agent path)
    assert res["ok"] is True
    assert seen == {"nid": disc.id, "ntype": "discussion", "text": "go"}


def test_gather_accumulates_incoming_branches(tmp_path, monkeypatch):
    monkeypatch.setattr(gr, "_GATHER_DEBOUNCE", 0.03)     # coalesce fast for the test
    store = RoomStore(path=tmp_path / "r.json")
    start = store.create("Start", workspace_id="w", node_type="start", content="t", now=1.0)
    g = store.create("Gather", workspace_id="w", node_type="gather", now=2.0)
    out = store.create("Out", workspace_id="w", node_type="output", now=3.0)
    store.update(start.id, connections=[g.id])
    store.update(g.id, connections=[out.id])
    gr.play("w", store=store, roster_fn=lambda: [(_C("m1"), "idle")], dispatch_fn=_capture([]))
    import time
    time.sleep(0.15)
    assert "t" in store.get(g.id).content and "t" in store.get(out.id).content


def _mk(store, **kw):
    return store.create(kw.pop("name", "N"), workspace_id="w", now=kw.pop("now", 1.0), **kw)


def test_condition_forwards_input_and_routes_by_decision(tmp_path):
    from orchestra.monitoring.dispatch_inbox import DispatchInbox
    store = RoomStore(path=tmp_path / "r.json")
    cond = _mk(store, name="Cond", node_type="condition", now=1.0)
    a = _mk(store, name="Fast Path", node_type="agent", now=2.0)
    b = _mk(store, name="Slow Path", node_type="agent", now=3.0)
    store.update(cond.id, connections=[a.id, b.id])
    inbox = DispatchInbox()
    did = inbox.add("m1", "in", room_id=cond.id)
    inbox.complete(did, "reasoning...\nROUTE: Slow Path", ok=True)   # decision output
    sink = []
    router = gr._make_router("w", 0, store=store, roster_fn=lambda: [(_C("m1"), "idle")],
                             dispatch_fn=_capture(sink), inbox=inbox, carry_input="ORIGINAL INPUT")
    router(did)
    # routed to Slow Path ONLY, and forwarded the ORIGINAL input (not the routing decision)
    assert len(sink) == 1 and sink[0]["room_id"] == b.id
    assert sink[0]["task"] == "ORIGINAL INPUT" and "ROUTE" not in sink[0]["task"]


def test_loop_fires_downstream_n_times(tmp_path):
    store = RoomStore(path=tmp_path / "r.json")
    start = _mk(store, name="Start", node_type="start", content="go", now=1.0)
    loop = _mk(store, name="Loop", node_type="loop", content="3", now=2.0)
    agent = _mk(store, name="A", node_type="agent", now=3.0)
    store.update(start.id, connections=[loop.id])
    store.update(loop.id, connections=[agent.id])
    sink = []
    gr.play("w", store=store, roster_fn=lambda: [(_C("m1"), "idle")], dispatch_fn=_capture(sink))
    assert len([s for s in sink if s["room_id"] == agent.id]) == 3      # looped 3×


def test_non_router_source_broadcasts_to_all_targets(tmp_path):
    store = RoomStore(path=tmp_path / "r.json")
    start = _mk(store, name="Start", node_type="start", content="go", now=1.0)
    a = _mk(store, name="A", node_type="agent", now=2.0)
    b = _mk(store, name="B", node_type="agent", now=3.0)
    store.update(start.id, connections=[a.id, b.id])
    sink = []
    gr.play("w", store=store, roster_fn=lambda: [(_C("m1"), "idle")], dispatch_fn=_capture(sink))
    assert {s["room_id"] for s in sink} == {a.id, b.id}                # both fired


def test_agent_uses_default_model_else_autocast(tmp_path):
    store = RoomStore(path=tmp_path / "r.json")
    start = _mk(store, name="Start", node_type="start", content="go", now=1.0)
    a = _mk(store, name="A", node_type="agent", now=2.0)
    store.update(a.id, default_model="m2")
    store.update(start.id, connections=[a.id])
    sink = []
    gr.play("w", store=store, roster_fn=lambda: [(_C("m1"), "idle"), (_C("m2"), "idle")],
            dispatch_fn=_capture(sink))
    assert sink and sink[0]["cid"] == "m2"                              # pinned model wins


def test_run_to_completion_collects_output(tmp_path, monkeypatch):
    from orchestra.monitoring import characters
    monkeypatch.setattr(characters, "_default_dispatch_stream",
                        lambda alias, prompt: iter(["model output"]))
    store = RoomStore(path=tmp_path / "r.json")
    start = _mk(store, name="Start", node_type="start", content="do research", now=1.0)
    agent = _mk(store, name="Dev", node_type="agent", now=2.0)
    store.update(agent.id, default_model="m1")
    out = _mk(store, name="Result", node_type="output", now=3.0)
    store.update(start.id, connections=[agent.id])
    store.update(agent.id, connections=[out.id])
    events = []
    res = gr.run_to_completion("w", store=store, roster_fn=lambda: [(_C("m1"), "idle")],
                               stable_for=0.25, poll=0.03, timeout=8,
                               on_complete=lambda d, n: events.append((n, d.status)))
    assert res["ok"] is True
    assert events == [("Dev", "completed")]                        # streamed progress
    assert any("model output" in c for _n, c in res["outputs"])    # Output node collected it


def test_run_to_completion_reports_bad_graph(tmp_path):
    store = RoomStore(path=tmp_path / "r.json")
    _mk(store, name="Start", node_type="start", content="", now=1.0)   # no task
    res = gr.run_to_completion("w", store=store, roster_fn=lambda: [(_C("m1"), "idle")],
                               stable_for=0.1, timeout=2)
    assert res["ok"] is False and "task" in res["error"]


def test_depth_cap_guard_stops_firing(tmp_path):
    store = RoomStore(path=tmp_path / "r.json")
    start = _mk(store, name="Start", node_type="start", content="go", now=1.0)
    a = _mk(store, name="A", node_type="agent", now=2.0)
    store.update(start.id, connections=[a.id])
    nodes = store.list()
    nbi = {n.id: n for n in nodes}
    sink = []
    # at the depth cap, no further targets fire (cycle/runaway guard)
    out = gr._fire_targets(start, "go", nodes, nbi, depth=gr._MAX_DEPTH, store=store,
                           roster_fn=lambda: [(_C("m1"), "idle")], dispatch_fn=_capture(sink), inbox=None)
    assert out == [] and sink == []


def test_incoming_prompts_and_agent_context(tmp_path):
    store = RoomStore(path=tmp_path / "r.json")
    agent = store.create("A", preset="researcher", workspace_id="w", node_type="agent", now=1.0)
    p1 = store.create("P1", workspace_id="w", node_type="prompt", content="Be terse.", now=2.0)
    p2 = store.create("P2", workspace_id="w", node_type="prompt", content="Cite sources.", now=3.0)
    store.update(p1.id, connections=[agent.id])
    store.update(p2.id, connections=[agent.id])
    nodes = store.list()
    ctx = gr.agent_context(store.get(agent.id), nodes)
    assert "rigorous researcher" in ctx and "Be terse." in ctx and "Cite sources." in ctx


def test_play_requires_start_and_task(tmp_path):
    store = RoomStore(path=tmp_path / "r.json")
    # no start node
    res = gr.play("w", store=store, roster_fn=lambda: [(_C("m1"), "idle")], dispatch_fn=_capture([]))
    assert res["ok"] is False and "Start" in res["error"]
    # start with no task
    s = store.create("Start", workspace_id="w", node_type="start", content="", now=1.0)
    res2 = gr.play("w", store=store, roster_fn=lambda: [(_C("m1"), "idle")], dispatch_fn=_capture([]))
    assert res2["ok"] is False and "task" in res2["error"]


def test_play_dispatches_start_downstream_agents_with_prompt(tmp_path):
    store = RoomStore(path=tmp_path / "r.json")
    start = store.create("Start", workspace_id="w", node_type="start", content="build X", now=1.0)
    agent = store.create("Dev", preset="senior_engineer", workspace_id="w", node_type="agent", now=2.0)
    prompt = store.create("P", workspace_id="w", node_type="prompt", content="No globals.", now=3.0)
    store.update(start.id, connections=[agent.id])
    store.update(prompt.id, connections=[agent.id])
    sink = []
    res = gr.play("w", store=store, roster_fn=lambda: [(_C("m1"), "idle")], dispatch_fn=_capture(sink))
    assert res["ok"] is True and res["started"] == 1
    assert sink[0]["room_id"] == agent.id and sink[0]["task"] == "build X"
    assert "senior software engineer" in sink[0]["context"] and "No globals." in sink[0]["context"]


def test_play_reports_unconnected_start(tmp_path):
    store = RoomStore(path=tmp_path / "r.json")
    store.create("Start", workspace_id="w", node_type="start", content="do it", now=1.0)
    res = gr.play("w", store=store, roster_fn=lambda: [(_C("m1"), "idle")], dispatch_fn=_capture([]))
    assert res["ok"] is False and "runnable" in res["error"]


def test_output_node_collects_result(tmp_path):
    from orchestra.monitoring.dispatch_inbox import DispatchInbox
    store = RoomStore(path=tmp_path / "r.json")
    agent = store.create("Dev", workspace_id="w", node_type="agent", now=1.0)
    out = store.create("Out", workspace_id="w", node_type="output", now=2.0)
    store.update(agent.id, connections=[out.id])
    inbox = DispatchInbox()
    did = inbox.add("m1", "task", room_id=agent.id)
    inbox.complete(did, "final answer", ok=True)
    # simulate the agent's completion router firing
    router = gr._make_router("w", 0, store=store, roster_fn=lambda: [(_C("m1"), "idle")],
                             dispatch_fn=_capture([]), inbox=inbox)
    router(did)
    assert store.get(out.id).content == "final answer"     # output node captured the result

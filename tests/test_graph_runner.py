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


def test_gather_accumulates_incoming_branches(tmp_path):
    store = RoomStore(path=tmp_path / "r.json")
    start = store.create("Start", workspace_id="w", node_type="start", content="t", now=1.0)
    g = store.create("Gather", workspace_id="w", node_type="gather", now=2.0)
    out = store.create("Out", workspace_id="w", node_type="output", now=3.0)
    store.update(start.id, connections=[g.id])
    store.update(g.id, connections=[out.id])
    gr.play("w", store=store, roster_fn=lambda: [(_C("m1"), "idle")], dispatch_fn=_capture([]))
    # Start's text lands in the gather node and is forwarded to the output node
    assert "t" in store.get(g.id).content and "t" in store.get(out.id).content


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

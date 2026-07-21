"""Tests for Phase 2 room routing (hand-off between connected rooms)."""
from orchestra.monitoring import room_router as rr
from orchestra.monitoring.dispatch_inbox import DispatchInbox
from orchestra.monitoring.room_store import RoomStore


class _R:
    def __init__(self, rid, name, preset="custom", connections=()):
        self.id, self.name, self.preset, self.rules = rid, name, preset, ""
        self.connections = tuple(connections)


def test_parse_route_prefers_directive():
    targets = [_R("r2", "Software Room"), _R("r3", "Research Room")]
    assert rr.parse_route("blah blah\nROUTE: Research Room", targets).id == "r3"
    assert rr.parse_route("NEXT: Software Room", targets).id == "r2"


def test_parse_route_falls_back_to_last_mention_or_none():
    targets = [_R("r2", "Software Room"), _R("r3", "Research Room")]
    # no directive → last-mentioned target name wins
    assert rr.parse_route("first Software Room ... then Research Room", targets).id == "r3"
    assert rr.parse_route("nothing relevant here", targets) is None


def _capture_dispatch(sink):
    def _fn(cid, task, *, context="", room_id=None, on_complete=None, **kw):
        sink.append({"cid": cid, "task": task, "room_id": room_id, "context": context})
        return {"ok": True, "dispatch_id": f"d-{len(sink)}"}
    return _fn


def test_route_completion_decision_routes_to_one(tmp_path):
    store = RoomStore(path=tmp_path / "rooms.json")
    decision = store.create("Decision Room", preset="decision_maker", now=1.0)
    soft = store.create("Software Room", preset="senior_engineer", now=2.0)
    research = store.create("Research Room", preset="researcher", now=3.0)
    store.update(decision.id, connections=[soft.id, research.id])

    inbox = DispatchInbox()
    did = inbox.add("m1", "decide", room_id=decision.id)
    inbox.complete(did, "Reasoning...\nROUTE: Research Room", ok=True)

    sink = []
    roster = [(_R("m1", "Ada"), "idle")]
    handoffs = rr.route_completion(did, inbox=inbox, store=store,
                                   roster_fn=lambda: roster, dispatch_fn=_capture_dispatch(sink))
    # decision routed to exactly ONE room (Research), passing the output as the new task
    assert [h[0] for h in handoffs] == [research.id]
    assert len(sink) == 1 and sink[0]["room_id"] == research.id
    assert "ROUTE: Research Room" in sink[0]["task"]
    assert "rigorous researcher" in sink[0]["context"]      # target room's preset injected


def test_route_completion_non_decision_broadcasts(tmp_path):
    store = RoomStore(path=tmp_path / "rooms.json")
    intake = store.create("Intake", preset="custom", now=1.0)
    a = store.create("A", now=2.0)
    b = store.create("B", now=3.0)
    store.update(intake.id, connections=[a.id, b.id])

    inbox = DispatchInbox()
    did = inbox.add("m1", "work", room_id=intake.id)
    inbox.complete(did, "some output", ok=True)

    sink = []
    handoffs = rr.route_completion(did, inbox=inbox, store=store,
                                   roster_fn=lambda: [(_R("m1", "Ada"), "idle")],
                                   dispatch_fn=_capture_dispatch(sink))
    assert {h[0] for h in handoffs} == {a.id, b.id}         # broadcast to all connections
    assert len(sink) == 2


def test_route_completion_stops_without_connections_or_when_failed(tmp_path):
    store = RoomStore(path=tmp_path / "rooms.json")
    lone = store.create("Lone", now=1.0)
    inbox = DispatchInbox()
    did = inbox.add("m1", "x", room_id=lone.id)
    inbox.complete(did, "out", ok=True)
    assert rr.route_completion(did, inbox=inbox, store=store,
                               roster_fn=lambda: [(_R("m1", "Ada"), "idle")],
                               dispatch_fn=_capture_dispatch([])) == []
    # failed dispatch never routes
    with_conn = store.create("Src", now=2.0)
    tgt = store.create("Tgt", now=3.0)
    store.update(with_conn.id, connections=[tgt.id])
    did2 = inbox.add("m1", "x", room_id=with_conn.id)
    inbox.complete(did2, "boom", ok=False)
    assert rr.route_completion(did2, inbox=inbox, store=store,
                               roster_fn=lambda: [(_R("m1", "Ada"), "idle")],
                               dispatch_fn=_capture_dispatch([])) == []


def test_route_completion_depth_cap(tmp_path):
    store = RoomStore(path=tmp_path / "rooms.json")
    a = store.create("A", now=1.0)
    b = store.create("B", now=2.0)
    store.update(a.id, connections=[b.id])
    inbox = DispatchInbox()
    did = inbox.add("m1", "x", room_id=a.id)
    inbox.complete(did, "out", ok=True)
    # at the depth cap, no further hand-offs are made
    assert rr.route_completion(did, depth=rr._MAX_DEPTH, inbox=inbox, store=store,
                               roster_fn=lambda: [(_R("m1", "Ada"), "idle")],
                               dispatch_fn=_capture_dispatch([])) == []

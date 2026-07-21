"""Tests for room presets + persistent room store (Phase 1: editable rooms)."""
from orchestra.monitoring import room_presets as rp
from orchestra.monitoring.room_store import Room, RoomStore


def test_presets_include_starter_roles_and_custom_first():
    keys = {p.key for p in rp.preset_list()}
    assert {"custom", "senior_engineer", "product_manager", "researcher",
            "reviewer", "architect", "decision_maker"} <= keys
    assert rp.preset_list()[0].key == "custom"          # custom sorts first
    assert rp.get_preset("SENIOR_ENGINEER").name == "Senior Engineer"  # case-insensitive
    assert rp.get_preset("nope") is None


def test_compose_context_combines_preset_and_rules():
    ctx = rp.compose_context("senior_engineer", "Always write tests first.")
    assert "senior software engineer" in ctx and "Always write tests first." in ctx
    # custom preset with no prompt → only the rules
    assert rp.compose_context("custom", "Be terse.") == "Be terse."
    assert rp.compose_context("custom", "") == ""


def test_store_create_get_list_persists(tmp_path):
    store = RoomStore(path=tmp_path / "rooms.json")
    assert store.list() == []
    r = store.create("Research Room", preset="researcher", rules="Cite sources.", now=1000.0)
    assert r.name == "Research Room" and r.preset == "researcher"
    # a fresh store over the same file sees the persisted room
    reopened = RoomStore(path=tmp_path / "rooms.json")
    got = reopened.get(r.id)
    assert got is not None and got.rules == "Cite sources."


def test_store_update_is_immutable_copy(tmp_path):
    store = RoomStore(path=tmp_path / "rooms.json")
    r = store.create("Software Room", preset="custom", now=1.0)
    updated = store.update(r.id, name="Dev Room", preset="senior_engineer", rules="No globals.")
    assert updated is not None
    assert updated.name == "Dev Room" and updated.preset == "senior_engineer"
    assert updated.created_at == r.created_at            # unchanged field preserved
    assert store.get(r.id).rules == "No globals."
    assert store.update("missing", name="x") is None


def test_store_delete_drops_dangling_connections(tmp_path):
    store = RoomStore(path=tmp_path / "rooms.json")
    a = store.create("A", now=1.0)
    b = store.create("B", now=2.0)
    store.update(a.id, connections=[b.id])
    assert store.get(a.id).connections == (b.id,)
    assert store.delete(b.id) is True
    assert store.get(a.id).connections == ()             # dangling edge cleaned up
    assert store.delete("missing") is False


def test_room_roundtrip_dict():
    r = Room(id="x", name="N", preset="architect", rules="r", created_at=5.0, connections=("y",))
    assert Room.from_dict(r.to_dict()) == r

"""Tests for the persistent workspace store (multiple workspaces)."""
from orchestra.monitoring.workspace_store import Workspace, WorkspaceStore


def test_create_activates_and_persists(tmp_path):
    store = WorkspaceStore(path=tmp_path / "ws.json")
    assert store.list() == [] and store.get_active() is None
    a = store.create("Research", now=1.0)
    assert store.get_active().id == a.id                 # first create becomes active
    b = store.create("Shipping", now=2.0)
    assert store.get_active().id == b.id                 # new create activates by default
    # persisted across a fresh store over the same file
    reopened = WorkspaceStore(path=tmp_path / "ws.json")
    assert {w.name for w in reopened.list()} == {"Research", "Shipping"}
    assert reopened.get_active().id == b.id


def test_set_active_and_rename(tmp_path):
    store = WorkspaceStore(path=tmp_path / "ws.json")
    a = store.create("A", now=1.0)
    b = store.create("B", now=2.0)
    assert store.set_active(a.id) is True
    assert store.get_active().id == a.id
    assert store.set_active("missing") is False
    renamed = store.rename(a.id, "Alpha")
    assert renamed.name == "Alpha" and store.get(a.id).name == "Alpha"


def test_delete_refuses_last_and_reassigns_active(tmp_path):
    store = WorkspaceStore(path=tmp_path / "ws.json")
    a = store.create("A", now=1.0)
    b = store.create("B", now=2.0)      # active = B
    assert store.delete(b.id) is True    # deleting active reassigns active to remaining
    assert store.get_active().id == a.id
    assert store.delete(a.id) is False   # refuse to delete the last workspace
    assert len(store.list()) == 1


def test_ensure_default_bootstraps(tmp_path):
    store = WorkspaceStore(path=tmp_path / "ws.json")
    ws = store.ensure_default()
    assert ws.name == "Default" and store.get_active().id == ws.id
    assert store.ensure_default().id == ws.id            # idempotent


def test_workspace_roundtrip():
    w = Workspace(id="x", name="N", created_at=3.0)
    assert Workspace.from_dict(w.to_dict()) == w

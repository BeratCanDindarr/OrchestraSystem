"""Tests for the `orchestra graph` CLI (run dashboard workspaces from the terminal)."""
from typer.testing import CliRunner

from orchestra.cli import app

runner = CliRunner()


def _seed(tmp_path, monkeypatch):
    from orchestra.monitoring import characters, room_store, workspace_store
    monkeypatch.setattr(workspace_store, "_store", workspace_store.WorkspaceStore(path=tmp_path / "w.json"))
    monkeypatch.setattr(room_store, "_store", room_store.RoomStore(path=tmp_path / "r.json"))
    monkeypatch.setattr(characters, "_default_dispatch_stream", lambda a, p: iter(["terminal result"]))
    monkeypatch.setattr(characters, "build_roster",
                        lambda **k: [characters.Character("m1", "Ada", "C", "codex", "x", True, "<svg/>")])
    ws = workspace_store.get_workspace_store().create("Deep Think", now=1.0)
    st = room_store.get_room_store()
    start = st.create("Start", workspace_id=ws.id, node_type="start", content="think hard", now=1.0)
    ag = st.create("Dev", workspace_id=ws.id, node_type="agent", now=2.0)
    st.update(ag.id, default_model="m1")
    out = st.create("Result", workspace_id=ws.id, node_type="output", now=3.0)
    st.update(start.id, connections=[ag.id])
    st.update(ag.id, connections=[out.id])
    return ws


def test_graph_list_shows_workspaces(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    res = runner.invoke(app, ["graph", "list"])
    assert res.exit_code == 0 and "Deep Think" in res.output


def test_graph_show_lists_nodes(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    res = runner.invoke(app, ["graph", "show", "Deep Think"])
    assert res.exit_code == 0
    assert "start" in res.output.lower() and "Dev" in res.output and "Result" in res.output


def test_graph_run_streams_and_prints_output(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    res = runner.invoke(app, ["graph", "run", "Deep Think"])
    assert res.exit_code == 0
    assert "Running" in res.output and "Result" in res.output and "terminal result" in res.output


def test_graph_run_unknown_workspace(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    res = runner.invoke(app, ["graph", "run", "Nope"])
    assert res.exit_code == 1 and "No workspace" in res.output

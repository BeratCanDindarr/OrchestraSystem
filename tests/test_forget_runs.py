"""Erasing a run takes everything that belongs to it and nothing else."""
from __future__ import annotations

from orchestra import config, service
from orchestra.engine import artifacts
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus
from orchestra.storage import db as db_module


def _stored(task: str, status: RunStatus = RunStatus.COMPLETED, parent: str | None = None) -> OrchestraRun:
    run = OrchestraRun(mode="ask", task=task, parent_run_id=parent)
    run.agents.append(AgentRun(alias="cld-deep", provider="test", model="test/model", status=AgentStatus.COMPLETED))
    run.status = status
    artifacts.write_manifest(run)
    artifacts.write_agent_log(run.run_id, "cld-deep", "an answer")
    db_module.upsert_run(run.to_manifest())
    return run


def _on_disk(run_id: str) -> bool:
    # not artifacts.run_dir: asking for a run's folder creates it
    return (config.artifact_root() / run_id).exists()


def _in_db(run_id: str) -> bool:
    connection = db_module.get_db()
    try:
        return connection.execute("SELECT COUNT(*) FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0] > 0
    finally:
        connection.close()


def test_a_run_goes_with_its_steps_and_its_rows():
    run = _stored("forget: the run")
    step = _stored("forget: its verifier", parent=run.run_id)
    other = _stored("forget: another run")

    result = service.forget_runs([run.run_id])

    assert sorted(result["forgotten"]) == sorted([run.run_id, step.run_id])
    assert not _on_disk(run.run_id) and not _on_disk(step.run_id)
    assert not _in_db(run.run_id) and not _in_db(step.run_id)
    assert _on_disk(other.run_id) and _in_db(other.run_id)


def test_a_running_run_is_left_alone():
    run = _stored("forget: still going", status=RunStatus.RUNNING)

    result = service.forget_runs([run.run_id, "nope0000"])

    assert result["forgotten"] == []
    assert {item["reason"] for item in result["skipped"]} == {"still running: cancel it first", "not found"}
    assert _on_disk(run.run_id)


# ── forgetting a run leaves its text in no store ────────────────────────────

import json  # noqa: E402
import sqlite3  # noqa: E402

import pytest  # noqa: E402

SECRET = "the launch date is the fourteenth of march"


def _text_anywhere(path, needle: str) -> bool:
    """Whether any text column of any table of a SQLite file still holds *needle*."""
    connection = sqlite3.connect(str(path))
    try:
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        for table in tables:
            try:
                rows = connection.execute(f'SELECT * FROM "{table}"').fetchall()
            except sqlite3.Error:
                continue
            if any(needle in str(value) for row in rows for value in row):
                return True
        return False
    finally:
        connection.close()


@pytest.fixture
def stores(tmp_path, monkeypatch, fake_embeddings):
    """A project of its own: run folder, databases, memory and answer cache."""
    from orchestra.engine.runners import common

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr("orchestra.config.orchestra_root", lambda: tmp_path)
    monkeypatch.setattr("orchestra.config.artifact_root", lambda: runs)
    monkeypatch.setattr("orchestra.config.db_path", lambda: tmp_path / "orchestra.db")
    monkeypatch.setattr("orchestra.config.memory_db_path", lambda: tmp_path / "memory.db")
    monkeypatch.setattr("orchestra.config.fts_cache_db_path", lambda: str(tmp_path / "fts_cache.db"))
    monkeypatch.setattr(db_module, "_event_log_instance", None)
    monkeypatch.setattr(common, "_event_log", None)
    monkeypatch.setattr(common, "_fts_cache", None)
    return tmp_path


def _answered(run, pairs, **kwargs):
    agent = AgentRun(alias=pairs[0][0], provider="codex", model="m", status=AgentStatus.COMPLETED)
    agent.stdout_log = f"Understood: {SECRET}."
    run.agents.append(agent)


def test_forgetting_a_run_removes_its_text_from_every_store(stores, monkeypatch):
    from unittest.mock import patch

    from orchestra.engine.runners import ask_runner, common

    with patch.object(common, "run_parallel", _answered):
        run = ask_runner.run_ask("cdx-deep", f"remember that {SECRET}")
    assert run.status == RunStatus.COMPLETED
    databases = ("orchestra.db", "memory.db", "fts_cache.db")
    assert any(_text_anywhere(stores / name, SECRET) for name in databases)      # it was stored somewhere

    service.forget_runs([run.run_id])

    for name in databases:
        assert not _text_anywhere(stores / name, SECRET), name
    with patch.object(common, "run_parallel", _answered) as asked_again:
        again = ask_runner.run_ask("cdx-deep", f"remember that {SECRET}")
    assert again.run_id != run.run_id                                # not replayed from a store
    asked_again.assert_called_once() if hasattr(asked_again, "assert_called_once") else None


def test_forgetting_a_run_finds_its_row_in_typed_memory(stores, monkeypatch):
    from orchestra import config as _config
    from orchestra.storage.memory import get_memory

    monkeypatch.setitem(_config._cfg, "memory", {"typed_enabled": True})
    memory = get_memory()
    memory.add("a note the owner wrote", source="note", memory_type="semantic")
    run = _stored("forget: typed")
    memory.add(run.task, metadata={"run_id": run.run_id}, source=run.run_id, memory_type="episodic")

    service.forget_runs([run.run_id])

    contents = [row["content"] for row in get_memory().get_memory_ledger()]
    assert contents == ["a note the owner wrote"]

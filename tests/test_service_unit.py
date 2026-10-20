"""Unit tests for orchestra.service."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from orchestra.utils.config_values import parse_cli_value, toml_literal

from orchestra import service

def test_parse_cli_value():
    assert parse_cli_value("true") is True
    assert parse_cli_value("false") is False
    assert parse_cli_value("123") == 123
    assert parse_cli_value("12.3") == 12.3
    assert parse_cli_value("[1, 2, 3]") == [1, 2, 3]
    assert parse_cli_value("hello") == "hello"

def test_toml_literal():
    assert toml_literal(True) == "true"
    assert toml_literal(123) == "123"
    assert toml_literal("hello") == '"hello"'
    assert toml_literal([1, 2]) == "[1, 2]"
    assert toml_literal({"a": 1}) == "{ a = 1 }"      # an inline table, not JSON

def test_serialize_rows():
    rows = [{"a": 1}, {"b": 2}]
    res = service._serialize_rows(rows)
    assert res == rows


def test_list_aliases():
    with patch("orchestra.service.available_aliases", return_value=["✓ a1"]):
        res = service.list_aliases()
        assert "a1" in res["aliases"][0]


def test_list_jobs():
    with patch("orchestra.service.list_job_records", return_value=[{"job_id": "j1"}]):
        with patch("orchestra.service.get_db"):
            res = service.list_jobs()
            assert len(res["jobs"]) == 1
            assert res["jobs"][0]["job_id"] == "j1"

def test_get_job():
    with patch("orchestra.service.get_job_record", return_value={"job_id": "j1"}):
        with patch("orchestra.service.get_db"):
            res = service.get_job("j1")
            assert res["job"]["job_id"] == "j1"


# ── reading the history does not rewrite it ─────────────────────────────────

@pytest.fixture
def fresh_project(tmp_path, monkeypatch):
    """A project with its own empty database and run folder."""
    from orchestra.storage import db

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr("orchestra.config.db_path", lambda: tmp_path / "orchestra.db")
    monkeypatch.setattr("orchestra.config.artifact_root", lambda: runs)
    monkeypatch.setattr(db, "_event_log_instance", None)
    return runs


def _recorded_run(run_id: str = "run-hist") -> str:
    from orchestra.engine import artifacts
    from orchestra.models import OrchestraRun, RunStatus

    run = OrchestraRun(mode="ask", task="history", run_id=run_id, status=RunStatus.COMPLETED)
    artifacts.write_manifest(run)
    for name in ("run_started", "agent_started", "run_finished"):
        artifacts.append_event(run_id, {"event": name})
    return run_id


def _event_seqs(run_id: str) -> list[int]:
    from orchestra.storage.db import db_connection

    with db_connection() as conn:
        return [row[0] for row in conn.execute("SELECT seq FROM events WHERE run_id = ? ORDER BY seq", (run_id,))]


def test_listing_runs_leaves_the_event_log_as_it_is(fresh_project):
    run_id = _recorded_run()
    before = _event_seqs(run_id)

    listed = service.list_runs()

    assert [row["run_id"] for row in listed["runs"]] == [run_id]
    assert len(before) == 3 and _event_seqs(run_id) == before


def test_backfill_twice_keeps_the_sequence_numbers(fresh_project):
    from orchestra.storage.db import backfill

    run_id = _recorded_run()
    before = _event_seqs(run_id)

    backfill()
    backfill()

    assert _event_seqs(run_id) == before


def test_backfill_keeps_events_whose_file_is_gone(fresh_project):
    from orchestra.storage.db import backfill

    run_id = _recorded_run()
    (fresh_project / run_id / "events.jsonl").unlink()

    backfill()

    assert len(_event_seqs(run_id)) == 3


def test_an_empty_index_is_rebuilt_from_the_run_folders(fresh_project):
    run_dir = fresh_project / "run-old"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps(
        {"run_id": "run-old", "mode": "ask", "status": "completed", "task": "from before the index",
         "created_at": "2026-01-01T00:00:00Z", "agents": []}))
    (run_dir / "events.jsonl").write_text('{"event": "run_started"}\n{"event": "run_finished"}\n')

    listed = service.list_runs()

    assert [row["run_id"] for row in listed["runs"]] == ["run-old"]
    assert len(_event_seqs("run-old")) == 2


def test_listing_runs_keeps_events_whose_file_is_gone(fresh_project):
    run_id = _recorded_run()
    (fresh_project / run_id / "events.jsonl").unlink()

    service.list_runs()

    assert len(_event_seqs(run_id)) == 3


def test_reading_the_history_does_not_reindex_every_run_folder(fresh_project):
    _recorded_run()

    with patch("orchestra.service.backfill") as reindex:
        service.list_runs()

    reindex.assert_not_called()


# ── runs whose process died are not RUNNING for ever ────────────────────────

def _running_run(run_id: str, *, minutes_old: int, pid=None) -> str:
    from datetime import datetime, timedelta, timezone

    from orchestra.engine import artifacts

    stamp = (datetime.now(timezone.utc) - timedelta(minutes=minutes_old)).isoformat()
    artifacts.write_manifest_data(run_id, {
        "run_id": run_id, "mode": "ask", "status": "running", "task": "stale probe",
        "created_at": stamp, "updated_at": stamp,
        "agents": [{"alias": "cdx-deep", "provider": "codex", "status": "started", "pid": pid}],
    })
    return run_id


def test_a_running_run_nobody_is_working_on_is_reaped(fresh_project):
    import os

    from orchestra.engine import artifacts

    dead = _running_run("run-dead", minutes_old=600)
    fresh = _running_run("run-fresh", minutes_old=1)
    alive = _running_run("run-alive", minutes_old=600, pid=os.getpid())

    reaped = service.reap_stale_runs()

    assert reaped == [dead]
    manifest = artifacts.load_manifest(dead)
    assert manifest["status"] == "failed"
    assert manifest["agents"][0]["status"] == "failed"
    assert "interrupted" in manifest["failure"]["message"]
    assert artifacts.load_manifest(fresh)["status"] == "running"
    assert artifacts.load_manifest(alive)["status"] == "running"


def test_listing_runs_reaps_first(fresh_project):
    _running_run("run-dead", minutes_old=600)

    listed = service.list_runs()

    assert [(row["run_id"], row["status"]) for row in listed["runs"]] == [("run-dead", "failed")]


# ── writing the config file ─────────────────────────────────────────────────

def test_what_the_config_writer_writes_reads_back_as_the_same_config():
    """`config set` rewrites the whole file: every key and value in it must survive that."""
    import tomllib

    from orchestra.utils.config_values import flatten_toml_lines

    tree = {
        "orchestra": {"timeout": 180, "ratio": 0.5, "on": True},
        "pricing": {"gpt-5.4/low": 0.002, "ollama/glm-5.3:cloud": 0.003,
                    "claude/opus": {"input": 0.015, "output": 0.075}},
        "providers": {"claude": {"tools": ["Read", "Grep"],
                                 "append_system_prompt": 'Say "hi".\nThen stop.\tDone \\ here'}},
        "remote_nodes": {'pc]\n[dashboard]\ndispatch_enabled = true\n[x': {"url": "http://10.0.0.9:7070"}},
        "mcp": {"client_servers": [{"name": "docs", "command": ["npx", "docs-server"]}]},
    }

    written = "\n".join(flatten_toml_lines(tree)) + "\n"

    assert tomllib.loads(written) == tree
    assert "dashboard" not in tomllib.loads(written)        # a hostile key did not become a section

"""Tests for the deterministic replay system."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from orchestra.engine.replay import (
    ReplayRecorder,
    ReplayRunner,
    get_recorder,
    _replay_path,
)
from orchestra.models import OrchestraRun, RunStatus


@pytest.fixture(autouse=True)
def _clean_recorders(monkeypatch, tmp_path):
    """Clear recorder cache and redirect replay logs to tmp_path."""
    from orchestra.engine import replay as _replay_mod
    _replay_mod._recorders.clear()
    monkeypatch.setattr(
        "orchestra.engine.replay._replay_path",
        lambda run_id: tmp_path / run_id / "replay.jsonl",
    )


class TestReplayRecorder:
    """Append-only replay log recording."""

    def test_record_run_started(self, tmp_path: Path) -> None:
        with patch("orchestra.engine.replay.config.replay_enabled", return_value=True):
            rec = ReplayRecorder("run-001")
            rec.record_run_started("dual", "test task")
            assert rec._path.exists()
            lines = rec._path.read_text().strip().splitlines()
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["type"] == "run_started"
            assert entry["mode"] == "dual"
            assert entry["task"] == "test task"
            assert "seq" in entry

    def test_record_agent_dispatch(self, tmp_path: Path) -> None:
        with patch("orchestra.engine.replay.config.replay_enabled", return_value=True):
            rec = ReplayRecorder("run-002")
            rec.record_agent_dispatch("cdx-deep", "prompt text", "gpt-5", "codex")
            lines = rec._path.read_text().strip().splitlines()
            entry = json.loads(lines[0])
            assert entry["type"] == "agent_dispatch"
            assert entry["alias"] == "cdx-deep"
            assert entry["prompt"] == "prompt text"

    def test_record_agent_completed(self, tmp_path: Path) -> None:
        with patch("orchestra.engine.replay.config.replay_enabled", return_value=True):
            rec = ReplayRecorder("run-003")
            rec.record_agent_completed("cdx-deep", "stdout output", 0)
            lines = rec._path.read_text().strip().splitlines()
            entry = json.loads(lines[0])
            assert entry["type"] == "agent_completed"
            assert entry["alias"] == "cdx-deep"
            assert entry["returncode"] == 0
            assert entry["stdout_digest"].startswith("sha256:")

    def test_record_tool_call_inline(self, tmp_path: Path) -> None:
        with patch("orchestra.engine.replay.config.replay_enabled", return_value=True):
            rec = ReplayRecorder("run-004")
            rec.record_tool_call("read_file", {"path": "foo.cs"}, "content")
            lines = rec._path.read_text().strip().splitlines()
            entry = json.loads(lines[0])
            assert entry["type"] == "tool_call"
            assert entry["tool"] == "read_file"
            assert entry["result"] == "content"

    def test_record_tool_call_digest(self, tmp_path: Path) -> None:
        with patch("orchestra.engine.replay.config.replay_enabled", return_value=True):
            rec = ReplayRecorder("run-005")
            large = "x" * 3000
            rec.record_tool_call("read_file", {"path": "foo.cs"}, large)
            lines = rec._path.read_text().strip().splitlines()
            entry = json.loads(lines[0])
            assert "result_digest" in entry
            assert "result" not in entry

    def test_record_synthesis(self, tmp_path: Path) -> None:
        with patch("orchestra.engine.replay.config.replay_enabled", return_value=True):
            rec = ReplayRecorder("run-006")
            rec.record_synthesis("cld-deep", "synth prompt")
            lines = rec._path.read_text().strip().splitlines()
            entry = json.loads(lines[0])
            assert entry["type"] == "synthesis"
            assert entry["alias"] == "cld-deep"

    def test_record_run_finalized(self, tmp_path: Path) -> None:
        with patch("orchestra.engine.replay.config.replay_enabled", return_value=True):
            rec = ReplayRecorder("run-007")
            rec.record_run_finalized("completed", "summary text")
            lines = rec._path.read_text().strip().splitlines()
            entry = json.loads(lines[0])
            assert entry["type"] == "run_finalized"
            assert entry["status"] == "completed"
            assert entry["summary_digest"].startswith("sha256:")

    def test_disabled_when_replay_off(self, tmp_path: Path) -> None:
        with patch("orchestra.engine.replay.config.replay_enabled", return_value=False):
            rec = ReplayRecorder("run-008")
            rec.record_run_started("dual", "task")
            assert not rec._path.exists()

    def test_seq_increments(self, tmp_path: Path) -> None:
        with patch("orchestra.engine.replay.config.replay_enabled", return_value=True):
            rec = ReplayRecorder("run-009")
            rec.record_run_started("dual", "t")
            rec.record_agent_dispatch("a", "p", "m", "pr")
            lines = rec._path.read_text().strip().splitlines()
            assert json.loads(lines[0])["seq"] == 1
            assert json.loads(lines[1])["seq"] == 2


class TestReplayRunner:
    """Replay execution from recorded logs."""

    def test_replay_no_log(self) -> None:
        runner = ReplayRunner("nonexistent")
        with patch("orchestra.engine.runners.run_ask") as mock_ask:
            runner.replay()
            mock_ask.assert_not_called()

    def test_replay_agent_dispatches(self, tmp_path: Path) -> None:
        with patch("orchestra.engine.replay.config.replay_enabled", return_value=True):
            rec = ReplayRecorder("run-010")
            rec.record_run_started("dual", "task")
            rec.record_agent_dispatch("cdx-deep", "prompt1", "m1", "codex")
            rec.record_agent_dispatch("gmn-pro", "prompt2", "m2", "gemini")
            rec.record_run_finalized("completed", None)

        with patch("orchestra.engine.runners.run_ask") as mock_ask:
            runner = ReplayRunner("run-010")
            runner.replay()
            assert mock_ask.call_count == 2
            mock_ask.assert_any_call("cdx-deep", "prompt1", emit_console=False, show_live=False)
            mock_ask.assert_any_call("gmn-pro", "prompt2", emit_console=False, show_live=False)

    def test_replay_deterministic_passes_params(self, tmp_path: Path) -> None:
        with patch("orchestra.engine.replay.config.replay_enabled", return_value=True):
            rec = ReplayRecorder("run-011")
            rec.record_agent_dispatch("cdx-deep", "p", "m", "codex")

        with patch("orchestra.engine.runners.run_ask") as mock_ask:
            runner = ReplayRunner("run-011")
            runner.replay(deterministic=True)
            _, kwargs = mock_ask.call_args
            assert kwargs["temperature"] == 0.0
            expected_seed = int(hashlib.sha256("run-011".encode()).hexdigest(), 16) % 2_147_483_647
            assert kwargs["seed"] == expected_seed

    def test_replay_builds_tool_cache(self, tmp_path: Path) -> None:
        with patch("orchestra.engine.replay.config.replay_enabled", return_value=True):
            rec = ReplayRecorder("run-012")
            rec.record_tool_call("read_file", {"path": "a.cs"}, "content-a")
            rec.record_tool_call("read_file", {"path": "b.cs"}, "content-b")

        runner = ReplayRunner("run-012")
        runner.replay()
        assert len(runner._tool_cache) == 2


class TestGetRecorder:
    """Module-level recorder cache."""

    def test_same_instance_per_run_id(self) -> None:
        with patch("orchestra.engine.replay.config.replay_enabled", return_value=True):
            a = get_recorder("run-x")
            b = get_recorder("run-x")
            assert a is b

    def test_different_instance_per_run_id(self) -> None:
        with patch("orchestra.engine.replay.config.replay_enabled", return_value=True):
            a = get_recorder("run-a")
            b = get_recorder("run-b")
            assert a is not b

    def test_tool_call_redacts_api_keys(self) -> None:
        with patch("orchestra.engine.replay.config.replay_enabled", return_value=True):
            rec = ReplayRecorder("run-redact")
            rec.record_tool_call(
                "call_api",
                {"api_key": "sk-abcdefghijklmnopqrstuvwxyz", "url": "https://example.com"},
                {"status": "ok"},
            )
            with open(rec._path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            entry = json.loads(lines[-1])
            assert "[REDACTED:" in entry["args"]["api_key"]
            assert entry["args"]["url"] == "https://example.com"


def test_a_replay_with_two_answers_records_their_consensus_graph(tmp_path: Path, caplog) -> None:
    # The runner called a recorder method it does not have; the AttributeError was swallowed
    # as "Failed to rebuild consensus graph" and no graph was ever written.
    run_id = "graph01"
    with patch("orchestra.engine.replay.config.replay_enabled", return_value=True), \
            patch("orchestra.engine.replay.config.artifact_root", return_value=tmp_path):
        recorder = get_recorder(run_id)
        recorder.record_agent_completed("cld-deep", "Pooling avoids allocations.", 0)
        recorder.record_agent_completed("kimi-deep", "Pooling avoids allocations.", 0)
        with caplog.at_level("WARNING", logger="orchestra.engine.replay"):
            ReplayRunner(run_id).replay()

        entries = [json.loads(line) for line in (tmp_path / run_id / "replay.jsonl").read_text().splitlines()]

    assert "Failed to rebuild consensus graph" not in caplog.text
    assert [e["type"] for e in entries][-1] == "consensus_graph"

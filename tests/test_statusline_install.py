"""Tests for statusline install helpers and CLI commands.

Run from a scratch directory:
    cd /tmp && /abs/.venv/bin/pytest /abs/tests/test_statusline_install.py -q --no-cov -p no:cacheprovider
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from orchestra.cli import app
from orchestra.utils.statusline_install import (
    ConflictError,
    MergeResult,
    build_statusline_block,
    merge_settings,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RUNNER = CliRunner()

_NOW_STR = "2026-01-15T12:00:00+00:00"


def _live_status_payload(runs: dict | None = None) -> dict:
    return {
        "version": 1,
        "updated_at": _NOW_STR,
        "runs": runs or {},
    }


def _write_live_status(orchestra_dir: Path, runs: dict | None = None) -> None:
    orchestra_dir.mkdir(parents=True, exist_ok=True)
    (orchestra_dir / "live_status.json").write_text(
        json.dumps(_live_status_payload(runs)), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Unit tests — build_statusline_block
# ---------------------------------------------------------------------------


class TestBuildStatuslineBlock:
    def test_returns_correct_keys(self):
        block = build_statusline_block("/usr/bin/python3", 2)
        assert set(block.keys()) == {"type", "command", "refreshInterval"}

    def test_command_contains_interpreter(self):
        block = build_statusline_block("/usr/bin/python3", 2)
        assert block["command"].startswith("/usr/bin/python3")
        assert "-m orchestra.statusline" in block["command"]

    def test_type_is_command(self):
        block = build_statusline_block("/usr/bin/python3", 2)
        assert block["type"] == "command"

    def test_refresh_interval_set(self):
        block = build_statusline_block("/usr/bin/python3", 5)
        assert block["refreshInterval"] == 5

    def test_interval_zero_raises(self):
        with pytest.raises(ValueError, match="interval must be >= 1"):
            build_statusline_block("/usr/bin/python3", 0)

    def test_interval_negative_raises(self):
        with pytest.raises(ValueError):
            build_statusline_block("/usr/bin/python3", -1)


# ---------------------------------------------------------------------------
# Unit tests — merge_settings
# ---------------------------------------------------------------------------


class TestMergeSettingsNoExistingFile:
    """Test 1: merge into missing file → created with exactly the block."""

    def test_none_existing_returns_block_only(self):
        block = build_statusline_block("/usr/bin/python3", 2)
        result = merge_settings(None, block)
        assert result.merged == {"statusLine": block}

    def test_none_existing_no_overwritten_flag(self):
        block = build_statusline_block("/usr/bin/python3", 2)
        result = merge_settings(None, block)
        assert result.was_overwritten is False

    def test_none_existing_no_conflict(self):
        block = build_statusline_block("/usr/bin/python3", 2)
        result = merge_settings(None, block)
        assert result.conflict is None


class TestMergeSettingsExistingWithoutStatusLine:
    """Test 2: merge into existing settings with other keys → other keys preserved."""

    def test_other_keys_preserved(self):
        existing = {"mcpServers": {"my-server": {}}, "theme": "dark"}
        block = build_statusline_block("/usr/bin/python3", 2)
        result = merge_settings(existing, block)
        assert result.merged["mcpServers"] == {"my-server": {}}
        assert result.merged["theme"] == "dark"

    def test_statusline_added(self):
        existing = {"theme": "dark"}
        block = build_statusline_block("/usr/bin/python3", 2)
        result = merge_settings(existing, block)
        assert result.merged["statusLine"] == block

    def test_input_dict_not_mutated(self):
        """Immutability: original dict must be unchanged after merge."""
        existing = {"theme": "dark"}
        original_keys = set(existing.keys())
        block = build_statusline_block("/usr/bin/python3", 2)
        merge_settings(existing, block)
        assert set(existing.keys()) == original_keys
        assert "statusLine" not in existing

    def test_returns_new_dict_not_same_object(self):
        existing = {"theme": "dark"}
        block = build_statusline_block("/usr/bin/python3", 2)
        result = merge_settings(existing, block)
        assert result.merged is not existing


class TestMergeSettingsConflictWithoutForce:
    """Test 3: existing statusLine without --force → refused, exit 1."""

    def test_raises_conflict_error(self):
        old_block = {"type": "command", "command": "old-cmd", "refreshInterval": 3}
        existing = {"statusLine": old_block}
        block = build_statusline_block("/usr/bin/python3", 2)
        with pytest.raises(ConflictError) as exc_info:
            merge_settings(existing, block, force=False)
        assert exc_info.value.existing_value == old_block

    def test_conflict_error_has_existing_value(self):
        old_block = {"type": "command", "command": "old", "refreshInterval": 1}
        existing = {"statusLine": old_block}
        block = build_statusline_block("/usr/bin/python3", 2)
        try:
            merge_settings(existing, block, force=False)
            pytest.fail("Expected ConflictError")
        except ConflictError as e:
            assert e.existing_value is old_block


class TestMergeSettingsConflictWithForce:
    """Test 4: existing statusLine with --force → new value written, conflict stored."""

    def test_force_overwrites_statusline(self):
        old_block = {"type": "command", "command": "old", "refreshInterval": 1}
        existing = {"statusLine": old_block, "theme": "light"}
        block = build_statusline_block("/usr/bin/python3", 5)
        result = merge_settings(existing, block, force=True)
        assert result.merged["statusLine"] == block

    def test_force_preserves_other_keys(self):
        old_block = {"type": "command", "command": "old", "refreshInterval": 1}
        existing = {"statusLine": old_block, "theme": "light"}
        block = build_statusline_block("/usr/bin/python3", 5)
        result = merge_settings(existing, block, force=True)
        assert result.merged["theme"] == "light"

    def test_force_sets_was_overwritten(self):
        old_block = {"type": "command", "command": "old", "refreshInterval": 1}
        existing = {"statusLine": old_block}
        block = build_statusline_block("/usr/bin/python3", 2)
        result = merge_settings(existing, block, force=True)
        assert result.was_overwritten is True

    def test_force_stores_old_value_in_conflict(self):
        old_block = {"type": "command", "command": "old", "refreshInterval": 1}
        existing = {"statusLine": old_block}
        block = build_statusline_block("/usr/bin/python3", 2)
        result = merge_settings(existing, block, force=True)
        assert result.conflict == old_block

    def test_force_does_not_mutate_existing(self):
        old_block = {"type": "command", "command": "old", "refreshInterval": 1}
        existing = {"statusLine": old_block}
        block = build_statusline_block("/usr/bin/python3", 2)
        merge_settings(existing, block, force=True)
        assert existing["statusLine"] is old_block  # unchanged


# ---------------------------------------------------------------------------
# CLI integration tests — statusline install
# ---------------------------------------------------------------------------


class TestStatuslineInstallCLI:
    """CLI-level install tests using typer.testing.CliRunner."""

    def test_creates_new_file_when_missing(self, tmp_path):
        """install into a non-existent file → creates it with statusLine block."""
        settings_path = tmp_path / "settings.json"
        result = RUNNER.invoke(
            app,
            ["statusline", "install", "--path", str(settings_path)],
        )
        assert result.exit_code == 0, result.output
        assert settings_path.exists()
        written = json.loads(settings_path.read_text())
        assert "statusLine" in written
        assert written["statusLine"]["type"] == "command"
        assert "-m orchestra.statusline" in written["statusLine"]["command"]
        assert written["statusLine"]["refreshInterval"] == 2  # default

    def test_creates_parent_dir_when_missing(self, tmp_path):
        """install creates parent directories if they don't exist."""
        settings_path = tmp_path / ".claude" / "settings.json"
        result = RUNNER.invoke(
            app,
            ["statusline", "install", "--path", str(settings_path)],
        )
        assert result.exit_code == 0, result.output
        assert settings_path.exists()

    def test_preserves_other_keys_in_existing_file(self, tmp_path):
        """install adds statusLine without disturbing existing keys."""
        settings_path = tmp_path / "settings.json"
        initial = {"theme": "dark", "mcpServers": {}}
        settings_path.write_text(json.dumps(initial), encoding="utf-8")

        result = RUNNER.invoke(
            app,
            ["statusline", "install", "--path", str(settings_path)],
        )
        assert result.exit_code == 0, result.output
        written = json.loads(settings_path.read_text())
        assert written["theme"] == "dark"
        assert written["mcpServers"] == {}
        assert "statusLine" in written

    def test_conflict_without_force_exits_1(self, tmp_path):
        """install refuses when statusLine already exists, exit 1."""
        settings_path = tmp_path / "settings.json"
        old = {"statusLine": {"type": "command", "command": "old", "refreshInterval": 3}}
        settings_path.write_text(json.dumps(old), encoding="utf-8")

        result = RUNNER.invoke(
            app,
            ["statusline", "install", "--path", str(settings_path)],
        )
        assert result.exit_code == 1
        # File must be unchanged
        unchanged = json.loads(settings_path.read_text())
        assert unchanged["statusLine"]["command"] == "old"
        # Output should mention --force
        assert "--force" in result.output

    def test_conflict_with_force_creates_backup_and_overwrites(self, tmp_path):
        """Test 4: --force creates backup file and writes new statusLine."""
        settings_path = tmp_path / "settings.json"
        old = {"statusLine": {"type": "command", "command": "old", "refreshInterval": 3}}
        settings_path.write_text(json.dumps(old), encoding="utf-8")

        result = RUNNER.invoke(
            app,
            ["statusline", "install", "--path", str(settings_path), "--force"],
        )
        assert result.exit_code == 0, result.output

        # Backup must exist
        backup_files = list(tmp_path.glob("settings.json.bak-*"))
        assert len(backup_files) == 1, f"Expected 1 backup, found: {backup_files}"
        # Original must be overwritten
        written = json.loads(settings_path.read_text())
        assert "-m orchestra.statusline" in written["statusLine"]["command"]

    def test_invalid_json_exits_1_file_unchanged(self, tmp_path):
        """Test 5: invalid JSON file → exit 1, file byte-identical, no backup."""
        settings_path = tmp_path / "settings.json"
        bad_content = "{ this is not json }"
        settings_path.write_text(bad_content, encoding="utf-8")

        result = RUNNER.invoke(
            app,
            ["statusline", "install", "--path", str(settings_path)],
        )
        assert result.exit_code == 1
        # File unchanged
        assert settings_path.read_text(encoding="utf-8") == bad_content
        # No backup created
        backups = list(tmp_path.glob("settings.json.bak-*"))
        assert backups == [], f"Unexpected backup files: {backups}"

    def test_interval_option_is_written(self, tmp_path):
        """Test 7a: --interval 5 → refreshInterval 5 in written file."""
        settings_path = tmp_path / "settings.json"
        result = RUNNER.invoke(
            app,
            ["statusline", "install", "--path", str(settings_path), "--interval", "5"],
        )
        assert result.exit_code == 0, result.output
        written = json.loads(settings_path.read_text())
        assert written["statusLine"]["refreshInterval"] == 5

    def test_interval_zero_is_validation_error(self, tmp_path):
        """Test 7b: --interval 0 → validation error, exit non-zero."""
        settings_path = tmp_path / "settings.json"
        result = RUNNER.invoke(
            app,
            ["statusline", "install", "--path", str(settings_path), "--interval", "0"],
        )
        assert result.exit_code != 0

    def test_user_flag_targets_home_claude_settings(self, tmp_path, monkeypatch):
        """--user flag should target ~/.claude/settings.json (mocked home)."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        # Monkeypatch Path.home() via the module
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

        result = RUNNER.invoke(app, ["statusline", "install", "--user"])
        assert result.exit_code == 0, result.output
        expected = fake_home / ".claude" / "settings.json"
        assert expected.exists()
        written = json.loads(expected.read_text())
        assert "statusLine" in written

    def test_trailing_newline_in_written_file(self, tmp_path):
        """Written JSON file ends with a trailing newline."""
        settings_path = tmp_path / "settings.json"
        RUNNER.invoke(
            app,
            ["statusline", "install", "--path", str(settings_path)],
        )
        raw = settings_path.read_bytes()
        assert raw.endswith(b"\n"), "File should end with a trailing newline"

    def test_output_contains_command_hint(self, tmp_path):
        """After install, output mentions the configured command."""
        settings_path = tmp_path / "settings.json"
        result = RUNNER.invoke(
            app,
            ["statusline", "install", "--path", str(settings_path)],
        )
        assert result.exit_code == 0
        assert "orchestra.statusline" in result.output


# ---------------------------------------------------------------------------
# CLI integration tests — statusline render
# ---------------------------------------------------------------------------


class TestStatuslineRenderCLI:
    """Test 6: render command invokes the renderer in-process."""

    def _make_running_status(self) -> dict:
        return {
            "run-abc123": {
                "status": "running",
                "mode": "dual",
                "started_at": _NOW_STR,
                "heartbeat": _NOW_STR,
                "agents": {
                    "cdx": {
                        "status": "started",
                        "model": "claude-sonnet",
                        "started_at": _NOW_STR,
                        "last_line": "Analyzing…",
                    }
                },
            }
        }

    def test_render_exits_0(self, tmp_path):
        """render command exits 0."""
        _write_live_status(tmp_path / ".orchestra", self._make_running_status())
        result = RUNNER.invoke(
            app,
            ["statusline", "render", "--dir", str(tmp_path), "--no-color"],
        )
        assert result.exit_code == 0, result.output

    def test_render_shows_agent_line(self, tmp_path):
        """render output contains the agent alias."""
        _write_live_status(tmp_path / ".orchestra", self._make_running_status())
        result = RUNNER.invoke(
            app,
            ["statusline", "render", "--dir", str(tmp_path), "--no-color"],
        )
        assert result.exit_code == 0
        assert "cdx" in result.output

    def test_render_idle_without_status_file(self, tmp_path):
        """render returns idle line when no status file found."""
        result = RUNNER.invoke(
            app,
            ["statusline", "render", "--dir", str(tmp_path), "--no-color"],
        )
        assert result.exit_code == 0
        assert "orchestra idle" in result.output

    def test_render_default_dir_is_cwd(self, tmp_path, monkeypatch):
        """render without --dir uses CWD."""
        _write_live_status(tmp_path / ".orchestra")
        monkeypatch.chdir(tmp_path)
        result = RUNNER.invoke(app, ["statusline", "render", "--no-color"])
        assert result.exit_code == 0
        assert "orchestra" in result.output

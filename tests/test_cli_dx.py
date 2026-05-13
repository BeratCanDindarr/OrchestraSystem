"""Tests for CLI Developer Experience: --dry-run, batch-run, shell completion."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from typer.testing import CliRunner

from orchestra.cli import app

runner = CliRunner()


class TestDryRunAsk(unittest.TestCase):
    def test_ask_dry_run_exits_zero(self):
        result = runner.invoke(app, ["ask", "cld-fast", "hello world", "--dry-run"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_ask_dry_run_contains_alias(self):
        result = runner.invoke(app, ["ask", "cdx-deep", "explain this code", "--dry-run"])
        self.assertIn("cdx-deep", result.output)

    def test_ask_dry_run_contains_label(self):
        result = runner.invoke(app, ["ask", "gmn-pro", "review code", "--dry-run"])
        self.assertIn("dry-run", result.output)

    def test_ask_dry_run_contains_cost(self):
        result = runner.invoke(app, ["ask", "cld-fast", "what is 2+2", "--dry-run"])
        self.assertIn("$", result.output)

    def test_ask_help_has_dry_run_flag(self):
        result = runner.invoke(app, ["ask", "--help"])
        self.assertIn("--dry-run", result.output)


class TestDryRunRun(unittest.TestCase):
    def test_run_auto_dry_run_exits_zero(self):
        result = runner.invoke(app, ["run", "auto", "explain this function", "--dry-run"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_run_dual_dry_run_exits_zero(self):
        result = runner.invoke(app, ["run", "dual", "review this PR", "--dry-run"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_run_critical_dry_run_exits_zero(self):
        result = runner.invoke(app, ["run", "critical", "deploy to production", "--dry-run"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_run_planned_dry_run_exits_zero(self):
        result = runner.invoke(app, ["run", "planned", "build a new feature end to end", "--dry-run"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_run_auto_dry_run_shows_router_reason(self):
        result = runner.invoke(app, ["run", "auto", "explain this function", "--dry-run"])
        self.assertIn("router", result.output)

    def test_run_dry_run_shows_cost(self):
        result = runner.invoke(app, ["run", "dual", "review this code", "--dry-run"])
        self.assertIn("$", result.output)

    def test_run_help_has_dry_run_flag(self):
        result = runner.invoke(app, ["run", "--help"])
        self.assertIn("--dry-run", result.output)


class TestBatchRun(unittest.TestCase):
    def _write_jsonl(self, lines: list[dict]) -> Path:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        for line in lines:
            f.write(json.dumps(line) + "\n")
        f.close()
        return Path(f.name)

    def _write_raw(self, content: str) -> Path:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        f.write(content)
        f.close()
        return Path(f.name)

    def test_dry_run_valid_file_exits_zero(self):
        path = self._write_jsonl([
            {"mode": "ask", "alias": "cld-fast", "task": "hello"},
            {"mode": "auto", "task": "explain this"},
        ])
        try:
            result = runner.invoke(app, ["batch-run", str(path), "--dry-run"])
            self.assertEqual(result.exit_code, 0, result.output)
        finally:
            path.unlink(missing_ok=True)

    def test_dry_run_shows_task_count(self):
        path = self._write_jsonl([
            {"mode": "ask", "alias": "cld-fast", "task": "task one"},
            {"mode": "dual", "task": "task two"},
            {"mode": "auto", "task": "task three"},
        ])
        try:
            result = runner.invoke(app, ["batch-run", str(path), "--dry-run"])
            self.assertIn("3", result.output)
        finally:
            path.unlink(missing_ok=True)

    def test_dry_run_missing_task_key_exits_nonzero(self):
        path = self._write_jsonl([{"mode": "ask", "alias": "cld-fast"}])
        try:
            result = runner.invoke(app, ["batch-run", str(path), "--dry-run"])
            self.assertNotEqual(result.exit_code, 0)
        finally:
            path.unlink(missing_ok=True)

    def test_dry_run_missing_mode_key_exits_nonzero(self):
        path = self._write_jsonl([{"task": "hello"}])
        try:
            result = runner.invoke(app, ["batch-run", str(path), "--dry-run"])
            self.assertNotEqual(result.exit_code, 0)
        finally:
            path.unlink(missing_ok=True)

    def test_dry_run_empty_file_exits_nonzero(self):
        path = self._write_jsonl([])
        try:
            result = runner.invoke(app, ["batch-run", str(path), "--dry-run"])
            self.assertNotEqual(result.exit_code, 0)
        finally:
            path.unlink(missing_ok=True)

    def test_dry_run_invalid_json_line_exits_nonzero(self):
        path = self._write_raw('{"mode": "ask", "task": "valid"}\nnot valid json\n')
        try:
            result = runner.invoke(app, ["batch-run", str(path), "--dry-run"])
            self.assertNotEqual(result.exit_code, 0)
        finally:
            path.unlink(missing_ok=True)

    def test_dry_run_skips_blank_lines(self):
        path = self._write_raw('\n{"mode": "ask", "task": "valid", "alias": "cld-fast"}\n\n')
        try:
            result = runner.invoke(app, ["batch-run", str(path), "--dry-run"])
            self.assertEqual(result.exit_code, 0, result.output)
        finally:
            path.unlink(missing_ok=True)

    def test_file_not_found_exits_nonzero(self):
        result = runner.invoke(app, ["batch-run", "/tmp/nonexistent_orchestra_test_xyz.jsonl", "--dry-run"])
        self.assertNotEqual(result.exit_code, 0)

    def test_help_shows_concurrency_flag(self):
        result = runner.invoke(app, ["batch-run", "--help"])
        self.assertIn("--concurrency", result.output)

    def test_help_shows_json_flag(self):
        result = runner.invoke(app, ["batch-run", "--help"])
        self.assertIn("--json", result.output)


class TestShellCompletion(unittest.TestCase):
    def test_help_exits_zero(self):
        result = runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)

    def test_help_contains_install_completion(self):
        result = runner.invoke(app, ["--help"])
        self.assertIn("completion", result.output.lower())


if __name__ == "__main__":
    unittest.main()

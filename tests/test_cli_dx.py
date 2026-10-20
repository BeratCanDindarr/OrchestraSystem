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
    def setUp(self):
        # "explain this function" is below the heuristic's confidence threshold, so the router
        # asks its model. The test is about the dry run, not about that model.
        from unittest.mock import patch

        router = patch("orchestra.router.classifier._model_route", return_value=None)
        router.start()
        self.addCleanup(router.stop)

    def test_run_auto_dry_run_exits_zero(self):
        result = runner.invoke(app, ["run", "auto", "explain this function", "--dry-run"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_run_dual_dry_run_exits_zero(self):
        result = runner.invoke(app, ["run", "dual", "review this PR", "--dry-run"])
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


class TestShellCompletion(unittest.TestCase):
    def test_help_exits_zero(self):
        result = runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)

    def test_help_contains_install_completion(self):
        result = runner.invoke(app, ["--help"])
        self.assertIn("completion", result.output.lower())


class TestConsolePiiScrubbing(unittest.TestCase):
    def test_ask_raw_scrubs_pii(self):
        from unittest.mock import patch
        from orchestra.models import OrchestraRun, AgentRun, AgentStatus, RunStatus

        run = OrchestraRun(mode="ask", task="test", run_id="r1")
        run.status = RunStatus.COMPLETED
        agent = AgentRun(alias="cld-fast", provider="test", model="test")
        agent.status = AgentStatus.COMPLETED
        agent.stdout_log = "Contact me at alice@example.com"
        run.agents.append(agent)

        with patch("orchestra.engine.runners.run_ask", return_value=run):
            result = runner.invoke(app, ["ask", "cld-fast", "hello", "--raw"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("alice@example.com", result.output)
        self.assertIn("[REDACTED:email]", result.output)

    def test_logs_scrubs_pii(self):
        from unittest.mock import patch
        import tempfile
        from orchestra.engine.artifacts import run_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("orchestra.engine.artifacts.config.artifact_root", return_value=Path(tmpdir)):
                run_id = "test-run-123"
                d = run_dir(run_id)
                agents_dir = d / "agents"
                agents_dir.mkdir(parents=True, exist_ok=True)
                log_file = agents_dir / "cld-fast.stdout.log"
                log_file.write_text("Key: sk_test_1234567890abcdefghij", encoding="utf-8")

                result = runner.invoke(app, ["logs", run_id])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("sk_test_1234567890abcdefghij", result.output)
        self.assertIn("[REDACTED:api_key_sk]", result.output)


if __name__ == "__main__":
    unittest.main()


class TestUnknownAlias(unittest.TestCase):
    def test_ask_with_a_mistyped_alias_fails_and_suggests_the_alias(self):
        result = runner.invoke(app, ["ask", "cdx-deeep", "hello"])
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("cdx-deep", result.output)
        self.assertNotIn("Traceback", result.output)


class TestExitCodes(unittest.TestCase):
    """A script that calls orchestra must be able to tell that it did not work."""

    @staticmethod
    def _run(status: str):
        from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus

        run = OrchestraRun(mode="ask", task="t", status=RunStatus(status))
        run.agents = [AgentRun(alias="cdx-deep", provider="codex", model="m", status=AgentStatus.COMPLETED)]
        return run

    def _invoke_with_run(self, argv, status, target):
        from unittest.mock import patch

        with patch(f"orchestra.engine.runners.{target}", return_value=self._run(status)):
            return runner.invoke(app, argv)

    def test_ask_exits_nonzero_when_the_run_failed(self):
        self.assertEqual(self._invoke_with_run(["ask", "cdx-deep", "x"], "failed", "run_ask").exit_code, 1)

    def test_ask_exits_zero_when_the_run_completed(self):
        self.assertEqual(self._invoke_with_run(["ask", "cdx-deep", "x"], "completed", "run_ask").exit_code, 0)

    def test_run_dual_exits_nonzero_when_the_run_failed(self):
        self.assertEqual(self._invoke_with_run(["run", "dual", "x"], "failed", "run_dual").exit_code, 1)

    def test_run_planned_exits_nonzero_when_the_run_was_cancelled(self):
        self.assertEqual(self._invoke_with_run(["run", "planned", "x"], "cancelled", "run_planned").exit_code, 1)

    def test_run_ask_points_at_the_ask_command(self):
        result = runner.invoke(app, ["run", "ask", "x"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("orchestra ask", result.output)

    def test_commands_about_a_run_that_does_not_exist_fail_and_create_nothing(self):
        from orchestra import config

        for command in ("logs", "cancel", "trace", "checkpoints"):
            with self.subTest(command=command):
                result = runner.invoke(app, [command, "no-such-run-id"])
                self.assertEqual(result.exit_code, 1, result.output)
                self.assertIn("not found", result.output)
                self.assertFalse((config.artifact_root() / "no-such-run-id").exists())

    def test_cancel_leaves_a_finished_run_as_it_ended(self):
        from orchestra.engine import artifacts

        run = self._run("completed")
        artifacts.write_manifest(run)

        result = runner.invoke(app, ["cancel", run.run_id])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(artifacts.load_manifest(run.run_id)["status"], "completed")
        self.assertIn("already", result.output)

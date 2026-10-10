"""Regression test: CLI command registry must have no duplicate groups and config must expose all four commands.

This test FAILS before the C-2 fix because:
- config_app is redefined (shadowed) with only get/set — validate/init are unreachable.
- playbook, cue and sessions are each registered twice.

After the fix, each group appears exactly once, and config has get/set/validate/init.
"""
from __future__ import annotations

import unittest

import typer.main

from orchestra.cli import app


def _click_app():
    """Build the Click command object from the Typer app (read-only introspection)."""
    return typer.main.get_command(app)


class TestCommandRegistryNoDuplicates(unittest.TestCase):
    """Each sub-app name must appear exactly once."""

    def _group_names(self):
        click_app = _click_app()
        return list(click_app.commands.keys())

    def test_config_appears_exactly_once(self):
        names = self._group_names()
        self.assertEqual(
            names.count("config"),
            1,
            f"'config' should appear exactly once in CLI groups, got: {names}",
        )

    def test_playbook_appears_exactly_once(self):
        names = self._group_names()
        self.assertEqual(names.count("playbook"), 1, f"'playbook' duplicated in: {names}")

    def test_cue_appears_exactly_once(self):
        names = self._group_names()
        self.assertEqual(names.count("cue"), 1, f"'cue' duplicated in: {names}")

    def test_sessions_appears_exactly_once(self):
        names = self._group_names()
        self.assertEqual(names.count("sessions"), 1, f"'sessions' duplicated in: {names}")

    def test_report_present(self):
        names = self._group_names()
        self.assertIn("report", names, f"'report' missing from: {names}")

    def test_memory_present(self):
        names = self._group_names()
        self.assertIn("memory", names, f"'memory' missing from: {names}")


class TestConfigSubCommandsComplete(unittest.TestCase):
    """config sub-app must expose all four commands: get, set, validate, init."""

    def _config_commands(self):
        click_app = _click_app()
        config_group = click_app.commands.get("config")
        self.assertIsNotNone(config_group, "config group not found in CLI")
        return list(config_group.commands.keys())

    def test_config_has_get(self):
        self.assertIn("get", self._config_commands())

    def test_config_has_set(self):
        self.assertIn("set", self._config_commands())

    def test_config_has_validate(self):
        cmds = self._config_commands()
        self.assertIn(
            "validate",
            cmds,
            f"config validate missing — shadowing bug present. config commands: {cmds}",
        )

    def test_config_has_init(self):
        cmds = self._config_commands()
        self.assertIn(
            "init",
            cmds,
            f"config init missing — shadowing bug present. config commands: {cmds}",
        )

    def test_config_has_exactly_four_commands(self):
        cmds = self._config_commands()
        self.assertEqual(
            len(cmds),
            4,
            f"Expected exactly 4 config commands (get/set/validate/init), got: {cmds}",
        )


if __name__ == "__main__":
    unittest.main()

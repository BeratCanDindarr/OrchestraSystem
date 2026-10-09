"""Unit tests for the Kimi CLI provider — deterministic, the CLI is never started."""
from __future__ import annotations

import subprocess
from unittest.mock import patch

from orchestra.providers.fallback import ALIAS_TABLE
from orchestra.providers.kimi import KimiProvider, parse_stream_json
from orchestra.providers.ollama import OllamaProvider
from orchestra.providers.registry import AliasRegistry

VERSION_LINE = '{"role":"meta","type":"system.version","version":"2.1.1"}'
RESUME_LINE = '{"role":"meta","type":"session.resume_hint","session_id":"session_x","content":"To resume: kimi -r session_x"}'


def _completed(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["kimi"], returncode=returncode, stdout=stdout, stderr=stderr)


class TestParseStreamJson:
    def test_returns_the_assistant_message_without_meta_lines(self):
        stdout = "\n".join([VERSION_LINE, '{"role":"assistant","content":"ok"}', RESUME_LINE])
        assert parse_stream_json(stdout) == "ok"

    def test_the_last_assistant_message_is_the_answer(self):
        stdout = "\n".join([
            '{"role":"assistant","content":"Let me read the file."}',
            '{"role":"tool","content":"file contents"}',
            '{"role":"assistant","content":"The bug is on line 3."}',
        ])
        assert parse_stream_json(stdout) == "The bug is on line 3."

    def test_reads_text_blocks_and_skips_other_blocks(self):
        stdout = '{"role":"assistant","content":[{"type":"thinking","text":"hm"},{"type":"text","text":"done"}]}'
        assert parse_stream_json(stdout) == "done"

    def test_ignores_lines_that_are_not_json(self):
        stdout = "Warning: something\n{not json}\n" + '{"role":"assistant","content":"ok"}'
        assert parse_stream_json(stdout) == "ok"

    def test_nothing_readable_gives_an_empty_answer(self):
        assert parse_stream_json(VERSION_LINE + "\n" + RESUME_LINE) == ""
        assert parse_stream_json("") == ""


class TestKimiProvider:
    def test_command_runs_the_prompt_mode_with_json_output(self):
        provider = KimiProvider()
        with patch.object(provider, "_binary", return_value="/bin/kimi"):
            command = provider.build_command("hello", "k3")
        assert command == ["/bin/kimi", "-p", "hello", "-m", "kimi-code/k3", "--output-format", "stream-json"]

    def test_tier_shortcuts_and_unknown_models(self):
        provider = KimiProvider()
        assert provider.build_command("x", "medium")[4] == "kimi-code/kimi-for-coding"
        assert provider.build_command("x", "kimi-code/k3-256k")[4] == "kimi-code/k3-256k"

    def test_model_label(self):
        provider = KimiProvider()
        assert provider.model_label("k3") == "kimi/k3"
        assert provider.model_label("code") == "kimi/kimi-for-coding"

    def test_run_returns_only_the_answer(self):
        stdout = "\n".join([VERSION_LINE, '{"role":"assistant","content":"ok"}', RESUME_LINE])
        with patch("orchestra.providers.base.subprocess.run", return_value=_completed(stdout)):
            assert KimiProvider().run("hi", "k3") == ("ok", 0)

    def test_run_passes_a_failure_through(self):
        with patch("orchestra.providers.base.subprocess.run",
                   return_value=_completed("", returncode=2, stderr="not logged in")):
            assert KimiProvider().run("hi", "k3") == ("not logged in", 2)

    def test_run_without_an_answer_is_an_error(self):
        with patch("orchestra.providers.base.subprocess.run", return_value=_completed(VERSION_LINE)):
            output, returncode = KimiProvider().run("hi", "k3")
        assert returncode == 1
        assert output.startswith("[ERROR]")

    def test_availability_follows_the_binary(self):
        provider = KimiProvider()
        with patch("orchestra.providers.kimi.shutil.which", return_value="/bin/kimi"):
            assert provider.is_available() is True
        with patch("orchestra.providers.kimi.shutil.which", return_value=None), \
                patch("orchestra.providers.kimi.os.access", return_value=False):
            assert provider.is_available() is False


class TestKimiAliases:
    def test_kimi_aliases_use_the_kimi_cli(self):
        assert isinstance(ALIAS_TABLE["kimi-deep"][0], KimiProvider)
        assert isinstance(ALIAS_TABLE["kimi-code"][0], KimiProvider)
        assert AliasRegistry.get("kimi-deep").provider == "kimi"
        assert AliasRegistry.get("kimi-code").provider == "kimi"

    def test_no_ollama_alias_reaches_a_kimi_model(self):
        for alias, (provider, model) in ALIAS_TABLE.items():
            if isinstance(provider, OllamaProvider):
                assert "kimi" not in alias.lower(), alias
                assert "kimi" not in str(model).lower(), alias
        for alias in AliasRegistry.by_provider("ollama"):
            assert "kimi" not in alias.name.lower() and "kimi" not in alias.model.lower(), alias.name

"""Unit tests for Antigravity provider."""
from __future__ import annotations

import json
from unittest.mock import patch

from orchestra.providers.antigravity import AntigravityProvider, parse_result
from orchestra.providers.fallback import ALIAS_TABLE
from orchestra.providers.registry import AliasRegistry


class TestParseResult:
    def test_extracts_response_and_usage(self):
        payload = json.dumps({
            "status": "SUCCESS",
            "response": "Hello world",
            "usage": {
                "input_tokens": 120,
                "output_tokens": 45,
                "thinking_tokens": 10
            }
        })
        text, usage, is_err = parse_result(payload)
        assert text == "Hello world"
        assert not is_err
        assert usage == {
            "prompt_tokens": 120,
            "completion_tokens": 45,
            "thinking_tokens": 10,
        }

    def test_handles_leading_logs_before_json(self):
        payload = "\n".join([
            "Info: checking updates...",
            "Warning: some warning",
            json.dumps({"status": "SUCCESS", "response": "Clean answer", "usage": {"input_tokens": 10, "output_tokens": 5}})
        ])
        text, usage, is_err = parse_result(payload)
        assert text == "Clean answer"
        assert not is_err
        assert usage["prompt_tokens"] == 10

    def test_handles_error_status(self):
        payload = json.dumps({
            "status": "ERROR",
            "error": "Model not found",
            "response": ""
        })
        text, usage, is_err = parse_result(payload)
        assert text == "Model not found"
        assert is_err
        assert usage is None

    def test_empty_answer_after_denied_tool_is_an_error(self):
        payload = json.dumps({"status": "SUCCESS", "response": "", "denied_actions": [{"action": "write_file"}]})
        text, usage, is_err = parse_result(payload)
        assert is_err
        assert "write_file" in text

    def test_handles_raw_non_json_output(self):
        text, usage, is_err = parse_result("Plain text fallback")
        assert text == "Plain text fallback"
        assert not is_err
        assert usage is None


    def test_a_stream_is_read_from_its_result_event(self):
        stream = "\n".join(json.dumps(x) for x in (
            {"event": "init", "conversation_id": "c1", "init": {"model": "gemini-4-argon-high"}},
            {"event": "step_update", "step_update": {"state": "DONE", "text_delta": "pon"}},
            {"event": "result", "result": {"status": "SUCCESS", "response": "pong\n",
                                          "usage": {"input_tokens": 8, "output_tokens": 2, "thinking_tokens": 1}}}))
        assert parse_result(stream) == ("pong", {"prompt_tokens": 8, "completion_tokens": 2, "thinking_tokens": 1}, False)

    def test_a_stream_that_ends_in_an_error_is_an_error(self):
        stream = "\n".join(json.dumps(x) for x in (
            {"event": "init", "conversation_id": "c1"},
            {"event": "result", "result": {"status": "ERROR", "error": "quota exhausted"}}))
        text, _, is_err = parse_result(stream)
        assert is_err and "quota exhausted" in text


class TestAntigravityProvider:
    def test_build_command(self):
        provider = AntigravityProvider()
        with patch.object(provider, "_binary", return_value="/bin/agy"), \
             patch("orchestra.providers.antigravity.config.provider_config", return_value={}):
            cmd = provider.build_command("hello", "argon")
        assert cmd[:4] == ["/bin/agy", "-p", "hello", "--model"]
        assert cmd[4] == "gemini-4-argon-high"
        # a stream: Gemini's thinking and tool steps are output, so a working agent is never silent
        assert "--output-format" in cmd and cmd[cmd.index("--output-format") + 1] == "stream-json"
        assert "--dangerously-skip-permissions" not in cmd   # bypass only when the profile asks

    def test_build_command_skips_permissions_when_configured(self):
        provider = AntigravityProvider()
        with patch.object(provider, "_binary", return_value="/bin/agy"), \
             patch("orchestra.providers.antigravity.config.provider_config", return_value={"skip_permissions": True}):
            cmd = provider.build_command("hello", "argon")
        assert "--dangerously-skip-permissions" in cmd

    def test_model_label(self):
        provider = AntigravityProvider()
        assert provider.model_label("argon") == "antigravity/gemini-4-argon-high"
        assert provider.model_label("flash") == "antigravity/gemini-3.8-flash-high"

    def test_run_success(self):
        provider = AntigravityProvider()
        payload = json.dumps({"status": "SUCCESS", "response": "ok", "usage": {"input_tokens": 10, "output_tokens": 5}})
        with patch("orchestra.providers.base.run_command", return_value=(payload, "", 0)):
            text, code = provider.run("hi", "argon")
        assert text == "ok"
        assert code == 0
        assert provider.last_usage()["prompt_tokens"] == 10

    def test_run_failure(self):
        provider = AntigravityProvider()
        with patch("orchestra.providers.base.run_command", return_value=("", "command failed", 1)):
            text, code = provider.run("hi", "argon")
        assert code == 1
        assert "command failed" in text

    def test_availability(self):
        provider = AntigravityProvider()
        with patch("orchestra.providers.antigravity.shutil.which", return_value="/bin/agy"):
            assert provider.is_available() is True
        with patch("orchestra.providers.antigravity.shutil.which", return_value=None),              patch("orchestra.providers.antigravity.os.access", return_value=False):
            assert provider.is_available() is False


class TestAntigravityAliases:
    def test_aliases_registered(self):
        assert "agy-argon" in ALIAS_TABLE
        assert "agy-deep" in ALIAS_TABLE
        assert "agy-flash" in ALIAS_TABLE
        assert "agy-fast" in ALIAS_TABLE

        assert isinstance(ALIAS_TABLE["agy-argon"][0], AntigravityProvider)
        assert AliasRegistry.get("agy-argon").provider == "antigravity"
        assert AliasRegistry.get("agy-argon").tier == "heavy"
        assert AliasRegistry.get("agy-flash").tier == "light"

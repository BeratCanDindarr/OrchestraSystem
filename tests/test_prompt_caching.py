"""Tests for prompt caching: ClaudeSDKProvider + config flag + cache_control payload."""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch


class TestCachingConfig(unittest.TestCase):
    def test_caching_config_returns_dict(self):
        from orchestra import config
        cfg = config.caching_config()
        self.assertIsInstance(cfg, dict)

    def test_caching_enabled_default_false(self):
        from orchestra import config
        cfg = config.caching_config()
        self.assertFalse(cfg.get("enabled", False))

    def test_caching_provider_default_sdk(self):
        from orchestra import config
        cfg = config.caching_config()
        self.assertEqual(cfg.get("provider", "sdk"), "sdk")

    def test_caching_models_has_light_medium_heavy(self):
        from orchestra import config
        models = config.caching_config().get("models", {})
        self.assertIn("light", models)
        self.assertIn("medium", models)
        self.assertIn("heavy", models)


class TestClaudeSDKProviderAvailability(unittest.TestCase):
    def test_unavailable_without_api_key(self):
        from orchestra.providers.claude_sdk import ClaudeSDKProvider
        provider = ClaudeSDKProvider()
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(provider.is_available())

    def test_available_with_api_key(self):
        from orchestra.providers.claude_sdk import ClaudeSDKProvider
        provider = ClaudeSDKProvider()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key"}):
            self.assertTrue(provider.is_available())

    def test_model_label_uses_config_models(self):
        from orchestra.providers.claude_sdk import ClaudeSDKProvider
        label = ClaudeSDKProvider().model_label("light")
        self.assertIn("claude_sdk/", label)

    def test_model_label_heavy_contains_opus(self):
        from orchestra.providers.claude_sdk import ClaudeSDKProvider
        label = ClaudeSDKProvider().model_label("heavy")
        self.assertIn("opus", label)

    def test_build_command_raises(self):
        from orchestra.providers.claude_sdk import ClaudeSDKProvider
        with self.assertRaises(NotImplementedError):
            ClaudeSDKProvider().build_command("hello", "light")


class TestClaudeSDKProviderRun(unittest.TestCase):
    def _mock_response(self, text: str):
        block = MagicMock()
        block.text = text
        resp = MagicMock()
        resp.content = [block]
        return resp

    def test_run_sends_cache_control_ephemeral(self):
        from orchestra.providers.claude_sdk import ClaudeSDKProvider

        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_response("Hello")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            with patch("anthropic.Anthropic", return_value=mock_client):
                output, code = ClaudeSDKProvider().run("hello", "light")

        self.assertEqual(code, 0)
        self.assertEqual(output, "Hello")
        system = mock_client.messages.create.call_args.kwargs["system"]
        self.assertEqual(system[0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(system[0]["type"], "text")

    def test_run_uses_opus_for_heavy(self):
        from orchestra.providers.claude_sdk import ClaudeSDKProvider

        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_response("ok")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            with patch("anthropic.Anthropic", return_value=mock_client):
                ClaudeSDKProvider().run("task", "heavy")

        model = mock_client.messages.create.call_args.kwargs["model"]
        self.assertIn("opus", model)

    def test_run_returns_error_on_api_status_error(self):
        import anthropic
        from orchestra.providers.claude_sdk import ClaudeSDKProvider

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_client.messages.create.side_effect = anthropic.APIStatusError(
            "rate limited", response=mock_resp, body={}
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            with patch("anthropic.Anthropic", return_value=mock_client):
                output, code = ClaudeSDKProvider().run("hello", "light")

        self.assertNotEqual(code, 0)
        self.assertIn("[ERROR]", output)

    def test_run_returns_error_on_connection_error(self):
        import anthropic
        from orchestra.providers.claude_sdk import ClaudeSDKProvider

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.APIConnectionError(
            request=MagicMock()
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            with patch("anthropic.Anthropic", return_value=mock_client):
                output, code = ClaudeSDKProvider().run("hello", "light")

        self.assertNotEqual(code, 0)
        self.assertIn("[ERROR]", output)


class TestClaudeProviderCachingDelegation(unittest.TestCase):
    def test_delegates_to_sdk_when_enabled(self):
        from orchestra.providers.claude import ClaudeProvider

        mock_sdk = MagicMock()
        mock_sdk.is_available.return_value = True
        mock_sdk.run.return_value = ("sdk_response", 0)

        with patch("orchestra.config.caching_config", return_value={"enabled": True, "provider": "sdk"}):
            with patch("orchestra.providers.claude_sdk.ClaudeSDKProvider", return_value=mock_sdk):
                output, code = ClaudeProvider().run("hello", "light")

        self.assertEqual(output, "sdk_response")
        self.assertEqual(code, 0)
        mock_sdk.run.assert_called_once()

    def test_falls_back_to_cli_when_disabled(self):
        from orchestra.providers.claude import ClaudeProvider

        with patch("orchestra.config.caching_config", return_value={"enabled": False}):
            with patch("subprocess.run") as mock_sub:
                mock_sub.return_value = MagicMock(stdout="cli_response", returncode=0)
                output, code = ClaudeProvider().run("hello", "light")

        self.assertEqual(output, "cli_response")
        self.assertEqual(code, 0)

    def test_falls_back_to_cli_when_sdk_unavailable(self):
        from orchestra.providers.claude import ClaudeProvider

        mock_sdk = MagicMock()
        mock_sdk.is_available.return_value = False

        with patch("orchestra.config.caching_config", return_value={"enabled": True, "provider": "sdk"}):
            with patch("orchestra.providers.claude_sdk.ClaudeSDKProvider", return_value=mock_sdk):
                with patch("subprocess.run") as mock_sub:
                    mock_sub.return_value = MagicMock(stdout="cli_fallback", returncode=0)
                    output, code = ClaudeProvider().run("hello", "light")

        self.assertEqual(output, "cli_fallback")


if __name__ == "__main__":
    unittest.main()

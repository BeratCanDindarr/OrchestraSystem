"""Claude CLI provider adapter."""
from __future__ import annotations

import os
import shutil
import subprocess

from orchestra import config
from orchestra.providers.base import BaseProvider


MODELS = {
    "sonnet": "sonnet",
    "opus": "opus",
    "medium": "sonnet",
    "high": "opus",
    "light": "sonnet",
    "heavy": "opus",
}


class ClaudeProvider(BaseProvider):
    name = "claude"

    def _binary(self) -> str:
        return shutil.which("claude") or "/opt/homebrew/bin/claude"

    def is_available(self) -> bool:
        if shutil.which("claude") is not None:
            return True
        shell = os.environ.get("SHELL", "/bin/sh")
        probe_timeout = int(config.availability_config().get("cli_probe_timeout_seconds", 5))
        try:
            result = subprocess.run(
                [shell, "-lc", f'"{self._binary()}" --version'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=probe_timeout,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0

    def model_label(self, effort_or_model: str) -> str:
        model = MODELS.get(effort_or_model, effort_or_model)
        return f"claude/{model}"

    def build_command(self, prompt: str, effort_or_model: str) -> list[str]:
        model = MODELS.get(effort_or_model, effort_or_model)
        return [
            self._binary(),
            "-p",
            "--dangerously-skip-permissions",
            "--output-format",
            "text",
            "--model",
            model,
            prompt,
        ]

    def run(self, prompt: str, effort_or_model: str, timeout: int = 180) -> tuple[str, int]:
        """Delegate to SDK provider when caching is enabled; otherwise use CLI subprocess."""
        cfg = config.caching_config()
        if cfg.get("enabled") and cfg.get("provider") == "sdk":
            from orchestra.providers.claude_sdk import ClaudeSDKProvider
            sdk = ClaudeSDKProvider()
            if sdk.is_available():
                return sdk.run(prompt, effort_or_model, timeout)
        return super().run(prompt, effort_or_model, timeout)

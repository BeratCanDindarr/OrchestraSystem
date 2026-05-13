"""Gemini CLI provider adapter."""
from __future__ import annotations

import os
import subprocess
import shutil
from orchestra import config
from orchestra.providers.base import BaseProvider


# Model mapping (matches agent_orchestrator.py)
MODELS = {
    "pro":        "pro",         # gmn-pro (gemini-2.5-pro)
    "flash":      "flash",       # gmn-fast (gemini-2.5-flash)
    "flash-lite": "flash-lite",  # light
    # tier shortcuts
    "heavy":  "pro",
    "medium": "flash",
    "light":  "flash-lite",
}


class GeminiProvider(BaseProvider):
    name = "gemini"

    def _binary(self) -> str:
        return shutil.which("gemini") or "/opt/homebrew/bin/gemini"

    def is_available(self) -> bool:
        if shutil.which("gemini") is not None:
            return True
        shell = os.environ.get("SHELL", "/bin/sh")
        probe_timeout = int(config.availability_config().get("cli_probe_timeout_seconds", 5))
        try:
            result = subprocess.run(
                [shell, "-lc", "gemini --version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=probe_timeout,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0

    def model_label(self, effort_or_model: str) -> str:
        model = MODELS.get(effort_or_model, effort_or_model)
        return f"gemini/{model}"

    def build_command(self, prompt: str, effort_or_model: str) -> list[str]:
        model = MODELS.get(effort_or_model, effort_or_model)
        return [self._binary(), "-p", prompt, "-m", model]

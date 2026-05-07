"""Claude CLI provider adapter."""
from __future__ import annotations

import os
import shutil
import subprocess

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
        try:
            result = subprocess.run(
                [shell, "-lc", f'"{self._binary()}" --version'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
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

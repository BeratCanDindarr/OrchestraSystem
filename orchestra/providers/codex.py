"""Codex CLI provider adapter."""
from __future__ import annotations

import os
import subprocess
import shutil
from orchestra.providers.base import BaseProvider


# Effort level mapping (matches agent_orchestrator.py)
EFFORTS = {
    "xhigh":  "xhigh",   # cdx-deep
    "high":   "high",
    "medium": "medium",
    "low":    "low",      # cdx-fast
    # tier shortcuts
    "heavy":  "xhigh",
    "light":  "low",
}


class CodexProvider(BaseProvider):
    name = "codex"

    def _binary(self) -> str:
        return shutil.which("codex") or "/opt/homebrew/bin/codex"

    def is_available(self) -> bool:
        if shutil.which("codex") is not None:
            return True
        shell = os.environ.get("SHELL", "/bin/sh")
        try:
            result = subprocess.run(
                [shell, "-lc", "codex --version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0

    def model_label(self, effort_or_model: str) -> str:
        effort = EFFORTS.get(effort_or_model, effort_or_model)
        return f"gpt-5.4/{effort}"

    def build_command(self, prompt: str, effort_or_model: str) -> list[str]:
        effort = EFFORTS.get(effort_or_model, effort_or_model)
        return [
            self._binary(), "exec",
            "--skip-git-repo-check",
            "-c", f'model_reasoning_effort="{effort}"',
            prompt,
        ]

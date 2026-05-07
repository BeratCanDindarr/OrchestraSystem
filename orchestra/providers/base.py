"""Base provider interface."""
from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod


class BaseProvider(ABC):
    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def model_label(self, effort_or_model: str) -> str: ...

    @abstractmethod
    def build_command(self, prompt: str, effort_or_model: str) -> list[str]: ...

    def run(self, prompt: str, effort_or_model: str, timeout: int = 180) -> tuple[str, int]:
        """
        Execute provider command synchronously.
        Returns (stdout, returncode).
        """
        cmd = self.build_command(prompt, effort_or_model)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=timeout,
            )
            output = result.stdout.strip()
            if not output and result.returncode != 0:
                output = result.stderr.strip()
            return output, result.returncode
        except subprocess.TimeoutExpired:
            return f"[ERROR] Timeout after {timeout}s", 124
        except Exception as e:
            return f"[ERROR] {e}", 1

"""Base provider interface."""
from __future__ import annotations

import asyncio
import subprocess
from abc import ABC, abstractmethod
from typing import Iterator


class BaseProvider(ABC):
    name: str = "base"
    _last_usage: dict[str, int] | None = None
    supports_native_messages: bool = False

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def model_label(self, effort_or_model: str) -> str: ...

    @abstractmethod
    def build_command(self, prompt: str, effort_or_model: str) -> list[str]: ...

    def run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        """
        Execute provider command synchronously.
        Returns (stdout, returncode).
        """
        cmd = self.build_command(prompt, effort_or_model)
        cwd = kwargs.get("cwd")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            output = result.stdout.strip()
            if not output and result.returncode != 0:
                output = result.stderr.strip()
            return output, result.returncode
        except subprocess.TimeoutExpired:
            return f"[ERROR] Timeout after {timeout}s", 124
        except Exception as e:
            return f"[ERROR] {e}", 1

    async def async_run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        """
        Execute provider command asynchronously.
        Default implementation wraps the synchronous run() in a thread.
        """
        return await asyncio.to_thread(self.run, prompt, effort_or_model, timeout=timeout, **kwargs)

    def stream(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> Iterator[str]:
        """Stream provider output as a generator of text chunks."""
        raise NotImplementedError(f"Provider {self.name} does not support streaming")

    def native_messages(self, payload: dict, timeout: int = 180, **kwargs) -> dict:
        """Anthropic-shaped request in, Anthropic-shaped response dict out.

        `payload`: {model?, system?, messages, tools?, tool_choice?, max_tokens?}.
        Returns at minimum: {"content": [<Anthropic content blocks>],
        "stop_reason": "end_turn"|"tool_use", "usage": {...}?}.
        Only providers that can emit structured tool calls implement this.
        """
        raise NotImplementedError(f"Provider {self.name} does not support native messages")

    def last_usage(self) -> dict[str, int] | None:
        """Return token usage from the last successful call, or None."""
        return self._last_usage

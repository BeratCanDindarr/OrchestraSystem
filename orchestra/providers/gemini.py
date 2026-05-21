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

    def run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        if kwargs.get("response_format") == "json":
            prompt += "\n\nPlease respond with valid JSON only."
        return super().run(prompt, effort_or_model, timeout, **kwargs)

    async def async_run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        import asyncio
        if kwargs.get("response_format") == "json":
            prompt += "\n\nPlease respond with valid JSON only."
        cmd = self.build_command(prompt, effort_or_model)
        pid_callback = kwargs.get("pid_callback")
        process_manager = kwargs.get("process_manager")

        async def _run() -> tuple[str, int]:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                cwd=kwargs.get("cwd"),
            )
            if process_manager:
                process_manager.register(proc.pid)
            if pid_callback:
                pid_callback(proc.pid)
            try:
                stdout, stderr = await proc.communicate()
                output = stdout.decode().strip()
                if not output and proc.returncode != 0:
                    output = stderr.decode().strip()
                return output, proc.returncode or 0
            finally:
                if process_manager:
                    process_manager.unregister(proc.pid)
                if pid_callback:
                    pid_callback(None)
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()

        try:
            return await asyncio.wait_for(_run(), timeout=timeout)
        except asyncio.TimeoutError:
            return f"[ERROR] Timeout after {timeout}s", 124
        except Exception as e:
            return f"[ERROR] {e}", 1

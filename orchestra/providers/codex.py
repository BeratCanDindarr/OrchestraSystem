"""Codex CLI provider adapter."""
from __future__ import annotations

import os
import subprocess
import shutil
import time
from orchestra import config
from orchestra.providers.base import AgentTimeout, BaseProvider, read_while_active


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


LOGIN_CHECK_SECONDS = 60.0
_login_checked_at = 0.0
_logged_in = False


def _codex_logged_in(binary: str) -> bool:
    """Whether the Codex CLI has an account, asked at most once a minute.

    An installed CLI without a login fails every call with 401 after a long wait, and the
    run then falls back to another model under Codex's name.
    """
    global _login_checked_at, _logged_in
    now = time.monotonic()
    if _login_checked_at and now - _login_checked_at < LOGIN_CHECK_SECONDS:
        return _logged_in
    try:
        done = subprocess.run([binary, "login", "status"], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, timeout=5)
        _logged_in = done.returncode == 0
    except (OSError, subprocess.SubprocessError):
        _logged_in = False
    _login_checked_at = now
    return _logged_in


class CodexProvider(BaseProvider):
    name = "codex"

    def _binary(self) -> str:
        return shutil.which("codex") or "/opt/homebrew/bin/codex"

    def is_available(self) -> bool:
        if not config.provider_enabled("codex"):
            return False
        if shutil.which("codex") is not None:
            return _codex_logged_in(self._binary())
        shell = os.environ.get("SHELL", "/bin/sh")
        probe_timeout = int(config.availability_config().get("cli_probe_timeout_seconds", 5))
        try:
            result = subprocess.run(
                [shell, "-lc", "codex --version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=probe_timeout,
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
            # Codex agents answer; whatever ~/.codex/config.toml says, they do not change the project.
            "-s", "read-only",
            "-c", f'model_reasoning_effort="{effort}"',
            prompt,
        ]

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
                # codex exec prints its progress to stderr: while it does, it is working
                stdout, stderr = await read_while_active(proc, silence=timeout, cap=max(timeout, config.max_agent_seconds()))
                output = stdout.decode(errors="replace").strip()
                if not output and proc.returncode != 0:
                    output = stderr.decode(errors="replace").strip()
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
            return await _run()
        except AgentTimeout as stopped:
            return f"[ERROR] Timeout: {stopped}", 124
        except Exception as e:
            return f"[ERROR] {e}", 1

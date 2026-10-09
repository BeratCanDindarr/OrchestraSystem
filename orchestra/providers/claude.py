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


def _as_list(value: object) -> list[str]:
    """A config value given as "a,b" or ["a", "b"], as a list of names."""
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part) for part in value if str(part).strip()]
    return []


class ClaudeProvider(BaseProvider):
    """Runs the Claude CLI as an agent.

    What the agent may do comes from a config section, `[providers.claude]` by
    default: `skip_permissions` (true when unset, the historical behaviour: the
    agent can edit files and run commands in the caller's directory), `tools`,
    `allowed_tools`, `disallowed_tools`, `permission_mode`, `append_system_prompt`
    and `extra_args`. A second instance with another `config_key` gives a
    second profile, e.g. a read-only reviewer next to a writing implementer.
    """

    name = "claude"

    def __init__(self, config_key: str = "claude") -> None:
        self.config_key = config_key

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
        profile = config.provider_config(self.config_key)
        command = [self._binary(), "-p"]
        if not profile.get("session_persistence", False):
            # An agent call is not a session the user would resume: unsaved, it stays out of
            # the project's `claude --resume` list.
            command.append("--no-session-persistence")
        if profile.get("skip_permissions", True):
            command.append("--dangerously-skip-permissions")
        if profile.get("permission_mode"):
            command += ["--permission-mode", str(profile["permission_mode"])]
        for key, flag in (("tools", "--tools"), ("allowed_tools", "--allowed-tools"),
                          ("disallowed_tools", "--disallowed-tools")):
            names = _as_list(profile.get(key))
            if names:
                command += [flag, ",".join(names)]
        if profile.get("append_system_prompt"):
            command += ["--append-system-prompt", str(profile["append_system_prompt"])]
        command += _as_list(profile.get("extra_args"))
        return command + ["--output-format", "text", "--model", model, prompt]

    def run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        """Delegate to SDK provider when caching is enabled; otherwise use CLI subprocess."""
        if kwargs.get("response_format") == "json":
            prompt += "\n\nPlease respond with valid JSON only."
        cfg = config.caching_config()
        if cfg.get("enabled") and cfg.get("provider") == "sdk":
            from orchestra.providers.claude_sdk import ClaudeSDKProvider
            sdk = ClaudeSDKProvider()
            if sdk.is_available():
                return sdk.run(prompt, effort_or_model, timeout, **kwargs)
        return super().run(prompt, effort_or_model, timeout, **kwargs)

    async def async_run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        import asyncio
        """Async delegate to SDK provider when caching is enabled; otherwise use CLI subprocess."""
        if kwargs.get("response_format") == "json":
            prompt += "\n\nPlease respond with valid JSON only."
        cfg = config.caching_config()
        if cfg.get("enabled") and cfg.get("provider") == "sdk":
            from orchestra.providers.claude_sdk import ClaudeSDKProvider
            sdk = ClaudeSDKProvider()
            if sdk.is_available():
                return await sdk.async_run(prompt, effort_or_model, timeout, **kwargs)

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

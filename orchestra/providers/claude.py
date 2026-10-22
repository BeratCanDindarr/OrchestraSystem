"""Claude CLI provider adapter."""
from __future__ import annotations

import contextvars
import json
import os
import shutil
import signal
import subprocess

from orchestra import config
from orchestra.providers import warm_pool
from orchestra.providers.base import AgentTimeout, _wait_for, read_while_active
from orchestra.providers.base import BaseProvider


MODELS = {
    "sonnet": "sonnet",
    "opus": "opus",
    "medium": "sonnet",
    "high": "opus",
    "light": "sonnet",
    "heavy": "opus",
}


# Usage of the call made in the current thread or task: cld-fast and cld-deep share one
# provider object, so an attribute on it would mix up agents running side by side.
_usage: contextvars.ContextVar[dict | None] = contextvars.ContextVar("claude_usage", default=None)


def parse_result(output: str) -> tuple[str, dict | None, bool]:
    """(answer, usage, failed) out of `claude -p --output-format json` or `stream-json`.

    The CLI reports the tokens it used and what the call cost at API prices; in a stream that
    is the last `result` line. Output that is neither (an older CLI, an error printed as text)
    is returned as it is.
    """
    data = _result_object(output)
    if not isinstance(data, dict) or "result" not in data:
        return output, None, False
    used = data.get("usage")
    if not isinstance(used, dict):
        used = {}
    usage: dict = {
        "prompt_tokens": sum(int(used.get(key) or 0) for key in
                             ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")),
        "completion_tokens": int(used.get("output_tokens") or 0),
    }
    if isinstance(data.get("total_cost_usd"), (int, float)):
        usage["cost_usd"] = float(data["total_cost_usd"])
    return str(data.get("result") or "").strip(), usage, bool(data.get("is_error"))


def _result_object(output: str) -> object:
    """The JSON object the CLI ended with: the whole output, or the last `result` line of a stream."""
    try:
        return json.loads(output)
    except (TypeError, ValueError):
        pass
    for line in reversed((output or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and '"result"' in line:
            try:
                data = json.loads(line)
            except ValueError:
                continue
            if isinstance(data, dict) and data.get("type") == "result":
                return data
    return None


def _as_list(value: object) -> list[str]:
    """A config value given as "a,b" or ["a", "b"], as a list of names."""
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part) for part in value if str(part).strip()]
    return []


READ_ONLY_TOOLS = ("Read", "Grep", "Glob", "Skill")   # a skill adds instructions, no rights


def unity_mcp_config() -> dict | None:
    """How to start UnityMCP for this project when `[unity] enabled`, else None (orchestra/unity_read.py)."""
    from orchestra.unity_read import upstream_config
    return upstream_config()
WRITING_PROFILES = frozenset({"claude_write"})


class ClaudeProvider(BaseProvider):
    """Runs the Claude CLI as an agent.

    What the agent may do comes from a config section, `[providers.claude]` by
    default: `skip_permissions`, `tools`, `allowed_tools`, `disallowed_tools`,
    `permission_mode`, `append_system_prompt` and `extra_args`. A second instance with
    another `config_key` gives a second profile, e.g. a read-only reviewer next to a
    writing implementer.

    Without a profile an agent can only read (`READ_ONLY_TOOLS`). The config file is found
    from the working directory, so "no profile" also means "started somewhere else": that
    must not turn a reviewer into an agent that edits files and runs commands. Only the
    profile named `claude_write`, which exists to change files, defaults to full permissions.
    """

    name = "claude"
    reports_errors = True   # the CLI's JSON says is_error; _answer turns that into exit code 1

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

    def can_write(self, effort_or_model: str = "") -> bool:
        profile = config.provider_config(self.config_key)
        if profile.get("skip_permissions", self.config_key in WRITING_PROFILES):
            return True
        if str(profile.get("permission_mode") or "") in ("acceptEdits", "bypassPermissions"):
            return True
        granted = set(_as_list(profile.get("tools"))) | set(_as_list(profile.get("allowed_tools")))
        return bool(granted & {"Write", "Edit", "MultiEdit", "NotebookEdit", "Bash"})

    def build_command(self, prompt: str, effort_or_model: str) -> list[str]:
        model = MODELS.get(effort_or_model, effort_or_model)
        profile = config.provider_config(self.config_key)
        command = [self._binary(), "-p"]
        if not profile.get("session_persistence", False):
            # An agent call is not a session the user would resume: unsaved, it stays out of
            # the project's `claude --resume` list.
            command.append("--no-session-persistence")
        skip_permissions = bool(profile.get("skip_permissions", self.config_key in WRITING_PROFILES))
        if skip_permissions:
            command.append("--dangerously-skip-permissions")
        elif not any(profile.get(key) for key in ("tools", "allowed_tools", "permission_mode")):
            profile = {**profile, "tools": list(READ_ONLY_TOOLS)}
        if profile.get("permission_mode"):
            command += ["--permission-mode", str(profile["permission_mode"])]
        for key, flag in (("tools", "--tools"), ("allowed_tools", "--allowed-tools"),
                          ("disallowed_tools", "--disallowed-tools")):
            names = _as_list(profile.get(key))
            if names:
                command += [flag, ",".join(names)]
        upstream = None if self.can_write() else unity_mcp_config()
        if upstream:
            # A reading agent sees the editor through the read-only proxy, never UnityMCP itself.
            import sys
            proxy = {"command": sys.executable, "args": ["-m", "orchestra.unity_read", json.dumps(upstream)]}
            command += ["--mcp-config", json.dumps({"mcpServers": {"unity": proxy}}), "--allowed-tools", "mcp__unity"]
        if profile.get("append_system_prompt"):
            command += ["--append-system-prompt", str(profile["append_system_prompt"])]
        command += _as_list(profile.get("extra_args"))
        # A stream: every thinking and tool step is output, so a working agent is never silent.
        return command + ["--output-format", "stream-json", "--verbose", "--include-partial-messages",
                          "--model", model, prompt]

    def last_usage(self) -> dict | None:
        """Tokens and cost the CLI reported for the last call in this thread or task."""
        return _usage.get()

    def reset_usage(self) -> None:
        _usage.set(None)

    @staticmethod
    def _answer(output: str, returncode: int) -> tuple[str, int]:
        """Turn the CLI's JSON into the answer text and note what the call used."""
        if returncode != 0:
            text, _, _ = parse_result(output)   # a stream that failed still ends with its result line
            return (text if text != output else output[-2000:]), returncode
        text, usage, failed = parse_result(output)
        _usage.set(usage)
        if failed:
            return f"[ERROR] {text}" if text else "[ERROR] claude reported an error", 1
        return text, 0

    def run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        if kwargs.get("response_format") == "json":
            prompt += "\n\nPlease respond with valid JSON only."
        _usage.set(None)
        return self._answer(*super().run(prompt, effort_or_model, timeout, **kwargs))

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
                start_new_session=True,   # no terminal to ask on; ended as a group at the deadline
            )
            if process_manager:
                process_manager.register(proc.pid)
            if pid_callback:
                pid_callback(proc.pid)
            try:
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
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except OSError:
                        proc.kill()
                    await proc.wait()

        async def _run_warm(proc) -> tuple[str, int]:
            # A process the pool started ahead of time: its start-up is already paid.
            if process_manager:
                process_manager.register(proc.pid)
            if pid_callback:
                pid_callback(proc.pid)
            try:
                warm_pool.send(proc, prompt)
                try:
                    stdout, stderr, returncode = await asyncio.to_thread(_wait_for, proc, timeout)
                except subprocess.TimeoutExpired:
                    raise AgentTimeout(f"silent for {timeout:g}s, or still running at [orchestra] max_agent_seconds") from None
                output = (stdout or "").strip()
                if not output and returncode != 0:
                    output = (stderr or "").strip()
                return output, returncode or 0
            finally:
                if process_manager:
                    process_manager.unregister(proc.pid)
                if pid_callback:
                    pid_callback(None)

        _usage.set(None)
        warm = warm_pool.take(cmd, kwargs.get("cwd"))
        warm_pool.refill(cmd, kwargs.get("cwd"))   # the process the next call with this command takes
        try:
            return self._answer(*await (_run_warm(warm) if warm is not None else _run()))
        except AgentTimeout as stopped:
            return f"[ERROR] Timeout: {stopped}", 124
        except Exception as e:
            return f"[ERROR] {e}", 1

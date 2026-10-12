"""Base provider interface."""
from __future__ import annotations

import asyncio
import contextvars
import os
import signal
import subprocess
from abc import ABC, abstractmethod
from typing import Iterator

ERROR_DETAIL_CHARS = 400


def run_command(cmd: list[str], timeout: int, cwd=None, on_start=None, on_exit=None) -> tuple[str, str, int]:
    """Run an agent CLI to completion and return (stdout, stderr, returncode).

    The CLI gets a session of its own: it has no terminal to open for a question nobody will
    answer, and at the deadline the whole group is ended, the helper processes it started
    (MCP servers) included. Raises subprocess.TimeoutExpired at the deadline.

    `on_start(pid)` is called once the process exists and `on_exit(pid)` when it is gone, so
    that a cancel can end the group (the pid is the group's id) while this call is waiting.
    """
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                               text=True, cwd=cwd, start_new_session=True)
    if on_start:
        on_start(process.pid)
    try:
        return _wait_for(process, timeout)
    finally:
        if on_exit:
            on_exit(process.pid)


def _wait_for(process: subprocess.Popen, timeout: int) -> tuple[str, str, int]:
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return stdout, stderr, process.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise


def error_detail(stderr: str) -> str:
    """The part of a failed CLI's stderr that says why: its error lines, else its tail."""
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    errors = [line for line in lines if "error" in line.lower()]
    return "\n".join(errors or lines[-3:])[-ERROR_DETAIL_CHARS:]


def _process_hooks(kwargs: dict) -> dict:
    """run_command hooks that register the CLI with the run's process manager and pid callback.

    Empty when the caller gave neither, so a plain call stays `run_command(cmd, timeout, cwd)`.
    """
    manager, report = kwargs.get("process_manager"), kwargs.get("pid_callback")
    if manager is None and report is None:
        return {}

    def on_start(pid: int) -> None:
        if manager is not None:
            manager.register(pid)
        if report is not None:
            report(pid)

    def on_exit(pid: int) -> None:
        if manager is not None:
            manager.unregister(pid)
        if report is not None:
            report(None)

    return {"on_start": on_start, "on_exit": on_exit}


# What each provider's last call used, per thread or task: provider id -> usage. One provider
# object serves every agent that uses it, several at once, so the usage cannot live on it.
_USAGE: contextvars.ContextVar[dict] = contextvars.ContextVar("provider_usage", default={})


class BaseProvider(ABC):
    name: str = "base"
    supports_native_messages: bool = False

    @property
    def _last_usage(self) -> dict | None:
        return _USAGE.get().get(id(self))

    @_last_usage.setter
    def _last_usage(self, usage: dict | None) -> None:
        _USAGE.set({**_USAGE.get(), id(self): usage})

    def reset_usage(self) -> None:
        """Forget the last call's usage; called before a call so a failure reports none."""
        self._last_usage = None

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
            stdout, stderr, returncode = run_command(cmd, timeout, cwd, **_process_hooks(kwargs))
            output = stdout.strip()
            if returncode != 0:
                # A CLI may print something to stdout and the reason for failing to stderr
                # (Kimi: a version line, then "usage limit reached"): keep the reason.
                output = f"{output}\n{error_detail(stderr)}".strip() if output else stderr.strip()
            return output, returncode
        except subprocess.TimeoutExpired:
            return f"[ERROR] Timeout after {timeout}s", 124
        except Exception as e:
            return f"[ERROR] {e}", 1

    async def async_run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        """
        Execute provider command asynchronously.
        Default implementation wraps the synchronous run() in a thread.
        """
        def call() -> tuple[tuple[str, int], dict | None]:
            return self.run(prompt, effort_or_model, timeout=timeout, **kwargs), self.last_usage()

        # The thread works on a copy of this task's context: what run() notes there is
        # brought back, or the caller would read no usage at all.
        result, usage = await asyncio.to_thread(call)
        self._last_usage = usage
        return result

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
        """Token usage of the last call made in this thread or task, or None."""
        return self._last_usage

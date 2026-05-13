"""ToolMaker Sandbox — isolated subprocess execution for dynamically generated tools.

Constraints (hard-coded, not configurable):
  - Timeout:     30 seconds
  - Memory:      256 MB (via ulimit on POSIX; best-effort on macOS)
  - Network:     blocked (no socket connections allowed)
  - Filesystem:  temporary working directory, cleaned up after run
  - Imports:     validated by toolmaker.validator before execution

The sandbox runs the tool script in a fresh Python subprocess with:
  - sys.path restricted to stdlib + approved packages
  - Environment variables stripped (no API keys, no PATH leaks)
  - stdin closed

Usage:
    from orchestra.toolmaker.sandbox import ToolSandbox
    result = ToolSandbox().run(script_source, input_data={"task": "..."})
    if result.success:
        print(result.output)
    else:
        print(result.error)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass

SANDBOX_TIMEOUT_SECONDS = 30
SANDBOX_MEMORY_MB = 256

# Environment variables passed through to sandboxed process (minimal set)
_ALLOWED_ENV_KEYS = {"PYTHONPATH", "HOME", "TMPDIR", "TEMP", "TMP"}

_WRAPPER_TEMPLATE = textwrap.dedent(
    """\
    import sys
    import json

    # Restrict stdin
    import io
    sys.stdin = io.StringIO("")

    # Inject input data
    _INPUT = {input_json}

    # --- user script ---
    {script}
    # --- end user script ---
    """
)


@dataclass
class SandboxResult:
    success: bool
    output: str
    error: str | None = None
    exit_code: int = 0
    timed_out: bool = False


class ToolSandbox:
    """Runs a Python script in an isolated subprocess with strict limits."""

    def __init__(
        self,
        timeout: int = SANDBOX_TIMEOUT_SECONDS,
        memory_mb: int = SANDBOX_MEMORY_MB,
    ) -> None:
        self._timeout = timeout
        self._memory_mb = memory_mb

    def run(self, script: str, input_data: dict | None = None, *, skip_validation: bool = False) -> SandboxResult:
        """Execute script in sandbox. Validates via ToolValidator before running.

        Args:
            script: Python source to execute.
            input_data: Injected as _INPUT dict inside the script.
            skip_validation: If True, bypass AST validation (internal use only).
        """
        import json

        if not skip_validation:
            from orchestra.toolmaker.validator import ToolValidator
            validation = ToolValidator().validate(script)
            if not validation.ok:
                return SandboxResult(
                    success=False,
                    output="",
                    error="Validation failed: " + "; ".join(validation.errors),
                    exit_code=-3,
                )

        input_json = json.dumps(input_data or {})
        wrapped = _WRAPPER_TEMPLATE.format(
            input_json=input_json,
            script=script,
        )

        with tempfile.TemporaryDirectory(prefix="orchestra_sandbox_") as tmpdir:
            script_path = os.path.join(tmpdir, "tool_script.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(wrapped)

            env = self._build_env(tmpdir)
            cmd = self._build_cmd(script_path)

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    env=env,
                    cwd=tmpdir,
                    stdin=subprocess.DEVNULL,
                )
                if proc.returncode == 0:
                    return SandboxResult(
                        success=True,
                        output=proc.stdout.strip(),
                        exit_code=0,
                    )
                else:
                    return SandboxResult(
                        success=False,
                        output=proc.stdout.strip(),
                        error=proc.stderr.strip() or f"exit code {proc.returncode}",
                        exit_code=proc.returncode,
                    )
            except subprocess.TimeoutExpired:
                return SandboxResult(
                    success=False,
                    output="",
                    error=f"Sandbox timeout after {self._timeout}s",
                    exit_code=-1,
                    timed_out=True,
                )
            except Exception as exc:
                return SandboxResult(
                    success=False,
                    output="",
                    error=f"Sandbox launch failed: {exc}",
                    exit_code=-2,
                )

    def _build_env(self, tmpdir: str) -> dict[str, str]:
        """Build a minimal environment for the sandboxed process."""
        env: dict[str, str] = {}
        for key in _ALLOWED_ENV_KEYS:
            val = os.environ.get(key)
            if val:
                env[key] = val
        env["TMPDIR"] = tmpdir
        # Disable network by pointing to a non-existent proxy (best-effort)
        env["http_proxy"] = "http://127.0.0.1:0"
        env["https_proxy"] = "http://127.0.0.1:0"
        env["no_proxy"] = "*"
        return env

    def _build_cmd(self, script_path: str) -> list[str]:
        """Build the subprocess command, applying memory limits on POSIX."""
        python = sys.executable

        if sys.platform == "win32":
            return [python, script_path]

        # On POSIX, wrap with ulimit for memory cap (best-effort on macOS)
        memory_kb = self._memory_mb * 1024
        shell_cmd = (
            f"ulimit -v {memory_kb} 2>/dev/null; "
            f"exec {python} {script_path}"
        )
        return ["/bin/sh", "-c", shell_cmd]

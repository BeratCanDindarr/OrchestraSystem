from __future__ import annotations
"""WasmSandbox — WebAssembly execution environment for dynamically generated tools.

Constraints:
  - Timeout:     configurable (default 10 seconds)
  - Memory:      configurable (default 128 MB, best-effort OS enforcement)
  - Network:     blocked (no socket APIs exposed)
  - Filesystem:  no pre-opened directories (WASI preopens are empty)
  - Imports:     whitelisted host functions only

Execution backends (tried in order):
  1. Node.js subprocess (most portable, works on macOS where wasmtime JIT
     may be blocked by security policy).
  2. wasmtime in-process (future — currently disabled on this platform
     because the JIT triggers SIGKILL under some endpoint-protection
     configurations).

Usage:
    from orchestra.toolmaker.wasm_sandbox import WasmSandbox
    result = WasmSandbox().run(wasm_bytes, input_data={"task": "..."})
    if result.success:
        print(result.output)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass

from orchestra.toolmaker.sandbox import SandboxResult

# Each WASM page is 64 KiB.
_WASM_PAGE_SIZE = 64 * 1024

_NODE_WRAPPER = textwrap.dedent(
    '''\
    const fs = require("fs");
    const { WASI } = require("wasi");

    const wasmPath = process.env.WASM_PATH;
    const inputData = JSON.parse(process.env.WASM_INPUT || "{}");
    const allowedFuncs = JSON.parse(process.env.WASM_ALLOWED_FUNCS || "[]");
    const useWasi = process.env.WASM_USE_WASI === "1";

    const wasmBuffer = fs.readFileSync(wasmPath);

    // Build import object — only expose explicitly allowed host functions.
    const importObject = {};
    for (const name of allowedFuncs) {
        const parts = name.split("::");
        if (parts.length !== 2) continue;
        const [mod, func] = parts;
        if (!importObject[mod]) importObject[mod] = {};
        importObject[mod][func] = () => {
            throw new Error(`Host function "${name}" is not implemented.`);
        };
    }

    // If the module imports WASI, provide a locked-down snapshot.
    let wasi = null;
    if (useWasi) {
        wasi = new WASI({
            version: "preview1",
            args: [],
            env: {},
            preopens: {}   // NO pre-opened directories → filesystem denied
        });
        importObject.wasi_snapshot_preview1 = wasi.wasiImport;
    }

    try {
        const mod = new WebAssembly.Module(wasmBuffer);
        const instance = new WebAssembly.Instance(mod, importObject);

        let output = "";
        if (useWasi && instance.exports._start) {
            const code = wasi.start(instance);
            if (code !== 0) {
                console.log(JSON.stringify({
                    success: false,
                    error: `WASI exit code ${code}`,
                    output: output
                }));
                process.exit(0);
            }
        } else if (instance.exports.run) {
            const raw = instance.exports.run();
            output = String(raw);
        } else if (instance.exports._start) {
            instance.exports._start();
        } else {
            console.log(JSON.stringify({
                success: false,
                error: "No entrypoint found (expected 'run' or '_start' export).",
                output: ""
            }));
            process.exit(0);
        }

        console.log(JSON.stringify({ success: true, output: output }));
    } catch (err) {
        console.log(JSON.stringify({
            success: false,
            error: err.message,
            output: ""
        }));
        process.exit(0);
    }
    '''
)


class WasmSandbox:
    """WASM sandbox with timeout, memory limit, and syscall filtering."""

    def __init__(self, timeout: int = 10, memory_mb: int = 128) -> None:
        self._timeout = timeout
        self._memory_mb = memory_mb
        self._node = shutil.which("node")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        wasm_bytes: bytes,
        input_data: dict | None = None,
        *,
        allowed_host_funcs: list[str] | None = None,
    ) -> SandboxResult:
        """Execute a WASM module in a sandboxed environment.

        Args:
            wasm_bytes: Compiled WebAssembly binary (*.wasm).
            input_data: JSON-serializable dict injected as an env var.
            allowed_host_funcs: List of host function names the guest may import.
                                Format: ``["module::func", ...]``.

        Returns:
            SandboxResult with ``success``, ``output``, ``error``, and
            ``timed_out`` fields.
        """
        # 1. Basic binary validation.
        ok, reason = self.validate(wasm_bytes)
        if not ok:
            return SandboxResult(
                success=False,
                output="",
                error=f"WASM validation failed: {reason}",
                exit_code=-3,
            )

        # 2. Structural validation + memory-limit pre-flight.
        try:
            mem_check = self._check_memory_limit(wasm_bytes)
            if not mem_check.ok:
                return SandboxResult(
                    success=False,
                    output="",
                    error=f"Memory limit exceeded: {mem_check.reason}",
                    exit_code=-4,
                )
        except Exception as exc:
            # wasmtime inspection can fail on edge-case modules; treat as
            # validation error rather than crashing.
            return SandboxResult(
                success=False,
                output="",
                error=f"WASM inspection failed: {exc}",
                exit_code=-3,
            )

        # 3. If Node.js is unavailable, degrade gracefully.
        if not self._node:
            return SandboxResult(
                success=False,
                output="",
                error=(
                    "WASM sandbox requires a JavaScript runtime (Node.js) "
                    "to execute WebAssembly, but 'node' was not found in PATH."
                ),
                exit_code=-42,
            )

        # 4. Execute in isolated subprocess.
        return self._run_nodejs(
            wasm_bytes,
            input_data=input_data,
            allowed_host_funcs=allowed_host_funcs,
        )

    def validate(self, wasm_bytes: bytes) -> tuple[bool, str]:
        """Validate WASM binary header (magic + version) and structural integrity.

        Returns:
            (is_valid, reason)
        """
        if len(wasm_bytes) < 8:
            return False, "WASM binary too short (< 8 bytes)"
        if wasm_bytes[:4] != b"\x00asm":
            return False, "Invalid WASM magic bytes"
        version = int.from_bytes(wasm_bytes[4:8], "little")
        if version != 1:
            return False, f"Unsupported WASM version: {version}"

        # Attempt a structural parse with wasmtime (safe, no JIT).
        try:
            import wasmtime

            engine = wasmtime.Engine()
            wasmtime.Module(engine, wasm_bytes)
        except Exception as exc:
            return False, f"WASM structural validation failed: {exc}"

        return True, ""

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @dataclass
    class _MemoryCheck:
        ok: bool
        reason: str = ""

    def _check_memory_limit(self, wasm_bytes: bytes) -> _MemoryCheck:
        """Ensure the module's declared initial memory fits within the limit."""
        import wasmtime

        engine = wasmtime.Engine()
        module = wasmtime.Module(engine, wasm_bytes)

        limit_bytes = self._memory_mb * 1024 * 1024

        # Check exported memories.
        for exp in module.exports:
            if isinstance(exp.type, wasmtime.MemoryType):
                initial_bytes = exp.type.limits.min * _WASM_PAGE_SIZE
                if initial_bytes > limit_bytes:
                    return self._MemoryCheck(
                        ok=False,
                        reason=(
                            f"Exported memory initial size {initial_bytes} bytes "
                            f"exceeds limit {limit_bytes} bytes"
                        ),
                    )

        # Check imported memories.
        for imp in module.imports:
            if isinstance(imp.type, wasmtime.MemoryType):
                initial_bytes = imp.type.limits.min * _WASM_PAGE_SIZE
                if initial_bytes > limit_bytes:
                    return self._MemoryCheck(
                        ok=False,
                        reason=(
                            f"Imported memory initial size {initial_bytes} bytes "
                            f"exceeds limit {limit_bytes} bytes"
                        ),
                    )

        return self._MemoryCheck(ok=True)

    def _needs_wasi(self, wasm_bytes: bytes) -> bool:
        """Return True if the module imports from wasi_snapshot_preview1."""
        import wasmtime

        engine = wasmtime.Engine()
        module = wasmtime.Module(engine, wasm_bytes)
        return any(
            imp.module == "wasi_snapshot_preview1" for imp in module.imports
        )

    def _run_nodejs(
        self,
        wasm_bytes: bytes,
        input_data: dict | None,
        allowed_host_funcs: list[str] | None,
    ) -> SandboxResult:
        """Run the WASM module inside a locked-down Node.js subprocess."""
        assert self._node is not None

        use_wasi = self._needs_wasi(wasm_bytes)

        with tempfile.TemporaryDirectory(prefix="orchestra_wasm_") as tmpdir:
            wasm_path = os.path.join(tmpdir, "module.wasm")
            with open(wasm_path, "wb") as f:
                f.write(wasm_bytes)

            script_path = os.path.join(tmpdir, "runner.js")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(_NODE_WRAPPER)

            env = os.environ.copy()
            env["WASM_PATH"] = wasm_path
            env["WASM_INPUT"] = json.dumps(input_data or {})
            env["WASM_ALLOWED_FUNCS"] = json.dumps(allowed_host_funcs or [])
            env["WASM_USE_WASI"] = "1" if use_wasi else "0"

            cmd = self._build_cmd(script_path)

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    env=env,
                    cwd=tmpdir,
                )
            except subprocess.TimeoutExpired:
                # subprocess.run has already killed the process on timeout.
                return SandboxResult(
                    success=False,
                    output="",
                    error=f"WASM sandbox timeout after {self._timeout}s",
                    exit_code=-1,
                    timed_out=True,
                )

            # If the process was killed by a signal (e.g. SIGKILL from OOM),
            # returncode is negative.
            if proc.returncode < 0:
                sig_name = self._signal_name(-proc.returncode)
                err_msg = f"WASM process terminated by signal {sig_name}"
                if -proc.returncode == 9:
                    err_msg += " (possible memory limit enforcement)"
                return SandboxResult(
                    success=False,
                    output=proc.stdout.strip(),
                    error=err_msg,
                    exit_code=proc.returncode,
                )

            # Try to parse the JSON result from stdout.
            result = self._parse_node_output(proc.stdout, proc.stderr)
            if result is not None:
                return result

            # Fallback for non-JSON output.
            if proc.returncode == 0:
                return SandboxResult(
                    success=True,
                    output=proc.stdout.strip(),
                    exit_code=0,
                )
            return SandboxResult(
                success=False,
                output=proc.stdout.strip(),
                error=proc.stderr.strip() or f"exit code {proc.returncode}",
                exit_code=proc.returncode,
            )

    def _build_cmd(self, script_path: str) -> list[str]:
        """Build the subprocess command, applying best-effort memory limits."""
        node = self._node
        assert node is not None

        if sys.platform == "win32":
            return [node, script_path]

        # On POSIX wrap with ulimit for memory cap (best-effort on macOS).
        memory_kb = self._memory_mb * 1024
        shell_cmd = (
            f"ulimit -v {memory_kb} 2>/dev/null; "
            f"exec {node} {script_path}"
        )
        return ["/bin/sh", "-c", shell_cmd]

    @staticmethod
    def _parse_node_output(stdout: str, stderr: str) -> SandboxResult | None:
        """Attempt to extract a JSON-encoded result from Node.js stdout."""
        # Node may print experimental warnings on stderr; ignore them.
        # Look for a JSON object on its own line.
        for line in stdout.strip().splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    data = json.loads(line)
                    success = data.get("success", False)
                    output = data.get("output", "")
                    error = data.get("error", "")
                    return SandboxResult(
                        success=success,
                        output=output,
                        error=error if not success else None,
                        exit_code=0 if success else 1,
                    )
                except json.JSONDecodeError:
                    continue
        return None

    @staticmethod
    def _signal_name(signum: int) -> str:
        """Return a human-readable signal name, or the raw number."""
        import signal

        for name, num in vars(signal).items():
            if num == signum and name.startswith("SIG") and not name.startswith("SIG_"):
                return name
        return str(signum)

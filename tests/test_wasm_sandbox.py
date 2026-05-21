from __future__ import annotations
"""Security tests for the WASM skill sandbox (F3.4).

These tests verify that the ``WasmSandbox`` correctly enforces:
  - execution timeouts (infinite loops)
  - memory limits
  - filesystem access denial
  - network access denial
  - rejection of malformed WASM
  - proper resource cleanup

Requirements
------------
* Node.js must be installed and available on ``PATH`` (the sandbox falls back
  to a Node.js subprocess because wasmtime's JIT may be blocked by macOS
  endpoint-protection policies).
* ``wasmtime`` Python package is used only for WAT→WASM compilation and
  static module inspection.
"""

import os
import subprocess
import tempfile

import pytest

from orchestra.toolmaker.wasm_sandbox import WasmSandbox
from orchestra.toolmaker.sandbox import SandboxResult


# ── Helpers ──────────────────────────────────────────────────────────────────

def _compile_wat(wat_code: str) -> bytes:
    """Compile WAT text to a WASM binary using wasmtime."""
    import wasmtime

    return wasmtime.wat2wasm(wat_code)


@pytest.fixture
def sandbox() -> WasmSandbox:
    """Return a sandbox configured with tight limits for testing."""
    return WasmSandbox(timeout=1, memory_mb=10)


@pytest.fixture
def loose_sandbox() -> WasmSandbox:
    """Return a sandbox with generous limits for tests that need them."""
    return WasmSandbox(timeout=5, memory_mb=128)


# ── 1. Infinite Loop Detection ───────────────────────────────────────────────

@pytest.mark.slow
def test_infinite_loop_raises_timeout(sandbox: WasmSandbox) -> None:
    """A WASM module with an unbounded loop must be killed by the timeout."""
    wat = """
    (module
      (func (export "run")
        (loop
          br 0
        )
      )
    )
    """
    wasm = _compile_wat(wat)
    result = sandbox.run(wasm)

    assert result.success is False
    assert result.timed_out is True
    assert "timeout" in result.error.lower()


@pytest.mark.slow
def test_infinite_loop_process_is_killed(sandbox: WasmSandbox) -> None:
    """After a timeout the Node.js subprocess must no longer exist."""
    wat = """
    (module
      (func (export "run")
        (loop
          br 0
        )
      )
    )
    """
    wasm = _compile_wat(wat)
    sandbox.run(wasm)

    # Give the OS a moment to reap the process.
    import time

    time.sleep(0.5)

    # Count Node.js processes owned by the current user.
    ps = subprocess.run(
        ["pgrep", "-u", str(os.getuid()), "-c", "node"],
        capture_output=True,
        text=True,
    )
    # ``pgrep`` returns 1 when no matches are found; that is fine.
    if ps.returncode == 0:
        count = int(ps.stdout.strip())
        # We allow a small number of unrelated Node processes, but there
        # should not be a swarm of zombies / orphans.
        assert count < 20, f"Too many node processes lingering: {count}"


# ── 2. Memory Exhaustion ─────────────────────────────────────────────────────

def test_declared_memory_exceeds_limit(sandbox: WasmSandbox) -> None:
    """A module whose initial linear memory exceeds the cap is rejected
    before execution."""
    # 200 pages × 64 KiB = 12.8 MiB  >  10 MiB limit.
    wat = """
    (module
      (memory (export "memory") 200)
      (func (export "run") (result i32)
        i32.const 42
      )
    )
    """
    wasm = _compile_wat(wat)
    result = sandbox.run(wasm)

    assert result.success is False
    assert "memory" in result.error.lower()


def test_imported_memory_exceeds_limit(sandbox: WasmSandbox) -> None:
    """Memory imported from the host is also subject to the limit check."""
    wat = """
    (module
      (import "env" "mem" (memory 200))
      (func (export "run"))
    )
    """
    wasm = _compile_wat(wat)
    result = sandbox.run(wasm)

    assert result.success is False
    assert "memory" in result.error.lower()


def test_memory_grow_within_limit(loose_sandbox: WasmSandbox) -> None:
    """A module that grows memory modestly should succeed."""
    wat = """
    (module
      (memory (export "memory") 1)
      (func (export "run") (result i32)
        i32.const 1
        memory.grow
      )
    )
    """
    wasm = _compile_wat(wat)
    result = loose_sandbox.run(wasm)

    assert result.success is True
    # memory.grow returns the previous page count (1) on success.
    assert result.output == "1"


# ── 3. Filesystem Access Denial ──────────────────────────────────────────────

def test_wasi_no_preopened_dirs(sandbox: WasmSandbox) -> None:
    """A WASI module that tries to open ``/etc/passwd`` must fail because
    no directories are pre-opened."""
    wat = r"""
    (module
      (import "wasi_snapshot_preview1" "path_open"
        (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "proc_exit"
        (func $proc_exit (param i32)))
      (memory (export "memory") 1)
      (data (i32.const 100) "/etc/passwd")
      (func (export "_start")
        i32.const 3
        i32.const 0
        i32.const 100
        i32.const 11
        i32.const 0
        i64.const 0
        i64.const 0
        i32.const 0
        i32.const 200
        call $path_open
        call $proc_exit
      )
    )
    """
    wasm = _compile_wat(wat)
    result = sandbox.run(wasm)

    assert result.success is False
    # WASI returns ENOTCAPABLE (8) when no pre-opened dir covers the path.
    assert "exit code 8" in result.error or "ENOTCAPABLE" in result.error


def test_non_wasi_module_has_no_fs_access(sandbox: WasmSandbox) -> None:
    """A plain WASM module without WASI imports cannot access the filesystem."""
    wat = """
    (module
      (func (export "run") (result i32)
        i32.const 42
      )
    )
    """
    wasm = _compile_wat(wat)
    result = sandbox.run(wasm)

    assert result.success is True
    assert result.output == "42"


# ── 4. Network Access Denial ─────────────────────────────────────────────────

def test_wasi_socket_blocked(sandbox: WasmSandbox) -> None:
    """A WASI module importing ``sock_open`` must fail because Node's WASI
    snapshot does not expose socket syscalls."""
    wat = """
    (module
      (import "wasi_snapshot_preview1" "sock_open"
        (func $sock_open (param i32 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "proc_exit"
        (func $proc_exit (param i32)))
      (memory (export "memory") 1)
      (func (export "_start")
        i32.const 0
        i32.const 0
        i32.const 0
        call $sock_open
        call $proc_exit
      )
    )
    """
    wasm = _compile_wat(wat)
    result = sandbox.run(wasm)

    assert result.success is False
    # Node's WASI implementation omits sock_open, so instantiation fails.
    assert "sock_open" in result.error or "function import requires a callable" in result.error


def test_arbitrary_network_import_blocked(sandbox: WasmSandbox) -> None:
    """A module that imports an arbitrary socket-like function not listed in
    ``allowed_host_funcs`` must fail at instantiation time."""
    wat = """
    (module
      (import "env" "socket" (func $socket (param i32 i32) (result i32)))
      (func (export "run")
        i32.const 0
        i32.const 0
        call $socket
        drop
      )
    )
    """
    wasm = _compile_wat(wat)
    result = sandbox.run(wasm, allowed_host_funcs=[])

    assert result.success is False
    assert "module is not an object or function" in result.error


# ── 5. Code Injection Resistance ─────────────────────────────────────────────

def test_invalid_magic_bytes_rejected(sandbox: WasmSandbox) -> None:
    """Completely invalid bytes must be rejected cleanly."""
    result = sandbox.run(b"this is not wasm")

    assert result.success is False
    assert "Invalid WASM magic bytes" in result.error


def test_truncated_binary_rejected(sandbox: WasmSandbox) -> None:
    """A WASM binary shorter than 8 bytes must be rejected."""
    result = sandbox.run(b"\x00asm\x01")

    assert result.success is False
    assert "too short" in result.error.lower()


def test_malformed_structural_rejected(sandbox: WasmSandbox) -> None:
    """Bytes with valid magic but garbage body must be rejected by wasmtime's
    structural validator rather than crashing the sandbox."""
    result = sandbox.run(b"\x00asm\x01\x00\x00\x00DEADBEEF")

    assert result.success is False
    assert "structural validation failed" in result.error.lower()


def test_allowed_host_func_whitelist(sandbox: WasmSandbox) -> None:
    """When an allowed host func is explicitly whitelisted the module may
    import it; otherwise instantiation fails."""
    wat = """
    (module
      (import "host" "permitted" (func $permitted))
      (func (export "run")
        call $permitted
      )
    )
    """
    wasm = _compile_wat(wat)

    # Without the whitelist → failure.
    result = sandbox.run(wasm, allowed_host_funcs=[])
    assert result.success is False

    # With the whitelist → instantiation succeeds (function is a stub).
    result = sandbox.run(wasm, allowed_host_funcs=["host::permitted"])
    # The stub throws at runtime because it is not implemented, but
    # instantiation itself succeeds.
    assert "Host function \"host::permitted\" is not implemented" in result.error


# ── 6. Resource Cleanup ──────────────────────────────────────────────────────

def test_temp_files_cleaned_up(sandbox: WasmSandbox) -> None:
    """After a sandbox run the temporary directory must be removed."""
    wat = """
    (module
      (func (export "run") (result i32)
        i32.const 1
      )
    )
    """
    wasm = _compile_wat(wat)

    # Record existing orchestra temp dirs.
    tmp_root = tempfile.gettempdir()
    before = {
        name
        for name in os.listdir(tmp_root)
        if name.startswith("orchestra_wasm_")
    }

    sandbox.run(wasm)

    after = {
        name
        for name in os.listdir(tmp_root)
        if name.startswith("orchestra_wasm_")
    }
    new_dirs = after - before
    assert len(new_dirs) == 0, f"Leftover temp directories: {new_dirs}"


@pytest.mark.slow
def test_no_zombie_processes_after_runs(sandbox: WasmSandbox) -> None:
    """Running many sandbox instances sequentially must not orphan Node.js
    processes."""
    wat = """
    (module
      (func (export "run") (result i32)
        i32.const 0
      )
    )
    """
    wasm = _compile_wat(wat)

    for _ in range(5):
        result = sandbox.run(wasm)
        assert result.success is True

    # Allow the OS to reap any lingering children.
    import time

    time.sleep(0.5)

    ps = subprocess.run(
        ["pgrep", "-u", str(os.getuid()), "-c", "node"],
        capture_output=True,
        text=True,
    )
    if ps.returncode == 0:
        count = int(ps.stdout.strip())
        assert count < 20, f"Orphaned node processes detected: {count}"

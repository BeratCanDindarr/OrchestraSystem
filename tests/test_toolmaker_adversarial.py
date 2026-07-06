"""Adversarial regression tests for ToolMaker AST validator — C-5 Phase 1.

Each test maps to a confirmed bypass. Tests marked "blocked" must return
``not result.ok``; tests marked "allowed" must return ``result.ok``.
"""
from __future__ import annotations

import pytest

from orchestra.toolmaker.validator import ToolValidator, ValidationResult


def _v(script: str) -> ValidationResult:
    return ToolValidator().validate(script)


# ===========================================================================
# Change 1: exec-style calls must be blocked by call name (alias-robust)
# ===========================================================================

class TestExecStyleCallsBlocked:
    """os.system / popen / exec* / spawn* / fork bypasses."""

    def test_os_system_blocked(self):
        result = _v("import os\nos.system('id')\n")
        assert not result.ok, f"os.system should be blocked, errors={result.errors}"

    def test_os_popen_alias_blocked(self):
        result = _v("import os as o\no.popen('id')\n")
        assert not result.ok, f"o.popen (aliased) should be blocked, errors={result.errors}"

    def test_from_os_import_system_blocked(self):
        result = _v("from os import system\n")
        assert not result.ok, f"from os import system should be blocked, errors={result.errors}"

    def test_bare_system_call_blocked(self):
        # Even without an explicit import, a call named system is blocked
        result = _v("system('x')\n")
        assert not result.ok, f"bare system() should be blocked, errors={result.errors}"

    def test_os_fork_blocked(self):
        result = _v("import os\nos.fork()\n")
        assert not result.ok, f"os.fork should be blocked, errors={result.errors}"

    def test_os_execv_blocked(self):
        result = _v("import os\nos.execv('/bin/sh', ['sh'])\n")
        assert not result.ok, f"os.execv should be blocked, errors={result.errors}"

    def test_os_execve_blocked(self):
        result = _v("import os\nos.execve('/bin/sh', ['sh'], {})\n")
        assert not result.ok, f"os.execve should be blocked, errors={result.errors}"

    def test_os_spawnl_blocked(self):
        result = _v("import os\nos.spawnl(os.P_WAIT, '/bin/ls', 'ls')\n")
        assert not result.ok, f"os.spawnl should be blocked, errors={result.errors}"

    def test_os_startfile_blocked(self):
        result = _v("import os\nos.startfile('evil.exe')\n")
        assert not result.ok, f"os.startfile should be blocked, errors={result.errors}"

    def test_from_os_import_popen_blocked(self):
        result = _v("from os import popen\n")
        assert not result.ok, f"from os import popen should be blocked, errors={result.errors}"

    def test_from_os_import_fork_blocked(self):
        result = _v("from os import fork\n")
        assert not result.ok, f"from os import fork should be blocked, errors={result.errors}"


# ===========================================================================
# Change 2: open() with non-constant mode must be blocked
# ===========================================================================

class TestOpenNonConstantModeBlocked:
    """open() mode bypass via variable/expression."""

    def test_open_variable_mode_blocked(self):
        result = _v("mode = 'w'\nopen('f', mode)\n")
        assert not result.ok, f"open() with variable mode should be blocked, errors={result.errors}"

    def test_open_expression_mode_blocked(self):
        result = _v("open('f', get_mode())\n")
        assert not result.ok, f"open() with computed mode should be blocked, errors={result.errors}"

    def test_open_constant_write_mode_still_blocked(self):
        # Existing behavior must be preserved
        result = _v("open('f', 'w')\n")
        assert not result.ok, f"open('f', 'w') must still be blocked, errors={result.errors}"

    def test_open_constant_append_mode_still_blocked(self):
        result = _v("open('f', 'a')\n")
        assert not result.ok, f"open('f', 'a') must still be blocked, errors={result.errors}"

    def test_open_no_mode_allowed(self):
        result = _v("open('f')\n")
        assert result.ok, f"open('f') should be allowed, errors={result.errors}"

    def test_open_read_mode_allowed(self):
        result = _v("open('f', 'r')\n")
        assert result.ok, f"open('f', 'r') should be allowed, errors={result.errors}"

    def test_open_keyword_variable_mode_blocked(self):
        result = _v("mode = 'w'\nopen('f', mode=mode)\n")
        assert not result.ok, f"open() with keyword variable mode should be blocked, errors={result.errors}"

    def test_open_keyword_write_mode_blocked(self):
        result = _v("open('f', mode='w')\n")
        assert not result.ok, f"open() with keyword mode='w' should be blocked, errors={result.errors}"

    def test_open_keyword_read_mode_allowed(self):
        result = _v("open('f', mode='r')\n")
        assert result.ok, f"open('f', mode='r') should be allowed, errors={result.errors}"


# ===========================================================================
# Change 3: getattr/setattr/delattr reflection bypasses must be blocked
# ===========================================================================

class TestReflectionBypassBlocked:
    """getattr/setattr/delattr with dunder or non-constant attribute name."""

    def test_getattr_dunder_globals_blocked(self):
        result = _v("x = object()\ngetattr(x, '__globals__')\n")
        assert not result.ok, f"getattr dunder should be blocked, errors={result.errors}"

    def test_getattr_dunder_class_blocked(self):
        result = _v("getattr(x, '__class__')\n")
        assert not result.ok, f"getattr __class__ should be blocked, errors={result.errors}"

    def test_getattr_variable_attr_blocked(self):
        # Non-constant — value can't be verified at AST time
        result = _v("dunder = '__globals__'\ngetattr(x, dunder)\n")
        assert not result.ok, f"getattr with variable attr should be blocked, errors={result.errors}"

    def test_getattr_constant_benign_allowed(self):
        # Constant non-dunder attr name — benign reflection should stay usable
        result = _v("getattr(x, 'value')\n")
        assert result.ok, f"getattr with constant benign attr should be allowed, errors={result.errors}"

    def test_setattr_dunder_blocked(self):
        result = _v("setattr(x, '__dict__', {})\n")
        assert not result.ok, f"setattr dunder should be blocked, errors={result.errors}"

    def test_setattr_variable_attr_blocked(self):
        result = _v("attr = '__dict__'\nsetattr(x, attr, {})\n")
        assert not result.ok, f"setattr with variable attr should be blocked, errors={result.errors}"

    def test_setattr_constant_benign_allowed(self):
        result = _v("setattr(x, 'value', 42)\n")
        assert result.ok, f"setattr with constant benign attr should be allowed, errors={result.errors}"

    def test_delattr_dunder_blocked(self):
        result = _v("delattr(x, '__class__')\n")
        assert not result.ok, f"delattr dunder should be blocked, errors={result.errors}"

    def test_delattr_variable_attr_blocked(self):
        result = _v("attr = 'x'\ndelattr(obj, attr)\n")
        assert not result.ok, f"delattr with variable attr should be blocked, errors={result.errors}"


# ===========================================================================
# Positive test: legitimate tool must NOT be over-blocked
# ===========================================================================

class TestNoOverBlocking:
    """A valid tool using os/json/re safely must still pass validation."""

    def test_legitimate_tool_passes(self):
        # Note: re.compile() is intentionally omitted — the existing validator
        # blocks any call whose attribute/function name is "compile" (it targets
        # the built-in compile()), which also catches re.compile(). That is a
        # pre-existing limitation noted as a Phase-2 gap (out of scope here).
        script = """\
import os
import json
import re

TOOL_SCHEMA = {
    "name": "read_file",
    "description": "Reads a file and returns its content as JSON",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"]
    }
}


def run(path: str) -> dict:
    base_dir = os.getenv("DATA_DIR", "/data")
    full_path = os.path.join(base_dir, path)
    name = os.path.basename(full_path)

    with open(full_path) as f:
        raw = f.read()

    with open(full_path, "r") as f:
        raw2 = f.read()

    matches = re.findall(r"\\d+", raw)
    data = json.loads(raw)
    return {"name": name, "data": data, "matches": matches}
"""
        result = _v(script)
        assert result.ok, (
            "A legitimate tool using os/json/re should NOT be blocked, "
            f"but got errors: {result.errors}"
        )

    def test_import_os_still_allowed(self):
        """Regression guard: import os must never be blocked (test_allows_stdlib_imports)."""
        result = _v("import os\nimport json\nimport re\n")
        assert result.ok, f"import os/json/re must be allowed, errors={result.errors}"

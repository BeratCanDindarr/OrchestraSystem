"""ToolMaker Validator — AST-based safety check for dynamically generated tool scripts.

Checks (in order):
  1. AST parse — script must be valid Python
  2. Blocked imports — deny-listed modules that could escape sandbox
  3. Blocked builtins — __import__, eval, exec, compile, open (write mode)
  4. MCP schema — if script defines a TOOL_SCHEMA dict, validate required fields

Usage:
    from orchestra.toolmaker.validator import ToolValidator
    result = ToolValidator().validate(script_source)
    if not result.ok:
        print(result.errors)
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

# Modules that must never be imported in tool scripts
_BLOCKED_IMPORTS: frozenset[str] = frozenset(
    {
        # Network
        "socket", "http", "urllib", "urllib2", "urllib3", "requests",
        "httpx", "aiohttp", "ftplib", "smtplib", "poplib", "imaplib",
        "xmlrpc",
        # Process / OS escape
        "subprocess", "multiprocessing", "pty", "signal",
        # Serialization risks
        "pickle", "shelve", "marshal",
        # Dynamic code execution
        "importlib", "imp", "pkgutil",
        # System internals
        "ctypes", "cffi", "mmap",
        # Crypto / key material
        "cryptography", "ssl",
    }
)

# AST call names that are blocked regardless of context
_BLOCKED_CALL_NAMES: frozenset[str] = frozenset(
    {
        "__import__",
        "eval",
        "exec",
        "compile",
        "globals",
        "locals",
        "vars",
        "breakpoint",
        "input",
    }
)

# open() write/append modes that are blocked
_BLOCKED_OPEN_MODES: frozenset[str] = frozenset({"w", "wb", "a", "ab", "x", "xb"})

# Required keys in TOOL_SCHEMA dict (MCP schema validation)
_REQUIRED_SCHEMA_KEYS: frozenset[str] = frozenset({"name", "description", "parameters"})


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)

    def add(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False


class _ASTVisitor(ast.NodeVisitor):
    """Walk AST and collect violations."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    # ── Import checks ──────────────────────────────────────────────────────

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in _BLOCKED_IMPORTS or alias.name in _BLOCKED_IMPORTS:
                self.violations.append(f"Blocked import: {alias.name!r} (line {node.lineno})")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        top = module.split(".")[0]
        if top in _BLOCKED_IMPORTS or module in _BLOCKED_IMPORTS:
            self.violations.append(f"Blocked import: {module!r} (line {node.lineno})")
        self.generic_visit(node)

    # ── Blocked call checks ────────────────────────────────────────────────

    def visit_Call(self, node: ast.Call) -> None:
        name = self._call_name(node)
        if name in _BLOCKED_CALL_NAMES:
            self.violations.append(f"Blocked call: {name!r} (line {node.lineno})")
        # Check open() write modes when mode is a literal
        if name == "open" and len(node.args) >= 2:
            mode_arg = node.args[1]
            if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
                if mode_arg.value in _BLOCKED_OPEN_MODES:
                    self.violations.append(
                        f"Blocked open() write mode {mode_arg.value!r} (line {node.lineno})"
                    )
        self.generic_visit(node)

    @staticmethod
    def _call_name(node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    # ── Dangerous attribute access ─────────────────────────────────────────

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in (
            "__class__", "__bases__", "__subclasses__",
            "__globals__", "__builtins__", "__code__",
        ):
            self.violations.append(
                f"Blocked attribute access: {node.attr!r} (line {node.lineno})"
            )
        self.generic_visit(node)


def _extract_tool_schema(tree: ast.Module) -> dict | None:
    """Extract TOOL_SCHEMA dict literal from top-level assignments, if present."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "TOOL_SCHEMA":
                if isinstance(node.value, ast.Dict):
                    keys = {
                        k.value if isinstance(k, ast.Constant) else None
                        for k in node.value.keys
                    }
                    return {k: True for k in keys if k}
    return None


class ToolValidator:
    """Validate a tool script via AST analysis before sandbox execution."""

    def validate(self, script: str) -> ValidationResult:
        result = ValidationResult(ok=True)

        # Step 1: parse
        try:
            tree = ast.parse(script)
        except SyntaxError as exc:
            result.add(f"Syntax error: {exc}")
            return result

        # Step 2: AST walk for import/call/attribute violations
        visitor = _ASTVisitor()
        visitor.visit(tree)
        for v in visitor.violations:
            result.add(v)

        # Step 3: MCP schema validation (optional — only if TOOL_SCHEMA defined)
        schema = _extract_tool_schema(tree)
        if schema is not None:
            missing = _REQUIRED_SCHEMA_KEYS - set(schema.keys())
            for key in sorted(missing):
                result.add(f"TOOL_SCHEMA missing required key: {key!r}")

        return result

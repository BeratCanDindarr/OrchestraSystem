"""Shared utilities for parsing and serialising CLI / TOML config values.

These are the canonical, superset implementations unified from the previously
diverged copies in orchestra/cli.py and orchestra/service.py.  The public API
uses names without a leading underscore; both callers import them under aliased
private names so existing call-sites require no changes.
"""
from __future__ import annotations

import re


def parse_cli_value(value):
    """Convert a raw CLI value (string or already-parsed type) to a Python scalar.

    Handles:
    - Non-str pass-through: bool, int, float, list, dict are returned as-is.
    - Scalar strings: "true"/"false" → bool, integer strings → int,
      float strings → float, everything else → str.
    - Bracketed list strings: "[a, b, c]" → list, recursing on each element.
    """
    if isinstance(value, (bool, int, float, list, dict)):
        return value
    if not isinstance(value, str):
        return str(value)

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_cli_value(part.strip()) for part in inner.split(",")]
    return value


def toml_literal(value) -> str:
    """Serialise a Python value to a TOML-compatible literal string.

    Handles bool, int/float, list (recursively), dict (via JSON), and str.
    dict support is the intentional improvement over the old cli.py version,
    which would have fallen through to a quoted str(value).
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{toml_key(key)} = {toml_literal(item)}" for key, item in value.items()) + " }"
    return toml_string(str(value))


_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def toml_string(text: str) -> str:
    """A TOML basic string: quotes, backslashes and control characters escaped."""
    escaped = "".join(
        _ESCAPES.get(char) or (f"\\u{ord(char):04x}" if ord(char) < 0x20 or ord(char) == 0x7F else char)
        for char in text
    )
    return f'"{escaped}"'


def toml_key(key) -> str:
    """A key as TOML reads it: bare when it can be, quoted otherwise.

    `gpt-5.4/low` unquoted is a dotted key (and `a]\n[b` a new section): a key is data.
    """
    key = str(key)
    return key if _BARE_KEY.fullmatch(key) else toml_string(key)


def flatten_toml_lines(data: dict, prefix: list[str] | None = None) -> list[str]:
    """Flatten a nested dict into a list of TOML-formatted lines.

    Scalar keys are emitted first under the current section header, then
    nested dicts follow as sub-sections, separated by blank lines.
    """
    prefix = prefix or []
    scalar_items: list[tuple[str, object]] = []
    nested_items: list[tuple[str, dict]] = []

    for key, value in data.items():
        if isinstance(value, dict):
            nested_items.append((key, value))
        else:
            scalar_items.append((key, value))

    lines: list[str] = []
    if prefix:
        lines.append(f"[{'.'.join(toml_key(part) for part in prefix)}]")
    for key, value in scalar_items:
        lines.append(f"{toml_key(key)} = {toml_literal(value)}")
    if prefix and nested_items:
        lines.append("")
    for index, (key, value) in enumerate(nested_items):
        lines.extend(flatten_toml_lines(value, prefix + [key]))
        if index != len(nested_items) - 1:
            lines.append("")
    return lines

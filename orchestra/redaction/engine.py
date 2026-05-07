"""Apply repository-specific redaction rules to Orchestra artifacts."""
from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore


def _rules_path() -> Path:
    return Path(__file__).with_name("rules.toml")


def _load_rules() -> list[dict]:
    path = _rules_path()
    if not path.exists():
        return []

    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return []

    rules = data.get("rules", [])
    return [rule for rule in rules if isinstance(rule, dict)]


def redact(text: str) -> str:
    """Redact secrets in text. Missing or invalid rules are ignored."""
    redacted = text
    for rule in _load_rules():
        pattern = rule.get("pattern")
        replacement = rule.get("replacement", "[REDACTED]")
        if not pattern:
            continue
        try:
            redacted = re.sub(pattern, replacement, redacted, flags=re.MULTILINE)
        except re.error:
            continue
    return redacted


def redact_file(path: Path) -> None:
    """Redact a text file in place. Missing files or rules are ignored."""
    rules = _load_rules()
    if not rules or not path.exists():
        return

    try:
        original = path.read_text(encoding="utf-8")
    except OSError:
        return

    redacted = original
    for rule in rules:
        pattern = rule.get("pattern")
        replacement = rule.get("replacement", "[REDACTED]")
        if not pattern:
            continue
        try:
            redacted = re.sub(pattern, replacement, redacted, flags=re.MULTILINE)
        except re.error:
            continue

    if redacted == original:
        return

    try:
        path.write_text(redacted, encoding="utf-8")
    except OSError:
        return

"""Pure helpers for the ``orchestra statusline install`` command.

These functions contain all the merge logic so they are unit-testable without
Typer.  The CLI layer in ``orchestra/cli.py`` handles I/O and user messages.

Immutability contract: ``merge_settings`` returns a *new* dict — it never
mutates its input.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MergeResult:
    """Outcome of a merge_settings call."""

    merged: dict
    """The new settings dict to write (always a fresh dict)."""

    was_overwritten: bool = False
    """True when an existing ``statusLine`` key was replaced (--force case)."""

    conflict: Optional[dict] = None
    """Existing ``statusLine`` value when refusing without --force."""


# ---------------------------------------------------------------------------
# Block builder
# ---------------------------------------------------------------------------

def build_statusline_block(interpreter: str, interval: int) -> dict:
    """Return the statusLine config dict.

    Args:
        interpreter: Absolute path to the Python interpreter (``sys.executable``).
        interval: Refresh interval in seconds (must be >= 1).

    Returns:
        A plain dict suitable for JSON serialisation.
    """
    if interval < 1:
        raise ValueError(f"interval must be >= 1, got {interval}")
    quoted = shlex.quote(interpreter)
    return {
        "type": "command",
        "command": f"{quoted} -m orchestra.statusline",
        "refreshInterval": interval,
    }


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def merge_settings(
    existing: Optional[dict],
    block: dict,
    force: bool = False,
) -> MergeResult:
    """Merge a statusLine block into an existing settings dict.

    Merge semantics:
    1. ``existing`` is ``None`` → return a new dict with just the block.
    2. ``existing`` has no ``statusLine`` key → return a new dict with all
       existing keys plus ``statusLine``.
    3. ``existing`` has a ``statusLine`` key and ``force`` is ``False`` →
       raise ``ConflictError`` (caller should use the ``conflict`` value to
       show the diff and exit 1).
    4. ``existing`` has a ``statusLine`` key and ``force`` is ``True`` →
       return a new dict with ``statusLine`` replaced.

    Args:
        existing: Current settings dict, or ``None`` when the file is absent.
        block: The statusLine value dict to insert/replace.
        force: When ``True``, overwrite an existing ``statusLine`` key.

    Returns:
        A :class:`MergeResult` with the merged dict.

    Raises:
        ConflictError: When ``statusLine`` exists and ``force`` is ``False``.
    """
    if existing is None:
        return MergeResult(merged={"statusLine": block})

    if "statusLine" not in existing:
        merged = {**existing, "statusLine": block}
        return MergeResult(merged=merged)

    # Conflict path
    current_value = existing["statusLine"]
    if not force:
        raise ConflictError(current_value)

    # Force overwrite — build new dict preserving all other keys
    merged = {k: v for k, v in existing.items() if k != "statusLine"}
    merged["statusLine"] = block
    return MergeResult(merged=merged, was_overwritten=True, conflict=current_value)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class ConflictError(Exception):
    """Raised when a statusLine key already exists and --force was not given."""

    def __init__(self, existing_value: dict) -> None:
        super().__init__("statusLine already exists; use --force to overwrite")
        self.existing_value = existing_value


# ---------------------------------------------------------------------------
# CLI helpers (pure; no Typer imports)
# ---------------------------------------------------------------------------

class InvalidSettingsError(Exception):
    """Raised when the existing settings file cannot be parsed as a JSON object."""


def resolve_target(
    path: Optional[str],
    user: bool,
) -> "Path":
    """Return the settings.json Path given --path / --user / default logic.

    Raises:
        ValueError: When both *path* and *user* are provided.
    """
    from pathlib import Path as _Path

    if path is not None and user:
        raise ValueError("--path and --user are mutually exclusive")
    if user:
        return _Path.home() / ".claude" / "settings.json"
    if path is not None:
        return _Path(path)
    return _Path.cwd() / ".claude" / "settings.json"


def load_existing_settings(target: "Path") -> Optional[dict]:
    """Load and validate an existing settings file.

    Returns:
        The parsed dict, or ``None`` when the file does not exist.

    Raises:
        InvalidSettingsError: When the file exists but is not a valid JSON object.
    """
    import json

    if not target.exists():
        return None
    raw = target.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidSettingsError(
            f"{target} contains invalid JSON — refusing to touch it"
        ) from exc
    if not isinstance(parsed, dict):
        raise InvalidSettingsError(
            f"{target} does not contain a JSON object — refusing to touch it"
        )
    return parsed

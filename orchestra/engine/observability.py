"""Observability utilities for reading and formatting Orchestra run data.

Phase 1: Core utility functions for loading run metadata, events, and agent logs.
These functions serve as the foundation for timeline, diff, and replay subcommands.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from orchestra import config
from orchestra.engine import artifacts
from orchestra.models import OrchestraRun

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Formatting Helpers
# ────────────────────────────────────────────────────────────────────────────


def format_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration string.

    Examples:
        - 0.001 → "1ms"
        - 1.234 → "1.23s"
        - 65.5 → "1m5s"
        - 3661 → "1h1m"

    Args:
        seconds: Duration in seconds (float)

    Returns:
        Formatted string
    """
    if seconds < 0.001:
        return f"{int(seconds * 1e6)}µs"
    if seconds < 1:
        return f"{int(seconds * 1e3)}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m{secs}s"

    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h{mins}m"


def format_cost(usd: float) -> str:
    """Format USD amount as a human-readable cost string.

    Examples:
        - 0.0001 → "$0.0001"
        - 0.001 → "$0.001"
        - 1.5 → "$1.50"

    Args:
        usd: Cost in USD (float)

    Returns:
        Formatted string with $ prefix
    """
    if usd < 0.001:
        return f"${usd:.6f}".rstrip("0").rstrip(".")
    if usd < 1:
        return f"${usd:.4f}".rstrip("0").rstrip(".")
    return f"${usd:.2f}"


def parse_event_time(event_dict: dict) -> float:
    """Parse ISO 8601 timestamp from event dict into epoch seconds.

    Args:
        event_dict: Event dictionary with 'ts' field in ISO 8601 format

    Returns:
        Unix timestamp (float, seconds since epoch)

    Raises:
        ValueError: If 'ts' field is missing or invalid format
    """
    ts_str = event_dict.get("ts")
    if not ts_str:
        raise ValueError("Event missing 'ts' field")

    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid timestamp format: {ts_str}") from e


# ────────────────────────────────────────────────────────────────────────────
# Data Loading Functions
# ────────────────────────────────────────────────────────────────────────────


def resolve_run_id(run_id_prefix: str) -> Optional[str]:
    """Resolve 8-char or full run ID prefix to full run_id by searching .orchestra/runs/.

    If prefix matches multiple runs, returns None (ambiguous).
    If no runs match, returns None.

    Args:
        run_id_prefix: Full run_id or 8+ char prefix (e.g., "a1b2c3d4")

    Returns:
        Full run_id if unambiguous match found, None otherwise
    """
    root = config.artifact_root()
    if not root.exists():
        return None

    # Exact match check first
    if (root / run_id_prefix).is_dir():
        return run_id_prefix

    # Prefix search
    candidates = []
    try:
        for d in root.iterdir():
            if d.is_dir() and d.name.startswith(run_id_prefix):
                candidates.append(d.name)
    except OSError:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Ambiguous or no match
    return None


def load_manifest(run_id: str) -> Optional[OrchestraRun]:
    """Load manifest.json and deserialize into OrchestraRun model.

    Returns None if run not found or manifest is invalid JSON.

    Args:
        run_id: Full run_id (use resolve_run_id to resolve prefixes)

    Returns:
        OrchestraRun object or None if not found/invalid
    """
    manifest_data = artifacts.load_manifest(run_id)
    if not manifest_data:
        return None

    try:
        return OrchestraRun.from_manifest(manifest_data)
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"Failed to deserialize manifest for {run_id}: {e}")
        return None


def load_events(run_id: str) -> list[dict]:
    """Load and parse events.jsonl, returning list of events.

    Each event dict includes parsed 'ts' field (ISO 8601 string) and computed fields:
    - ts_sec: seconds since first event (relative timing)
    - ts_epoch: Unix timestamp (float)

    Gracefully handles missing files and invalid JSON lines (logs warnings, skips).

    Args:
        run_id: Full run_id

    Returns:
        List of event dicts, sorted by timestamp. Empty list if no events found.
    """
    events = artifacts.read_events(run_id)
    if not events:
        return []

    # Compute relative timestamps
    first_ts_epoch: Optional[float] = None
    enriched_events: list[dict] = []

    for event in events:
        try:
            ts_epoch = parse_event_time(event)
            if first_ts_epoch is None:
                first_ts_epoch = ts_epoch

            ts_sec = ts_epoch - first_ts_epoch
            event_copy = dict(event)
            event_copy["ts_epoch"] = ts_epoch
            event_copy["ts_sec"] = ts_sec
            enriched_events.append(event_copy)

        except ValueError as e:
            logger.warning(f"Skipping invalid event in {run_id}: {e}")
            continue

    return enriched_events


def load_agent_logs(run_id: str) -> dict[str, str]:
    """Load all agents/*.stdout.log files and aggregate by agent alias.

    Returns empty dict if run has no agents or directory is missing.
    Gracefully skips unreadable files.

    Args:
        run_id: Full run_id

    Returns:
        Dict mapping agent alias (str) to stdout log content (str).
        Example: {"cdx-deep": "output text...", "gmn-pro": "output text..."}
    """
    agents_dir = config.artifact_root() / run_id / "agents"
    if not agents_dir.exists():
        return {}

    logs: dict[str, str] = {}

    try:
        for log_file in agents_dir.glob("*.stdout.log"):
            alias = log_file.stem  # e.g., "cdx-deep" from "cdx-deep.stdout.log"
            try:
                content = log_file.read_text(encoding="utf-8")
                logs[alias] = content
            except (OSError, UnicodeDecodeError) as e:
                logger.warning(f"Failed to read {log_file}: {e}")
                continue

    except OSError as e:
        logger.warning(f"Failed to scan agents directory for {run_id}: {e}")

    return logs


# ────────────────────────────────────────────────────────────────────────────
# Timeline Event Enrichment
# ────────────────────────────────────────────────────────────────────────────


def enrich_timeline_events(events: list[dict]) -> list[dict]:
    """Enrich raw events with computed timeline fields for rendering.

    Adds these fields to each event:
    - event_type: Normalized event category (run|agent|phase|span|other)
    - hierarchy: Tuple (phase_id, step_id, agent_alias) for nesting, or None

    Args:
        events: List of event dicts from load_events()

    Returns:
        Enhanced event list with timeline-specific fields
    """
    enriched: list[dict] = []

    for event in events:
        e = dict(event)

        # Infer event_type from event field
        event_name = e.get("event", "")
        if event_name.startswith("run_"):
            e["event_type"] = "run"
        elif event_name.startswith("agent_"):
            e["event_type"] = "agent"
        elif event_name.startswith("phase_"):
            e["event_type"] = "phase"
        elif event_name.startswith("span_"):
            e["event_type"] = "span"
        else:
            e["event_type"] = "other"

        # Compute hierarchy for nesting (phase_id, step_id, agent_alias)
        phase_id = e.get("phase_id")
        step_id = e.get("step_id")
        alias = e.get("alias")
        if phase_id is not None or step_id is not None or alias:
            e["hierarchy"] = (phase_id, step_id, alias)
        else:
            e["hierarchy"] = None

        enriched.append(e)

    return enriched


# ────────────────────────────────────────────────────────────────────────────
# Run Comparison Helpers
# ────────────────────────────────────────────────────────────────────────────


def compare_field_values(value_a: any, value_b: any) -> str:
    """Compare two values and return a status indicator.

    Returns:
        - "✓" if values are equal
        - "A better" if A is greater (for numeric values)
        - "B better" if B is greater (for numeric values)
        - "different" if values differ but not comparable
    """
    if value_a == value_b:
        return "✓"

    # For numeric comparisons (cost, confidence, etc.)
    try:
        a_num = float(value_a) if value_a is not None else 0
        b_num = float(value_b) if value_b is not None else 0

        # Lower is better for cost, higher is better for confidence
        if isinstance(value_a, float) or isinstance(value_b, float):
            # Cost comparison: lower is better
            if "cost" in str(value_a).lower() or "usd" in str(value_a).lower():
                return "A cheaper" if a_num < b_num else "B cheaper"
            # Confidence comparison: higher is better
            elif "confid" in str(value_a).lower():
                return "A better" if a_num > b_num else "B better"

        return "different"
    except (TypeError, ValueError):
        return "different"


def agent_from_manifest_dict(agent_dict: dict) -> dict:
    """Normalize agent data from manifest dict to observability format.

    Args:
        agent_dict: Agent entry from manifest["agents"]

    Returns:
        Dict with keys: alias, provider, model, status, elapsed, cost_usd, confidence, error
    """
    return {
        "alias": agent_dict.get("alias", "unknown"),
        "provider": agent_dict.get("provider", "unknown"),
        "model": agent_dict.get("model", "unknown"),
        "status": agent_dict.get("status", "unknown"),
        "elapsed": agent_dict.get("elapsed", "--:--"),
        "cost_usd": agent_dict.get("estimated_cost_usd", 0.0),
        "confidence": agent_dict.get("confidence", 0.0),
        "error": agent_dict.get("error"),
    }


# ────────────────────────────────────────────────────────────────────────────
# Cached Agent Loading (for Replay)
# ────────────────────────────────────────────────────────────────────────────


def load_cached_agents(run_id: str) -> dict[str, str]:
    """Alias for load_agent_logs() — used by replay subcommand.

    Returns dict of agent alias -> stdout content.
    Raises ValueError if < 2 agents found (synthesis requires duality).

    Args:
        run_id: Full run_id

    Returns:
        Dict mapping agent alias to stdout content

    Raises:
        ValueError: If fewer than 2 agents found
    """
    agents = load_agent_logs(run_id)
    if len(agents) < 2:
        raise ValueError(
            f"Run {run_id} has only {len(agents)} agent(s). "
            "Replay requires dual+ mode runs (2+ agents) for synthesis."
        )
    return agents

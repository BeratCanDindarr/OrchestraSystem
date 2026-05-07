"""Sync Orchestra run state into the project's .claude/session-state/active.json."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from orchestra.models import OrchestraRun


def _active_json_path() -> Path | None:
    """Walk up from CWD to find .claude/session-state/active.json."""
    from pathlib import Path
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        candidate = parent / ".claude" / "session-state" / "active.json"
        if candidate.exists():
            return candidate
    return None


def sync_run_to_session(run: OrchestraRun) -> bool:
    """
    Write orchestra run summary into active.json under the 'orchestra' key.
    Non-blocking — errors are silently ignored to never block the tool call.
    Returns True if update succeeded.
    """
    path = _active_json_path()
    if not path:
        return False

    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    try:
        # Additive update — don't touch existing keys
        state.setdefault("orchestra", {})
        state["orchestra"] = {
            "last_run_id": run.run_id,
            "mode": run.mode,
            "status": run.status.value,
            "task": run.task[:120],
            "agent_count": len(run.agents),
            "completed_agents": sum(
                1 for a in run.agents if a.status.value == "completed"
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        state["updated_at"] = datetime.now(timezone.utc).isoformat()

        path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return True
    except OSError:
        return False

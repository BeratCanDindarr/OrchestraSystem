"""Global Blackboard: A shared memory space for agents to collaborate during a run."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from orchestra.storage.db import get_db

def write_note(run_id: str, alias: str, content: str, tags: List[str] = None):
    """Write a persistent note for other agents to see."""
    ts = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    with conn:
        conn.execute(
            "INSERT INTO blackboard (run_id, alias, content, tags, ts) VALUES (?, ?, ?, ?, ?)",
            (run_id, alias, content, json.dumps(tags or []), ts)
        )
        conn.commit()
    
    # Trigger an event for live UI/TUI observers
    from orchestra.engine import artifacts
    artifacts.append_event(run_id, {"event": "blackboard_updated", "alias": alias, "note": content[:100]})

def read_notes(run_id: str) -> List[Dict[str, Any]]:
    """Retrieve all shared notes for a specific run."""
    conn = get_db()
    rows = conn.execute(
        "SELECT alias, content, tags, ts FROM blackboard WHERE run_id = ? ORDER BY ts ASC",
        (run_id,)
    ).fetchall()
    
    return [
        {"alias": row[0], "content": row[1], "tags": json.loads(row[2]), "ts": row[3]}
        for row in rows
    ]

def get_blackboard_context(run_id: str) -> str:
    """Format blackboard notes as a prompt context for agents."""
    notes = read_notes(run_id)
    if not notes: return ""
    
    context = "\n[GLOBAL BLACKBOARD (Shared Wisdom)]\n"
    for n in notes:
        context += f"- From {n['alias']}: {n['content']}\n"
    return context + "\n"


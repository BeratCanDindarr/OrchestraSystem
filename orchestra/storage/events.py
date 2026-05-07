"""Event Store for Orchestra: Atomic SQLite + JSONL event logging."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from orchestra.storage.db import get_db

def append_event(
    run_id: str,
    event_type: str,
    data: Dict[str, Any],
    causation_id: Optional[str] = None,
    idempotency_key: Optional[str] = None
) -> int:
    """
    Atomically append event to SQLite and return its sequence ID.
    """
    ts = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    with conn:
        # Get next sequence ID for this run
        row = conn.execute(
            "SELECT last_seq_id FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        
        # If run doesn't exist, last_seq_id is 0. 
        # Actually, run should exist before events, but we handle it gracefully.
        last_seq = row[0] if row else 0
        new_seq = last_seq + 1
        
        # Insert event
        conn.execute(
            """
            INSERT INTO events (run_id, seq, event_type, causation_id, idempotency_key, ts, data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, new_seq, event_type, causation_id, idempotency_key, ts, json.dumps(data))
        )
        
        # Update run snapshot meta
        conn.execute(
            "UPDATE runs SET last_seq_id = ?, updated_at = ? WHERE run_id = ?",
            (new_seq, ts, run_id)
        )
        conn.commit()
    
    # Also write to local JSONL for redundancy (Phase 0 logic preserved)
    try:
        from orchestra.engine import artifacts
        artifacts.append_event_to_file(run_id, {
            "seq": new_seq,
            "event_type": event_type,
            "ts": ts,
            "causation_id": causation_id,
            "idempotency_key": idempotency_key,
            "data": data
        })
    except Exception:
        pass # Logging failure shouldn't stop DB commit
        
    return new_seq

def get_event_stream(run_id: str, after_seq: int = 0) -> List[Dict[str, Any]]:
    """Retrieve all events for a run after a certain sequence number."""
    conn = get_db()
    rows = conn.execute(
        "SELECT seq, event_type, ts, causation_id, idempotency_key, data FROM events WHERE run_id = ? AND seq > ? ORDER BY seq ASC",
        (run_id, after_seq)
    ).fetchall()
    
    events = []
    for row in rows:
        events.append({
            "seq": row[0],
            "event_type": row[1],
            "ts": row[2],
            "causation_id": row[3],
            "idempotency_key": row[4],
            "data": json.loads(row[5])
        })
    return events

def find_tool_result(idempotency_key: str) -> Optional[Dict[str, Any]]:
    """Look for a completed tool result by its idempotency key."""
    if not idempotency_key:
        return None
    conn = get_db()
    row = conn.execute(
        "SELECT data FROM events WHERE idempotency_key = ? AND event_type = 'TOOL_CALL_COMPLETED'",
        (idempotency_key,)
    ).fetchone()
    if row:
        return json.loads(row[0])
    return None

from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
"""Event Store for Orchestra: Atomic SQLite + JSONL event logging."""

import json
from typing import Any, Dict, List, Optional
from orchestra.storage.db import get_event_log


def append_event(
    run_id: str,
    event_type: str,
    data: Dict[str, Any],
    causation_id: Optional[str] = None,
    idempotency_key: Optional[str] = None
) -> int:
    """
    Atomically append event to SQLite and return its sequence ID.

    Deprecated: Use EventLog.append_raw() directly for new code.
    """
    if causation_id:
        data["causation_id"] = causation_id
    if idempotency_key:
        data["idempotency_key"] = idempotency_key

    log = get_event_log()
    return log.append_raw(event_type, run_id, data)


def get_event_stream(run_id: str, after_seq: int = 0) -> List[Dict[str, Any]]:
    """Retrieve all events for a run after a certain sequence number."""
    log = get_event_log()
    events = [e for e in log.replay(run_id) if e.seq > after_seq]
    
    # Translate Event dataclass to legacy dict format
    res = []
    for e in events:
        res.append({
            "seq": e.seq,
            "event_type": e.event_type,
            "ts": e.ts, # legacy ts might be isoformat, EventLog returns float epoch
            "causation_id": e.payload.get("causation_id"),
            "idempotency_key": e.payload.get("idempotency_key"),
            "data": e.payload,
        })
    return res


def find_tool_result(idempotency_key: str) -> Optional[Dict[str, Any]]:
    """Look for a completed tool result by its idempotency key."""
    if not idempotency_key:
        return None
    # We have to scan events for idempotency_key in payload.
    # Because find_tool_result doesn't take run_id, we need to query the raw db
    from orchestra.storage.db import get_db
    conn = get_db()
    columns = [row[1] for row in conn.execute("PRAGMA table_info(events)")]
    payload_col = "payload" if "payload" in columns else "data"
    rows = conn.execute(
        f"SELECT {payload_col} FROM events WHERE event_type = 'TOOL_CALL_COMPLETED'"
    ).fetchall()
    for row in rows:
        payload_str = row[0]
        if not payload_str:
            continue
        try:
            payload = json.loads(payload_str)
            if payload.get("idempotency_key") == idempotency_key:
                return payload
        except json.JSONDecodeError:
            continue
    return None

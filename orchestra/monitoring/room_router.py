"""Phase 2 routing: hand a completed room's output off to connected rooms.

When a dispatch tagged with a ``room_id`` finishes, its output flows to that room's
``connections``:

- A **Decision-Maker** room routes to exactly one connected room. It reads a routing
  directive from the model's own output — a line ``ROUTE: <room name>`` (or the room
  name appearing anywhere) — falling back to the first connection.
- Any other room **broadcasts** its output to all connected rooms.

Each hand-off auto-casts a free model, runs the previous output as the new task under
the target room's role + rules, and chains its own router (depth-capped to stop cycles).
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque

_MAX_DEPTH = 6      # hard cap on hand-off chain length (cycle/runaway guard)
_ROUTE_RE = re.compile(r"^\s*(?:ROUTE|NEXT)\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

# ── flow history: a bounded log of every hand-off (from → to) ─────────────────
_flow_log: deque = deque(maxlen=200)
_flow_lock = threading.Lock()


def record_flow(entry: dict) -> None:
    with _flow_lock:
        _flow_log.appendleft(entry)


def recent_flows(limit: int = 50) -> list:
    with _flow_lock:
        return list(_flow_log)[:limit]


def clear_flows() -> None:
    with _flow_lock:
        _flow_log.clear()


def parse_routes(output: str, targets: list) -> list:
    """All target rooms a decision output points to (conditional multi-routing).

    Collects every ``ROUTE: <name>`` / ``NEXT: <name>`` directive and matches each to a
    target (order-preserving, deduped). With no matching directive, falls back to the
    single last-mentioned target. ``targets`` = candidate Room objects.
    """
    text = output or ""
    matched, seen = [], set()
    for directive in _ROUTE_RE.findall(text):
        low = directive.strip().lower()
        for t in targets:
            tn = t.name.strip().lower()
            if (tn in low or low in tn) and t.id not in seen:
                seen.add(t.id)
                matched.append(t)
    if matched:
        return matched
    # no explicit directive → last-mentioned target name
    chosen, best = None, -1
    lowtext = text.lower()
    for t in targets:
        idx = lowtext.rfind(t.name.strip().lower())
        if idx > best:
            best, chosen = idx, t
    return [chosen] if chosen else []


def parse_route(output: str, targets: list):
    """First routed target (back-compat single-route helper), or None."""
    routes = parse_routes(output, targets)
    return routes[0] if routes else None


def _pick_model(roster, task: str):
    """Auto-cast a free model for a hand-off task; return its character id or None."""
    from orchestra.monitoring import characters
    try:
        cast = characters.auto_cast(task, roster)
        if "error" not in cast and cast.get("character_id"):
            return cast["character_id"]
    except Exception:               # router infra unavailable → fall back below
        pass
    for c, state in roster:         # fallback: prefer a free model, else any non-offline
        if state in ("available", "idle"):
            return c.id
    for c, state in roster:
        if state != "offline":
            return c.id
    return None


def make_router(depth: int = 0, *, dispatch_fn=None, roster_fn=None):
    """Build an ``on_complete(dispatch_id)`` callback that routes at the given depth."""
    def _on_complete(dispatch_id: str) -> None:
        route_completion(dispatch_id, depth=depth, dispatch_fn=dispatch_fn, roster_fn=roster_fn)
    return _on_complete


def route_completion(dispatch_id: str, *, depth: int = 0, inbox=None, store=None,
                     roster_fn=None, dispatch_fn=None) -> list:
    """Route a finished room dispatch to its connected rooms. Returns [(room_id, cid)]."""
    if depth >= _MAX_DEPTH:
        return []
    from orchestra.monitoring import characters, room_presets
    if inbox is None:
        from orchestra.monitoring.dispatch_inbox import get_inbox
        inbox = get_inbox()
    d = inbox.get(dispatch_id)
    if d is None or d.status != "completed" or not d.room_id:
        return []
    if store is None:
        from orchestra.monitoring.room_store import get_room_store
        store = get_room_store()
    room = store.get(d.room_id)
    if room is None or not room.connections:
        return []
    all_rooms = {r.id: r for r in store.list()}
    targets = [all_rooms[c] for c in room.connections if c in all_rooms]
    if not targets:
        return []
    if room.preset == "decision_maker":
        chosen = parse_routes(d.output, targets)     # conditional: may route to several
        targets = chosen if chosen else targets[:1]

    dispatch = dispatch_fn if dispatch_fn is not None else characters.dispatch_task
    if roster_fn is None:
        from orchestra.monitoring.dashboard_app import _fleet_items as roster_fn  # type: ignore
    roster = roster_fn()
    roster_ids = {c.id for c, _s in roster}
    handoffs = []
    for t in targets:
        # a room's default model wins (if still present); otherwise auto-cast a free model
        default = getattr(t, "default_model", "") or ""
        if default:
            from orchestra.monitoring.node_runners import runnable_model
            default = runnable_model(default)
        cid = default if default in roster_ids else _pick_model(roster, d.output)
        if not cid:
            continue
        context = room_presets.compose_context(t.preset, t.rules)
        nxt = make_router(depth + 1, dispatch_fn=dispatch_fn, roster_fn=roster_fn)
        dispatch(cid, d.output, context=context, room_id=t.id, on_complete=nxt)
        record_flow({"from_id": room.id, "from_name": room.name,
                     "to_id": t.id, "to_name": t.name, "cid": cid, "ts": time.time()})
        handoffs.append((t.id, cid))
    return handoffs

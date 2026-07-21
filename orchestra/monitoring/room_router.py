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

_MAX_DEPTH = 6      # hard cap on hand-off chain length (cycle/runaway guard)
_ROUTE_RE = re.compile(r"^\s*(?:ROUTE|NEXT)\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_route(output: str, targets: list):
    """Pick the target room a decision output points to, or None.

    Prefers a ``ROUTE: <name>`` / ``NEXT: <name>`` directive; otherwise the last target
    whose name appears in the output. ``targets`` = candidate Room objects.
    """
    text = output or ""
    directives = _ROUTE_RE.findall(text)
    for directive in reversed(directives):
        low = directive.strip().lower()
        for t in targets:
            if t.name.strip().lower() in low or low in t.name.strip().lower():
                return t
    # no explicit directive → last-mentioned target name
    chosen = None
    lowtext = text.lower()
    best = -1
    for t in targets:
        idx = lowtext.rfind(t.name.strip().lower())
        if idx > best:
            best, chosen = idx, t
    return chosen


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
        chosen = parse_route(d.output, targets)
        targets = [chosen] if chosen else targets[:1]

    dispatch = dispatch_fn if dispatch_fn is not None else characters.dispatch_task
    if roster_fn is None:
        from orchestra.monitoring.dashboard_app import _fleet_items as roster_fn  # type: ignore
    roster = roster_fn()
    roster_ids = {c.id for c, _s in roster}
    handoffs = []
    for t in targets:
        # a room's default model wins (if still present); otherwise auto-cast a free model
        default = getattr(t, "default_model", "") or ""
        cid = default if default in roster_ids else _pick_model(roster, d.output)
        if not cid:
            continue
        context = room_presets.compose_context(t.preset, t.rules)
        nxt = make_router(depth + 1, dispatch_fn=dispatch_fn, roster_fn=roster_fn)
        dispatch(cid, d.output, context=context, room_id=t.id, on_complete=nxt)
        handoffs.append((t.id, cid))
    return handoffs

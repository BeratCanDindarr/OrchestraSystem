from __future__ import annotations
"""Pure HTML-partial renderers for the live dashboard.

Every function returns an HTML string built for HTMX swaps — no I/O, so they are
trivially unit-testable. User-supplied text is escaped. ``sse_pack`` frames a
Server-Sent Events message for the live stream.
"""
import html
from typing import Any, Optional


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def sse_pack(event: str, data: str) -> bytes:
    """Frame one SSE message. Data is flattened to a single line (SSE is line-based)."""
    flat = data.replace("\n", " ")
    return f"event: {event}\ndata: {flat}\n\n".encode("utf-8")


def sparkline_svg(values: list[float], *, width: int = 120, height: int = 28) -> str:
    """Render a minimal inline SVG sparkline polyline from ``values``."""
    if not values:
        return f'<svg class="spark" width="{width}" height="{height}"></svg>'
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    n = len(values)
    step = width / max(n - 1, 1)
    points = " ".join(
        f"{i * step:.1f},{height - ((v - lo) / span) * height:.1f}" for i, v in enumerate(values)
    )
    return (
        f'<svg class="spark" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<polyline fill="none" stroke="currentColor" stroke-width="1.5" points="{points}"/></svg>'
    )


def bars_svg(values: list[float], *, width: int = 220, height: int = 46, gap: int = 2) -> str:
    """Render a minimal inline SVG bar chart (baseline at the bottom)."""
    if not values:
        return f'<svg class="bars" width="{width}" height="{height}"></svg>'
    hi = max(values) or 1.0
    n = len(values)
    bw = max((width - gap * (n - 1)) / n, 1.0)
    rects = []
    for i, v in enumerate(values):
        bh = max((v / hi) * (height - 2), 0.0)
        x = i * (bw + gap)
        rects.append(f'<rect x="{x:.1f}" y="{height - bh:.1f}" width="{bw:.1f}" '
                     f'height="{bh:.1f}" rx="1"/>')
    return (f'<svg class="bars" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" preserveAspectRatio="none">{"".join(rects)}</svg>')


def _ts_epoch(value: Any) -> Optional[float]:
    """Parse an ISO-8601 timestamp to epoch seconds; None if unparseable."""
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (ValueError, TypeError):
        return None


def aggregate_timeseries(rows: list[dict], *, buckets: int = 12) -> dict:
    """Bucket run rows into a cost/latency/volume time series.

    Rows span ``[min(created_at), max(created_at)]`` split into ``buckets`` equal
    slices — no wall-clock dependency, so it is deterministic and unit-testable.
    Latency is ``updated_at - created_at`` (seconds, floored at 0).
    """
    parsed = []
    for row in rows:
        ts = _ts_epoch(row.get("created_at"))
        if ts is None:
            continue
        done = _ts_epoch(row.get("updated_at"))
        latency = max((done - ts), 0.0) if done is not None else 0.0
        parsed.append((ts, float(row.get("total_cost_usd") or 0.0), latency,
                       str(row.get("status") or "")))
    zero = {"buckets": [], "totals": {"cost": 0.0, "runs": 0, "failures": 0, "avg_latency_s": 0.0}}
    if not parsed:
        return zero
    from datetime import datetime, timezone
    lo = min(p[0] for p in parsed)
    hi = max(p[0] for p in parsed)
    span = hi - lo
    slots: list[dict] = [{"cost": 0.0, "count": 0, "_lat": []} for _ in range(buckets)]
    for ts, cost, latency, _status in parsed:
        idx = 0 if span <= 0 else min(buckets - 1, int((ts - lo) / span * buckets))
        slots[idx]["cost"] += cost
        slots[idx]["count"] += 1
        slots[idx]["_lat"].append(latency)
    out_buckets = []
    for i, slot in enumerate(slots):
        start = lo + (span * i / buckets if span > 0 else 0)
        label = datetime.fromtimestamp(start, timezone.utc).strftime("%m-%d %H:%M")
        lat = slot["_lat"]
        out_buckets.append({
            "label": label, "cost": round(slot["cost"], 6), "count": slot["count"],
            "avg_latency_s": round(sum(lat) / len(lat), 3) if lat else 0.0,
        })
    all_lat = [p[2] for p in parsed]
    return {
        "buckets": out_buckets,
        "totals": {
            "cost": round(sum(p[1] for p in parsed), 6),
            "runs": len(parsed),
            "failures": sum(1 for p in parsed if p[3] == "failed"),
            "avg_latency_s": round(sum(all_lat) / len(all_lat), 3) if all_lat else 0.0,
        },
    }


def render_charts(series: dict) -> str:
    """Cost bar chart + latency sparkline + summary line, from ``aggregate_timeseries``."""
    buckets = series.get("buckets", []) or []
    if not buckets:
        return '<div class="charts empty"><h3>Trends</h3><p>No run history yet</p></div>'
    totals = series.get("totals", {}) or {}
    cost_bars = bars_svg([float(b["cost"]) for b in buckets])
    lat_line = sparkline_svg([float(b["avg_latency_s"]) for b in buckets])
    first, last = _esc(buckets[0]["label"]), _esc(buckets[-1]["label"])
    return (
        '<div class="charts"><h3>Trends</h3>'
        f'<div class="chart"><div class="chart-head"><span>Cost</span>'
        f'<b>${float(totals.get("cost", 0.0)):.2f}</b></div>{cost_bars}</div>'
        f'<div class="chart lat"><div class="chart-head"><span>Latency</span>'
        f'<b>{float(totals.get("avg_latency_s", 0.0)):.1f}s avg</b></div>{lat_line}</div>'
        f'<div class="chart-axis"><span>{first}</span><span>{last}</span></div>'
        f'<div class="chart-foot">{_esc(totals.get("runs", 0))} runs · '
        f'{_esc(totals.get("failures", 0))} failed</div></div>'
    )


def filter_runs(rows: list[dict], query: str) -> list[dict]:
    """Filter run rows by a search string.

    Free-text tokens match (case-insensitively) against run_id/task/mode/status;
    ``field:value`` tokens (``status:``, ``mode:``) constrain a single field. A
    blank query returns nothing (search is opt-in). All tokens must match (AND).
    """
    q = (query or "").strip().lower()
    if not q:
        return []
    field_tokens: list[tuple[str, str]] = []
    free_tokens: list[str] = []
    for tok in q.split():
        if tok.startswith(("status:", "mode:")) and ":" in tok:
            field, _, val = tok.partition(":")
            if val:
                field_tokens.append((field, val))
        else:
            free_tokens.append(tok)

    def matches(row: dict) -> bool:
        for field, val in field_tokens:
            if str(row.get(field, "")).lower() != val:
                return False
        if free_tokens:
            hay = " ".join(str(row.get(k, "")) for k in ("run_id", "task", "mode", "status")).lower()
            if not all(tok in hay for tok in free_tokens):
                return False
        return True

    return [row for row in rows if matches(row)]


def render_search_results(rows: list[dict], query: str) -> str:
    """Compact clickable result list for the global search dropdown."""
    if not (query or "").strip():
        return '<div class="search-results empty">Type to search runs…</div>'
    if not rows:
        return '<div class="search-results empty">No matches</div>'
    items = []
    for row in rows[:20]:
        rid = _esc(row.get("run_id", ""))
        task = _esc(row.get("task", "") or "")
        if len(task) > 70:
            task = task[:70] + "…"
        items.append(
            f'<div class="sr-row" hx-get="/api/runs/{rid}" hx-target="#drill" hx-swap="innerHTML" '
            f'role="button" tabindex="0">'
            f'<span class="sr-status st-{_esc(row.get("status", ""))}">{_esc(row.get("status", ""))}</span>'
            f'<span class="sr-mode">{_esc(row.get("mode", ""))}</span>'
            f'<span class="sr-task">{task}</span>'
            f'<span class="sr-id">{rid}</span></div>'
        )
    return f'<div class="search-results">{"".join(items)}</div>'


def render_metrics(m: dict) -> str:
    by_mode = m.get("by_mode", {}) or {}
    mode_spark = sparkline_svg([float(v) for v in by_mode.values()]) if by_mode else ""
    cards = [
        ("Runs", str(m.get("total_runs", 0))),
        ("Total cost", f"${float(m.get('total_cost_usd', 0.0)):.2f}"),
        ("Avg cost", f"${float(m.get('avg_cost_usd', 0.0)):.2f}"),
        ("Failure rate", f"{float(m.get('failure_rate', 0.0)) * 100:.0f}%"),
    ]
    body = "".join(
        f'<div class="metric"><span class="metric-label">{_esc(label)}</span>'
        f'<span class="metric-value">{_esc(value)}</span></div>'
        for label, value in cards
    )
    return f'<div class="metrics">{body}<div class="metric"><span class="metric-label">By mode</span>{mode_spark}</div></div>'


def render_incidents(d: dict) -> str:
    if not d:
        return '<div class="incidents empty">No incidents</div>'
    rows = "".join(
        f'<li><span class="itype">{_esc(k)}</span><span class="icount">{_esc(v)}</span></li>'
        for k, v in d.items()
    )
    return f'<ul class="incidents">{rows}</ul>'


def render_eval(e: dict) -> str:
    total = e.get("total", 0)
    passed = e.get("passed", 0)
    rate = float(e.get("pass_rate", 0.0)) * 100
    return (
        f'<div class="eval"><span class="metric-value">{_esc(passed)}/{_esc(total)}</span>'
        f'<span class="metric-label">pass ({rate:.0f}%)</span></div>'
    )


def render_budget(b: dict) -> str:
    spent = float(b.get("spent", 0.0))
    budget = b.get("budget")
    remaining = b.get("remaining")
    if budget:
        rem_val = float(remaining) if remaining is not None else 0.0
        pct = min(spent / float(budget), 1.0) * 100 if float(budget) > 0 else 0.0
        over = rem_val < 0
        fill_cls = "bar-fill over" if over else "bar-fill"
        bar = f'<div class="bar"><div class="{fill_cls}" style="width:{pct:.0f}%"></div></div>'
        if over:
            rem = f'<span class="metric-label over">over budget by ${abs(rem_val):.2f} (of ${float(budget):.2f})</span>'
        else:
            rem = f'<span class="metric-label">${rem_val:.2f} left of ${float(budget):.2f}</span>'
    else:
        bar = ""
        rem = '<span class="metric-label">no budget set</span>'
    return f'<div class="budget"><span class="metric-value">${spent:.2f} spent</span>{bar}{rem}</div>'


def render_runs_table(rows: list[dict]) -> str:
    if not rows:
        return '<table class="runs"><tbody><tr><td>No runs yet</td></tr></tbody></table>'
    body = "".join(
        f'<tr class="run-row" hx-get="/api/runs/{_esc(row.get("run_id"))}" '
        f'hx-target="#drill" hx-swap="innerHTML" role="button">'
        f'<td>{_esc(row.get("run_id"))}</td><td>{_esc(row.get("mode"))}</td>'
        f'<td class="st-{_esc(row.get("status"))}">{_esc(row.get("status"))}</td>'
        f'<td>${float(row.get("total_cost_usd", 0.0) or 0.0):.2f}</td>'
        f'<td>{_esc(row.get("created_at"))}</td></tr>'
        for row in rows
    )
    return (
        '<table class="runs"><thead><tr><th>run</th><th>mode</th><th>status</th>'
        f'<th>cost</th><th>created</th></tr></thead><tbody>{body}</tbody></table>'
    )


def render_run_detail(run: dict) -> str:
    rid = _esc(run.get("run_id"))
    agents = run.get("agents", []) or []
    arows = "".join(
        f'<tr><td>{_esc(a.get("alias"))}</td><td>{_esc(a.get("status"))}</td>'
        f'<td>{float(a.get("estimated_cost_usd", 0.0) or 0.0):.4f}</td>'
        f'<td>{float(a.get("confidence", 0.0) or 0.0):.2f}</td></tr>'
        for a in agents
    )
    summary = _esc(run.get("summary") or "")
    return (
        f'<div class="run-detail"><h3>Run {rid} — {_esc(run.get("status"))}</h3>'
        f'<table class="agents"><thead><tr><th>alias</th><th>status</th><th>cost</th><th>conf</th></tr></thead>'
        f'<tbody>{arows}</tbody></table>'
        f'<div class="summary">{summary}</div>'
        f'<div hx-get="/api/runs/{rid}/events" hx-trigger="load" hx-swap="innerHTML">loading timeline…</div>'
        f'<div hx-get="/api/runs/{rid}/trace" hx-trigger="load" hx-swap="innerHTML"></div></div>'
    )


def render_timeline(events: list[dict]) -> str:
    if not events:
        return '<div class="timeline empty">No events</div>'
    items = "".join(
        f'<li><span class="seq">{_esc(e.get("seq"))}</span>'
        f'<span class="etype">{_esc(e.get("event_type"))}</span>'
        f'<span class="ts">{_esc(e.get("ts"))}</span></li>'
        for e in events
    )
    return f'<ol class="timeline">{items}</ol>'


def render_trace(spans: list[dict]) -> str:
    if not spans:
        return '<div class="trace empty">No trace spans</div>'
    items = "".join(
        f'<li>{_esc(s.get("name"))} '
        f'<span class="ms">{float(s.get("total_latency_ms", 0.0) or 0.0):.0f}ms</span></li>'
        for s in spans
    )
    return f'<ul class="trace">{items}</ul>'


def render_paused_cards(rows: list[dict], *, enabled: bool) -> str:
    if not rows:
        return '<div class="hitl empty">No runs awaiting approval</div>'
    cards = []
    for row in rows:
        rid = _esc(row.get("run_id"))
        head = (
            f'<div class="hitl-head">Run {rid}</div>'
            f'<div class="hitl-meta">by {_esc(row.get("paused_by"))} at {_esc(row.get("paused_at"))}</div>'
        )
        if enabled:
            buttons = (
                f'<button hx-post="/api/approve" hx-vals=\'{{"run_id":"{rid}","decision":"approve"}}\' '
                f'hx-target="#hitl" hx-swap="innerHTML">Approve</button>'
                f'<button hx-post="/api/approve" hx-vals=\'{{"run_id":"{rid}","decision":"reject"}}\' '
                f'hx-target="#hitl" hx-swap="innerHTML">Reject</button>'
            )
        else:
            buttons = '<span class="disabled">approvals disabled</span>'
        cards.append(f'<div class="hitl-card">{head}<div class="hitl-actions">{buttons}</div></div>')
    return f'<div class="hitl">{"".join(cards)}</div>'


def render_run_update(run_id: str, event: dict) -> str:
    """Small HTML fragment pushed over SSE when a run/agent event fires."""
    etype = _esc(event.get("event") or event.get("event_type") or "update")
    return f'<div class="feed-item"><span class="rid">{_esc(run_id)}</span> <span class="etype">{etype}</span></div>'


def render_character_card(char, state: str, *, dispatch_enabled: bool) -> str:
    """Render one model as a character card. ``char.avatar`` is our own generated
    SVG (trusted markup); all persona text is escaped."""
    rid = _esc(char.id)
    dispatch = ""
    if dispatch_enabled:
        dispatch = (
            '<form class="dispatch" hx-post="/api/dispatch" hx-target="#fleet" hx-swap="innerHTML">'
            f'<input type="hidden" name="character_id" value="{rid}">'
            '<input class="task-in" name="task" placeholder="Assign a task…" autocomplete="off">'
            '<button>Dispatch</button></form>'
        )
    return (
        f'<div class="char-card st-{_esc(state)}" data-cid="{rid}">'
        f'{char.avatar}'
        f'<div class="char-head"><span class="char-name">{_esc(char.name)}</span>'
        f'<span class="char-state">{_esc(state)}</span></div>'
        f'<div class="char-role">{_esc(char.role)} · {_esc(char.provider)} · {_esc(char.model)}</div>'
        f'<button class="chat-btn" hx-get="/api/character/{rid}" hx-target="#chat" hx-swap="innerHTML">Chat</button>'
        f'{dispatch}</div>'
    )


_DISPATCH_HINT = (
    '<div class="cast-notice warn">Dispatch &amp; chat are read-only. To assign tasks or chat, set '
    '<code>dashboard.dispatch_enabled = true</code> in <code>.orchestra/config.toml</code> — or launch with '
    '<code>ORCHESTRA_DASHBOARD__DISPATCH_ENABLED=true orchestra dashboard</code> — then reload.</div>'
)


def render_fleet(items, *, dispatch_enabled: bool, notice: Optional[str] = None) -> str:
    """Render the whole fleet from a list of (Character, state) pairs.

    ``notice`` shows an auto-cast explanation banner ("picked X because …"). When
    dispatch is disabled, a persistent hint explains how to enable it — otherwise a
    gated click looks like nothing happened.
    """
    banner = f'<div class="cast-notice">{_esc(notice)}</div>' if notice else ""
    if not dispatch_enabled:
        banner += _DISPATCH_HINT
    if not items:
        return f'{banner}<div class="fleet empty">No models configured</div>'
    cards = "".join(render_character_card(c, s, dispatch_enabled=dispatch_enabled) for c, s in items)
    return f'{banner}<div class="fleet">{cards}</div>'


def render_character_detail(char, *, chat_enabled: bool) -> str:
    """Character detail panel: persona + assign-a-task + chat + work history."""
    rid = _esc(char.id)
    if chat_enabled:
        # assign a task (target the office) + chat (data-chat-cid → fleet.js SSE stream).
        interact = (
            f'<form class="detail-dispatch" hx-post="/api/dispatch" hx-target="#fleet" hx-swap="innerHTML" '
            f'hx-on::after-request="this.reset()">'
            f'<input type="hidden" name="character_id" value="{rid}">'
            f'<input class="task-in" name="task" placeholder="Assign a task to {_esc(char.name)}…" autocomplete="off">'
            '<button>Assign</button></form>'
            f'<form class="chat-form" data-chat-cid="{rid}">'
            f'<input class="task-in" name="message" placeholder="Chat with {_esc(char.name)}…" autocomplete="off">'
            '<button>Send</button></form>'
        )
    else:
        interact = '<div class="disabled">assign &amp; chat disabled — enable dashboard.dispatch_enabled</div>'
    return (
        '<div class="char-detail">'
        f'<div class="cd-head">{char.avatar}<div><span class="char-name">{_esc(char.name)}</span>'
        f'<div class="char-role">{_esc(char.role)} · {_esc(char.provider)} · {_esc(char.model)}</div></div>'
        f'<button class="hist-btn" hx-get="/api/character/{rid}/history" hx-target="#char-history" '
        f'hx-swap="innerHTML">History</button></div>'
        '<div id="char-history"></div>'
        '<div id="chat-log" class="chat-log"></div>'
        f'{interact}</div>'
    )


def render_chat_exchange(name: str, message: str, reply: str) -> str:
    """One user message + the character's reply, appended to the chat log."""
    return (
        f'<div class="chat-msg you"><span class="who">you</span> {_esc(message)}</div>'
        f'<div class="chat-msg them"><span class="who">{_esc(name)}</span> {_esc(reply)}</div>'
    )


def render_inbox(dispatches, names: dict) -> str:
    """Top tray: running dispatches + completed-but-unread results (click to read)."""
    if not dispatches:
        return '<div class="inbox empty">No active work. Assign a task to a model above.</div>'
    unread = sum(1 for d in dispatches if d.status in ("completed", "failed") and not d.read)
    head = f'<div class="inbox-head">Inbox{f" · {unread} new" if unread else ""}</div>'
    items = []
    for d in dispatches:
        who = _esc(names.get(d.character_id, d.character_id))
        task = _esc((d.task or "")[:70])
        if d.status == "running":
            items.append(
                f'<div class="inbox-item running" data-did="{_esc(d.id)}" '
                f'hx-get="/api/dispatch/{_esc(d.id)}" hx-target="#output" hx-swap="innerHTML" '
                f'role="button" title="Watch live"><span class="ii-icon">⏳</span>'
                f'<span class="ii-who">{who}</span><span class="ii-task">{task}</span></div>'
            )
        else:
            cls = "done" if d.status == "completed" else "failed"
            icon = "✅" if d.status == "completed" else "⚠"
            items.append(
                f'<div class="inbox-item {cls}" data-did="{_esc(d.id)}" hx-get="/api/dispatch/{_esc(d.id)}" '
                f'hx-target="#output" hx-swap="innerHTML" role="button" title="Read result">'
                f'<span class="ii-icon">{icon}</span><span class="ii-who">{who}</span>'
                f'<span class="ii-task">{task}</span><span class="ii-new">new</span></div>'
            )
    return f'<div class="inbox">{head}{"".join(items)}</div>'


def render_dispatch_output(dispatch, name: str) -> str:
    """Terminal-like viewer for a dispatch's task + output (+ pin/archive)."""
    status = _esc(dispatch.status)
    did = _esc(dispatch.id)
    pin_label = "Unpin" if dispatch.pinned else "Pin"
    pin_action = "unpin" if dispatch.pinned else "pin"
    actions = (
        f'<button hx-post="/api/dispatch/{did}/{pin_action}" hx-target="#inbox" hx-swap="innerHTML">{pin_label}</button>'
        f'<button hx-post="/api/dispatch/{did}/archive" hx-target="#inbox" hx-swap="innerHTML" '
        f'hx-on::after-request="document.getElementById(\'output\').innerHTML=\'\'">Archive</button>'
    )
    if dispatch.status == "running":
        # live terminal: connect to the stream endpoint, append tokens as they arrive
        body = (
            f'<pre class="ov-body live" hx-ext="sse" sse-connect="/api/dispatch/{did}/stream" '
            f'sse-swap="token" hx-swap="beforeend">{_esc(dispatch.output)}</pre>'
        )
        actions = '<span class="ov-live">● live</span>'
    else:
        body = f'<pre class="ov-body">{_esc(dispatch.output or "(no output)")}</pre>'
    return (
        '<div class="output-viewer">'
        f'<div class="ov-head"><span class="ov-who">{_esc(name)}</span>'
        f'<span class="ov-status st-{status}">{status}</span>'
        f'<span class="ov-actions">{actions}</span></div>'
        f'<div class="ov-task">$ {_esc(dispatch.task)}</div>'
        f'{body}</div>'
    )


def render_history(dispatches, name: str) -> str:
    """A character's past dispatches (click to open the result)."""
    if not dispatches:
        return '<div class="hist empty">No past work yet</div>'
    rows = "".join(
        f'<div class="hist-row st-{_esc(d.status)}" hx-get="/api/dispatch/{_esc(d.id)}" '
        f'hx-target="#output" hx-swap="innerHTML" role="button">'
        f'<span class="hist-status">{_esc(d.status)}</span>'
        f'<span class="hist-task">{_esc((d.task or "")[:60])}</span></div>'
        for d in dispatches
    )
    return f'<div class="hist"><div class="hist-head">{_esc(name)} · history</div>{rows}</div>'


_OFFICE_BUBBLE = {
    "busy": "typing…", "available": "ready", "idle": "zzz", "offline": "away",
}


# ── isometric floor: a 2:1 diamond grid; positions come from grid (col,row) ──
_ISO_COLS = 6          # workers per row across the floor
_ISO_TILE_W = 90       # tile footprint width (px) — half is the +col x-step
_ISO_TILE_H = 46       # tile footprint height (px) — half is the shared y-step
_ISO_OX = 470          # screen origin (grid cell 0,0), roughly floor centre-top
_ISO_OY = 74
_ISO_LOUNGE_ROW = 5    # lounge sits this many rows in front of the workstations


def _iso(col: float, row: float) -> tuple[float, float]:
    """Project a grid cell onto screen px (feet-anchored, 2:1 isometric)."""
    x = _ISO_OX + (col - row) * (_ISO_TILE_W / 2)
    y = _ISO_OY + (col + row) * (_ISO_TILE_H / 2)
    return x, y


def render_office(items, *, dispatch_enabled: bool, notify_ids=frozenset(), notice: Optional[str] = None) -> str:
    """Isometric 2.5D office: workers rest in the lounge and walk to a desk to work.

    Each character owns a grid cell in the front lounge and one in the back
    workstation rows; both are projected to screen px (``--lx/--ly`` lounge,
    ``--dx/--dy`` desk) so CSS animates an iso walk between them. Sprites are
    upright billboards; ``--lz/--dz`` carry the depth (z-index) for each spot.
    ``items`` = list of (Character, presence-state); ``notify_ids`` = ids with an
    unread result. The zoom/pan viewport is driven by fleet.js.
    """
    banner = f'<div class="cast-notice">{_esc(notice)}</div>' if notice else ""
    if not dispatch_enabled:
        banner += _DISPATCH_HINT
    if not items:
        return f'{banner}<div class="office empty">No models configured</div>'
    sprites = []
    for i, (c, state) in enumerate(items):
        cid = _esc(c.id)
        col, row = i % _ISO_COLS, i // _ISO_COLS
        dx, dy = _iso(col, row)                                  # workstation (back)
        lx, ly = _iso(col, row + _ISO_LOUNGE_ROW)               # lounge (front, nearer viewer)
        dz, lz = 100 + int(col + row), 100 + int(col + row + _ISO_LOUNGE_ROW)
        style = (f"--dx:{dx:.0f}px;--dy:{dy:.0f}px;--dz:{dz};"
                 f"--lx:{lx:.0f}px;--ly:{ly:.0f}px;--lz:{lz}")
        has_result = c.id in notify_ids
        at_desk = state == "busy" or has_result
        bubble = "done!" if has_result else _OFFICE_BUBBLE.get(state, "")
        bubble_html = f'<div class="bubble">{"✅ " if has_result else ""}{_esc(bubble)}</div>' if bubble else ""
        cls = ("sprite" + (" at-desk" if at_desk else "") + f" st-{_esc(state)}"
               + (" has-result" if has_result else ""))
        # hover info card — richer than the tag, revealed on :hover / focus.
        card = (f'<div class="card"><b>{_esc(c.name)}</b>'
                f'<span class="crole">{_esc(c.role)}</span>'
                f'<span class="cmeta">{_esc(c.provider)} · {_esc(c.model)}</span>'
                f'<span class="cstate st-{_esc(state)}">{_esc(state)}</span></div>')
        # desk (rhombus) + upright monitor materialise on demand only when at-desk.
        sprites.append(
            f'<div class="{cls}" data-cid="{cid}" style="{style}" hx-get="/api/character/{cid}" '
            f'hx-target="#chat" hx-swap="innerHTML" role="button" tabindex="0" '
            f'title="{_esc(c.name)} — {_esc(c.role)}">'
            f'{bubble_html}{card}<div class="desk"></div><span class="mon"></span>'
            f'<div class="sbody"></div><div class="shead">{c.avatar}</div>'
            f'<div class="tag">{_esc(c.name)}</div></div>'
        )
    zones = ('<div class="zone zdesks"><span>Workstations</span></div>'
             '<div class="zone zlounge"><span>Lounge</span></div>')
    decor = (
        '<div class="decor plant p1"></div><div class="decor plant p2"></div>'
        '<div class="decor cooler"></div>'
    )
    reset = '<button class="iso-reset" type="button" title="Reset view">⤢</button>'
    scene = (f'<div class="iso-scene" data-iso><div class="iso-stage">'
             f'<div class="floor"></div>{zones}{decor}{"".join(sprites)}</div></div>{reset}')
    return f'{banner}<div class="office iso">{scene}</div>'


def render_status(s: dict) -> str:
    """Header status chips: models online/working, active jobs, version."""
    return (
        '<div class="status-chips">'
        f'<span class="chip"><b>{_esc(s.get("online", 0))}</b>/{_esc(s.get("total", 0))} online</span>'
        f'<span class="chip working"><b>{_esc(s.get("busy", 0))}</b> working</span>'
        f'<span class="chip jobs"><b>{_esc(s.get("active", 0))}</b> active</span>'
        f'<span class="chip ver">v{_esc(s.get("version", ""))}</span>'
        '</div>'
    )


def render_health(providers: list) -> str:
    """Provider health panel: which backends are reachable (green) or not (red)."""
    if not providers:
        return '<div class="health empty">No providers</div>'
    rows = "".join(
        f'<div class="hz-row"><span class="hz-dot {"up" if p.get("online", 0) > 0 else "down"}"></span>'
        f'<span class="hz-name">{_esc(p.get("name"))}</span>'
        f'<span class="hz-count">{_esc(p.get("online", 0))}/{_esc(p.get("total", 0))}</span></div>'
        for p in providers
    )
    up = sum(1 for p in providers if p.get("online", 0) > 0)
    return f'<div class="health"><div class="hz-head">Providers · {up}/{len(providers)} up</div>{rows}</div>'


def render_settings(cfg: dict) -> str:
    """Settings panel: toggle dispatch/approvals/remote + set budget from the browser.

    All controls live in one form that posts on change, so unchecked boxes are known
    by their absence. Session-scoped (env overrides; a restart reverts to config.toml).
    """
    def toggle(name, label):
        checked = "checked" if cfg.get(name) else ""
        return (f'<label class="setting"><input type="checkbox" name="{name}" {checked}>'
                f'<span>{_esc(label)}</span></label>')
    budget = float(cfg.get("budget_usd", 0.0) or 0.0)
    return (
        '<form class="settings" hx-post="/api/settings" hx-target="#settings" hx-swap="innerHTML" hx-trigger="change">'
        '<div class="s-head">Settings</div>'
        f'{toggle("dispatch_enabled", "Allow dispatch & chat (runs models)")}'
        f'{toggle("approvals_enabled", "Allow HITL approvals")}'
        f'{toggle("include_remote", "Include remote-node models")}'
        '<label class="setting num"><span>Budget ($, 0 = none)</span>'
        f'<input class="s-num" type="number" step="0.5" min="0" name="budget_usd" value="{budget:g}"></label>'
        '<div class="s-note">Session-scoped — reverts to config.toml on restart.</div>'
        '</form>'
    )

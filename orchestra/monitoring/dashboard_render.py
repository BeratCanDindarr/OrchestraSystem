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


def sse_pack_stream(event: str, data: str) -> bytes:
    """Frame an SSE message preserving newlines (terminal output).

    Each line becomes its own ``data:`` field; EventSource rejoins them with ``\\n``,
    so multi-line model output survives intact in the live viewer.
    """
    body = "".join(f"data: {line}\n" for line in data.split("\n"))
    return f"event: {event}\n{body}\n".encode("utf-8")


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


_AUDIT_FAIL_HINTS = ("fail", "error", "denied", "reject", "cancel", "timeout")


def render_audit(events: list[dict]) -> str:
    """Append-only event log view: most-recent events with type / run / time."""
    if not events:
        return '<div class="audit empty"><h3>Audit log</h3><p>No events yet</p></div>'
    rows = []
    for ev in events:
        etype = str(ev.get("event_type", ""))
        low = etype.lower()
        cls = "ev-fail" if any(h in low for h in _AUDIT_FAIL_HINTS) else "ev-ok"
        ts = str(ev.get("ts", ""))
        stamp = ts[11:19] if len(ts) >= 19 and ts[10] in ("T", " ") else ts
        rid = _esc(ev.get("run_id", "") or "—")
        rows.append(
            f'<div class="au-row {cls}">'
            f'<span class="au-time">{_esc(stamp)}</span>'
            f'<span class="au-type">{_esc(etype)}</span>'
            f'<span class="au-run" hx-get="/api/runs/{rid}" hx-target="#drill" hx-swap="innerHTML" '
            f'role="button" tabindex="0">{rid}</span></div>'
        )
    return f'<div class="audit"><h3>Audit log</h3>{"".join(rows)}</div>'


def render_flow(flows: list, names: Optional[dict] = None) -> str:
    """Recent room-to-room hand-offs (routing flow history), newest first."""
    if not flows:
        return '<div class="flow empty"><h3>Routing flow</h3><p>No hand-offs yet</p></div>'
    names = names or {}
    rows = []
    for f in flows:
        who = _esc(names.get(f.get("cid"), f.get("cid", "")))
        rows.append(
            '<div class="flow-row">'
            f'<span class="fl-from">{_esc(f.get("from_name", ""))}</span>'
            '<span class="fl-arrow">⤳</span>'
            f'<span class="fl-to">{_esc(f.get("to_name", ""))}</span>'
            f'<span class="fl-who">{who}</span></div>'
        )
    return f'<div class="flow"><h3>Routing flow</h3>{"".join(rows)}</div>'


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
        # live terminal: fleet.js opens an EventSource to the stream endpoint and
        # appends tokens as they arrive (robust JS path, same as chat streaming).
        body = f'<pre class="ov-body live" data-stream-did="{did}">{_esc(dispatch.output)}</pre>'
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


_ROSTER_STATE_LABEL = {
    "busy": "working", "available": "ready", "idle": "idle", "offline": "offline",
}
_MEMBER_STATUS_ICON = {"running": "⏳", "completed": "✅", "failed": "⚠"}
# roster sort order: active models on top, unavailable/offline sink to the bottom
_ROSTER_STATE_ORDER = {"busy": 0, "available": 1, "idle": 2, "offline": 4}


def _preset_label(key: str) -> str:
    from orchestra.monitoring.room_presets import get_preset
    p = get_preset(key)
    return p.name if p else (key or "Custom")


def rooms_with_members(store_rooms, dispatches, char_by_id: dict) -> list[dict]:
    """Attach live members (dispatches tagged with a room's id) to each stored room.

    ``store_rooms`` = Room objects (id, name, preset, rules). ``dispatches`` = Dispatch
    objects carrying ``room_id``. Members are deduped per character (most-active status
    wins). Rooms keep their stored order; each dict is ready for ``render_office``.
    """
    order = {"running": 0, "completed": 1, "failed": 2}
    by_room: dict[str, dict] = {}
    for d in dispatches:
        rid = getattr(d, "room_id", None)
        if not rid:
            continue
        members = by_room.setdefault(rid, {})
        prev = members.get(d.character_id)
        if prev is None or order.get(d.status, 9) < order.get(prev["status"], 9):
            members[d.character_id] = {"cid": d.character_id, "status": d.status}
    out = []
    for room in store_rooms:
        mem_list = list(by_room.get(room.id, {}).values())
        running = sum(1 for m in mem_list if m["status"] == "running")
        out.append({"id": room.id, "name": room.name, "preset": room.preset,
                    "rules": room.rules, "members": mem_list, "running": running,
                    "connections": list(getattr(room, "connections", ()) or ()),
                    "default_model": getattr(room, "default_model", ""),
                    "node_type": getattr(room, "node_type", "agent"),
                    "x": getattr(room, "x", 40.0), "y": getattr(room, "y", 40.0),
                    "content": getattr(room, "content", ""),
                    "panel": list(getattr(room, "panel", ()) or [])})
    return out


def _member_html(member: dict, char_by_id: dict) -> str:
    c = char_by_id.get(member["cid"])
    name = c.name if c else member["cid"]
    avatar = c.avatar if c else ""
    status = member["status"]
    icon = _MEMBER_STATUS_ICON.get(status, "")
    return (f'<div class="member st-{_esc(status)}" title="{_esc(name)} — {_esc(status)}">'
            f'<div class="m-av">{avatar}<span class="m-badge">{icon}</span></div>'
            f'<span class="m-name">{_esc(name)}</span></div>')


def render_room_editor(room=None, presets=(), all_rooms=(), models=()) -> str:
    """Create/edit form for a room: name, role preset, custom rules, downstream
    connections (Phase 2 routing), and a default model for hand-offs (Phase 3).

    ``room`` is a Room (edit) or None (create). ``presets`` = RoomPreset objects.
    ``all_rooms`` = every Room (for the "connects to" picker; self is excluded).
    ``models`` = (id, name) pairs for the default-model picker. Submits to
    ``/api/rooms`` (create) or ``/api/rooms/{id}`` (update).
    """
    editing = room is not None
    name = _esc(room.name) if editing else ""
    rules = _esc(room.rules) if editing else ""
    cur_preset = room.preset if editing else "custom"
    cur_conns = set(getattr(room, "connections", ()) or ()) if editing else set()
    cur_model = getattr(room, "default_model", "") if editing else ""
    options = "".join(
        f'<option value="{_esc(p.key)}"{" selected" if p.key == cur_preset else ""}>{_esc(p.name)}</option>'
        for p in presets
    )
    model_opts = '<option value="">Auto-cast (pick best free model)</option>' + "".join(
        f'<option value="{_esc(mid)}"{" selected" if mid == cur_model else ""}>{_esc(mname)}</option>'
        for mid, mname in models
    )
    model_field = (f'<label>Default model (hand-offs)<select name="default_model">{model_opts}</select></label>'
                   if models else "")
    targets = [r for r in all_rooms if not (editing and r.id == room.id)]
    if targets:
        boxes = "".join(
            f'<label class="re-conn"><input type="checkbox" name="connections" value="{_esc(t.id)}"'
            f'{" checked" if t.id in cur_conns else ""}> {_esc(t.name)}</label>'
            for t in targets
        )
        connections = (f'<fieldset class="re-conns"><legend>Hand off to (routing)</legend>{boxes}'
                       f'<span class="re-hint">On completion, output flows to these rooms. '
                       f'A Decision-Maker room routes to just one; others broadcast to all.</span></fieldset>')
    else:
        connections = '<div class="re-hint">Create more rooms to connect this one for routing.</div>'
    action = f'/api/rooms/{_esc(room.id)}' if editing else '/api/rooms'
    delete = ""
    if editing:
        delete = (f'<button type="button" class="re-del" hx-post="/api/rooms/{_esc(room.id)}/delete" '
                  f'hx-target="#fleet" hx-swap="innerHTML" '
                  f'hx-on:click="document.getElementById(\'room-editor\').innerHTML=\'\'">Delete</button>')
    return (
        f'<form class="room-editor" hx-post="{action}" hx-target="#fleet" hx-swap="innerHTML" '
        f'hx-on:htmx:after-request="if(event.detail.successful)this.closest(\'#room-editor\').innerHTML=\'\'">'
        f'<div class="re-title">{"Edit room" if editing else "New room"}</div>'
        f'<label>Name<input name="name" value="{name}" placeholder="e.g. Research Room" required></label>'
        f'<label>Role preset<select name="preset">{options}</select></label>'
        f'{model_field}'
        f'<label>Rules / guidelines<textarea name="rules" rows="4" '
        f'placeholder="Room-specific rules the model must follow…">{rules}</textarea></label>'
        f'{connections}'
        f'<div class="re-actions"><button type="submit">Save</button>{delete}'
        f'<button type="button" class="re-cancel" '
        f'hx-on:click="document.getElementById(\'room-editor\').innerHTML=\'\'">Cancel</button></div></form>'
    )


def render_workspace_tabs(workspaces, active_id: str) -> str:
    """Workspace switcher: one tab per workspace + new/rename/delete controls."""
    if not workspaces:
        return ""
    tabs = []
    for w in workspaces:
        active = w.id == active_id
        tabs.append(
            f'<button class="ws-tab{" active" if active else ""}" '
            f'hx-post="/api/workspaces/{_esc(w.id)}/activate" hx-target="#fleet" hx-swap="innerHTML">'
            f'{_esc(w.name)}</button>'
        )
    tools = ('<button class="ws-new" title="New workspace" hx-post="/api/workspaces" '
             'hx-target="#fleet" hx-swap="innerHTML">＋</button>')
    if active_id:
        tools += (f'<button class="ws-edit" title="Rename workspace" '
                  f'hx-get="/api/workspaces/{_esc(active_id)}/edit" hx-target="#room-editor" '
                  f'hx-swap="innerHTML">✎</button>')
        if len(workspaces) > 1:
            tools += (f'<button class="ws-del" title="Delete this workspace + its rooms" '
                      f'hx-post="/api/workspaces/{_esc(active_id)}/delete" hx-target="#fleet" '
                      f'hx-swap="innerHTML" hx-confirm="Delete this workspace and all its rooms?">✕</button>')
    return f'<div class="ws-tabs">{"".join(tabs)}<span class="ws-tools">{tools}</span></div>'


def render_workspace_editor(workspace) -> str:
    """Rename form for a workspace (loaded into #room-editor)."""
    return (
        f'<form class="room-editor" hx-post="/api/workspaces/{_esc(workspace.id)}" '
        f'hx-target="#fleet" hx-swap="innerHTML" '
        f'hx-on:htmx:after-request="if(event.detail.successful)this.closest(\'#room-editor\').innerHTML=\'\'">'
        f'<div class="re-title">Rename workspace</div>'
        f'<label>Name<input name="name" value="{_esc(workspace.name)}" required></label>'
        f'<div class="re-actions"><button type="submit">Save</button>'
        f'<button type="button" class="re-cancel" '
        f'hx-on:click="document.getElementById(\'room-editor\').innerHTML=\'\'">Cancel</button></div></form>'
    )


_NODE_TYPE_LABEL = {
    "start": "START", "agent": "AGENT", "prompt": "PROMPT", "output": "OUTPUT",
    "discussion": "DISCUSSION", "consensus": "CONSENSUS", "judge": "JUDGE",
    "gather": "GATHER", "condition": "CONDITION", "loop": "LOOP",
}
# palette buttons: (node_type, label) — order shown in the toolbar
_NODE_PALETTE = [
    ("start", "Start"), ("agent", "Agent"), ("prompt", "Prompt"), ("output", "Output"),
    ("discussion", "Discussion"), ("consensus", "Consensus"), ("judge", "Judge"),
    ("gather", "Gather"), ("condition", "Condition"), ("loop", "Loop"),
]
_PANEL_NODES = {"discussion", "consensus"}       # nodes that hold several models
_NODE_GLYPH = {
    "start": "▶", "agent": "🤖", "prompt": "📝", "output": "⎯", "discussion": "💬",
    "consensus": "🤝", "judge": "⚖", "gather": "⑃", "condition": "◇", "loop": "↻",
}
_NODE_TIP = {
    "start": "Graph entry — type the task; Play starts here",
    "agent": "One model runs the input under a role + rules",
    "prompt": "Extra instructions injected into connected agents",
    "output": "Terminal — shows the final result",
    "discussion": "A panel of models debate, then synthesise a consensus",
    "consensus": "A panel answers in parallel, then merges into one",
    "judge": "Critiques the input against criteria (PASS / REVISE)",
    "gather": "Merges several incoming branches into one",
    "condition": "Routes the input to one connected node by a rule",
    "loop": "Repeats the downstream nodes N times",
}


def _clip(text: str, n: int = 90) -> str:
    text = (text or "").strip()
    return _esc(text[:n] + "…") if len(text) > n else _esc(text)


def _node_card(node: dict, char_by_id: dict, names: dict, presets=(), models=()) -> str:
    """One blueprint node positioned on the canvas, edited INLINE (no bottom panel).

    Fields are live controls that POST their own value to ``/api/nodes/{id}`` on change
    (``hx-swap="none"`` so the canvas isn't re-rendered mid-edit). Connections are made by
    dragging output→input ports.
    """
    nid = _esc(node["id"])
    ntype = node.get("node_type", "agent")
    x, y = float(node.get("x", 40.0)), float(node.get("y", 40.0))
    conns = [c for c in node.get("connections", []) if c in names]
    conn_attr = f' data-connections="{_esc(",".join(conns))}"' if conns else ""
    # inline fields save via a plain-JS change→fetch handler in fleet.js (reliable; the
    # htmx path raced with background re-renders). ``data-nf`` marks a node field.
    inl = 'data-nf'
    in_port = ('' if ntype == "start" else
               f'<span class="port in" data-port="in" data-node="{nid}" title="drop a connection here"></span>')
    out_port = ('' if ntype == "output" else
                f'<span class="port out" data-port="out" data-node="{nid}" title="drag to connect →"></span>')

    def _model_select() -> str:
        cur = node.get("default_model", "")
        opts = '<option value="">Auto-cast</option>' + "".join(
            f'<option value="{_esc(mid)}"{" selected" if mid == cur else ""}>{_esc(mn)}</option>'
            for mid, mn in models)
        return f'<select class="ni ni-model" name="default_model" {inl}>{opts}</select>'

    def _rules(ph: str) -> str:
        return (f'<textarea class="ni ni-rules" name="rules" rows="2" {inl} '
                f'placeholder="{_esc(ph)}">{_esc(node.get("rules", ""))}</textarea>')

    def _panel_chips() -> str:
        chips = "".join(
            f'<span class="panel-chip">'
            f'{_esc(char_by_id[cid].name if cid in char_by_id else cid)}'
            f'<button class="pc-x" title="Remove" data-panel-remove="{_esc(cid)}">✕</button></span>'
            for cid in node.get("panel", []))
        return (f'<div class="node-panel">{chips or "<span class=\'nm-empty\'>drop models here</span>"}</div>')

    if ntype == "start":
        body = (f'<textarea class="ni ni-task" name="content" rows="2" {inl} '
                f'placeholder="▶ enter a task…">{_esc(node.get("content", ""))}</textarea>')
    elif ntype == "prompt":
        body = (f'<textarea class="ni ni-prompt" name="content" rows="2" {inl} '
                f'placeholder="extra prompt to inject…">{_esc(node.get("content", ""))}</textarea>')
    elif ntype == "output":
        out = _clip(node.get("content", ""), 200)
        body = f'<div class="node-output">{out or "(waiting for result)"}</div>'
    elif ntype in ("discussion", "consensus"):
        rounds = (f'<label class="ni-lbl">rounds<input class="ni ni-rounds" name="content" type="number" '
                  f'min="1" max="5" {inl} value="{_esc(node.get("content", "") or "2")}"></label>'
                  if ntype == "discussion" else "")
        # the panel chips ARE the participants; they pulse while the node runs (all models engage
        # in parallel), so no separate single-avatar row.
        body = f'{_panel_chips()}{rounds}'
    elif ntype in ("judge", "condition"):
        ph = "review criteria…" if ntype == "judge" else "routing rules… (model ends with ROUTE: <node>)"
        members = "".join(_member_html(m, char_by_id) for m in node.get("members", []))
        body = (f'{_model_select()}{_rules(ph)}'
                f'<div class="node-members">{members or "<span class=\'nm-empty\'>idle</span>"}</div>')
    elif ntype == "loop":
        body = (f'<label class="ni-lbl">repeat ×<input class="ni ni-loop" name="content" type="number" '
                f'min="1" max="10" {inl} value="{_esc(node.get("content", "") or "2")}"></label>')
    elif ntype == "gather":
        out = _clip(node.get("content", ""), 200)
        body = f'<div class="node-output">{out or "(merges incoming branches)"}</div>'
    else:  # agent — inline preset + model + rules
        cur_preset = node.get("preset", "custom")
        preset_opts = "".join(
            f'<option value="{_esc(p.key)}"{" selected" if p.key == cur_preset else ""}>{_esc(p.name)}</option>'
            for p in presets) or f'<option>{_esc(_preset_label(cur_preset))}</option>'
        router = ('<span class="room-router" title="Routes to one connected node">⤳ router</span>'
                  if cur_preset == "decision_maker" else '')
        members = "".join(_member_html(m, char_by_id) for m in node.get("members", []))
        body = (
            f'<select class="ni ni-preset" name="preset" {inl}>{preset_opts}</select>{router}'
            f'{_model_select()}{_rules("rules / guidelines…")}'
            f'<div class="node-members">{members or "<span class=\'nm-empty\'>idle</span>"}</div>'
        )
    running = " running" if node.get("running") else ""
    return (
        f'<div class="node nt-{_esc(ntype)}{running}" data-node="{nid}" data-node-type="{_esc(ntype)}"'
        f'{conn_attr} style="left:{x:.0f}px;top:{y:.0f}px">'
        f'<div class="node-head"><span class="node-glyph">{_NODE_GLYPH.get(ntype, "")}</span>'
        f'<span class="node-type-badge">{_NODE_TYPE_LABEL.get(ntype, "")}</span>'
        f'<input class="node-title-in" name="name" value="{_esc(node.get("name", ""))}" {inl}>'
        f'<button class="node-del" title="Delete" hx-post="/api/nodes/{nid}/delete" '
        f'hx-target="#fleet" hx-swap="innerHTML" hx-confirm="Delete this node?">✕</button></div>'
        f'{in_port}{out_port}<div class="node-body">{body}</div></div>'
    )


def render_office(roster, rooms, *, dispatch_enabled: bool, notice: Optional[str] = None,
                  workspaces=(), active_ws: str = "", presets=(), models=()) -> str:
    """Blueprint graph canvas for the active workspace: draggable Start/Agent/Prompt/Output
    nodes wired port-to-port, edited inline, plus a Play button and a model roster.

    ``roster`` = list of (Character, presence-state). ``rooms`` = ``rooms_with_members``
    output for the workspace's nodes (each carries node_type/x/y/content/connections/members).
    ``presets``/``models`` feed the inline preset + default-model dropdowns on agent nodes.
    """
    char_by_id = {c.id: c for c, _s in roster}
    banner = f'<div class="cast-notice">{_esc(notice)}</div>' if notice else ""
    if not dispatch_enabled:
        banner += _DISPATCH_HINT
    tabs = render_workspace_tabs(workspaces, active_ws)

    names = {n["id"]: n["name"] for n in rooms}
    node_cards = "".join(_node_card(n, char_by_id, names, presets, models) for n in rooms)
    wires = '<svg class="node-wires" aria-hidden="true"></svg>'
    # empty-state guidance overlaid on a blank canvas
    empty = ('' if rooms else
             '<div class="graph-empty"><b>Build a flow</b>'
             '<span>Add a <b>Start</b> node, then an <b>Agent</b>, then an <b>Output</b>.</span>'
             '<span>Drag a node’s <b>●</b> output port to the next node’s <b>●</b> input to wire them.</span>'
             '<span>Press <b>▶ Play</b> to run.</span></div>')
    node_btns = "".join(
        f'<button class="np-add np-{t}" title="{_esc(_NODE_TIP.get(t, ""))}" '
        f'hx-post="/api/nodes" hx-vals=\'{{"type":"{t}"}}\' hx-target="#fleet" hx-swap="innerHTML">'
        f'{_NODE_GLYPH.get(t, "")} {_esc(label)}</button>'
        for t, label in _NODE_PALETTE
    )
    palette = (
        f'<div class="node-palette"><div class="np-nodes">{node_btns}</div>'
        '<div class="np-actions"><span class="graph-status" aria-live="polite"></span>'
        '<button class="graph-play" hx-post="/api/graph/play" '
        'hx-target="#fleet" hx-swap="innerHTML" title="Run the graph from the Start node">'
        '▶ Play</button></div></div>'
        '<div class="graph-legend">Drag <b>●</b>→<b>●</b> to connect · click a wire to delete · '
        'drag a node header to move · drop a model onto an agent</div>'
    )

    # active models first, inactive/offline sink to the bottom (dynamic availability)
    ordered = sorted(enumerate(roster), key=lambda p: (_ROSTER_STATE_ORDER.get(p[1][1], 5), p[0]))
    chips = []
    inactive_started = False
    for _i, (c, state) in ordered:
        cid = _esc(c.id)
        label = _ROSTER_STATE_LABEL.get(state, state)
        if state == "offline" and not inactive_started:
            inactive_started = True
            chips.append('<div class="roster-sep">Inactive</div>')
        chips.append(
            f'<div class="roster-chip st-{_esc(state)}" draggable="true" data-cid="{cid}" '
            f'hx-get="/api/character/{cid}" hx-target="#chat" hx-swap="innerHTML" '
            f'role="button" tabindex="0" title="{_esc(c.name)} — {_esc(c.role)}">'
            f'<div class="rc-av">{c.avatar}</div>'
            f'<div class="rc-meta"><span class="rc-name">{_esc(c.name)}</span>'
            f'<span class="rc-role">{_esc(c.role)}</span></div>'
            f'<span class="rc-dot" title="{_esc(label)}"></span></div>'
        )
    roster_html = (f'<aside class="roster"><div class="roster-head">Models '
                   f'<span class="roster-hint">drag → an agent node</span></div>'
                   f'{"".join(chips)}</aside>')

    return (f'{banner}{tabs}{palette}<div class="office graph">'
            f'<div class="graph-canvas" data-graph>{wires}{empty}{node_cards}</div>'
            f'{roster_html}</div>')


def render_node_editor(node, presets=(), all_nodes=(), models=()) -> str:
    """Edit form for a node, varying by type. Posts to ``/api/nodes/{id}``."""
    nid = _esc(node.id)
    ntype = getattr(node, "node_type", "agent")
    action = f'/api/nodes/{nid}'
    head = (
        f'<form class="room-editor" hx-post="{action}" hx-target="#fleet" hx-swap="innerHTML" '
        f'hx-on:htmx:after-request="if(event.detail.successful)this.closest(\'#room-editor\').innerHTML=\'\'">'
        f'<div class="re-title">Edit {_esc(_NODE_TYPE_LABEL.get(ntype, "").title())} node</div>'
        f'<label>Name<input name="name" value="{_esc(node.name)}" required></label>'
    )
    fields = ""
    if ntype == "start":
        fields = (f'<label>Task<textarea name="content" rows="4" '
                  f'placeholder="What should the graph do…">{_esc(node.content)}</textarea></label>')
    elif ntype == "prompt":
        fields = (f'<label>Prompt (injected into connected agents)<textarea name="content" rows="4" '
                  f'placeholder="Extra guidance to inject…">{_esc(node.content)}</textarea></label>')
    elif ntype == "agent":
        cur_preset = getattr(node, "preset", "custom")
        options = "".join(
            f'<option value="{_esc(p.key)}"{" selected" if p.key == cur_preset else ""}>{_esc(p.name)}</option>'
            for p in presets)
        model_opts = '<option value="">Auto-cast</option>' + "".join(
            f'<option value="{_esc(mid)}"{" selected" if mid == getattr(node, "default_model", "") else ""}>{_esc(mn)}</option>'
            for mid, mn in models)
        fields = (
            f'<label>Role preset<select name="preset">{options}</select></label>'
            f'<label>Default model<select name="default_model">{model_opts}</select></label>'
            f'<label>Rules<textarea name="rules" rows="3">{_esc(node.rules)}</textarea></label>'
        )
    # output nodes: name only (content is filled by execution)
    tail = ('<div class="re-actions"><button type="submit">Save</button>'
            '<button type="button" class="re-cancel" '
            'hx-on:click="document.getElementById(\'room-editor\').innerHTML=\'\'">Cancel</button></div></form>')
    return head + fields + tail


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

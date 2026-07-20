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
        f'<div class="char-card st-{_esc(state)}">'
        f'{char.avatar}'
        f'<div class="char-head"><span class="char-name">{_esc(char.name)}</span>'
        f'<span class="char-state">{_esc(state)}</span></div>'
        f'<div class="char-role">{_esc(char.role)} · {_esc(char.provider)} · {_esc(char.model)}</div>'
        f'<button class="chat-btn" hx-get="/api/character/{rid}" hx-target="#chat" hx-swap="innerHTML">Chat</button>'
        f'{dispatch}</div>'
    )


def render_fleet(items, *, dispatch_enabled: bool, notice: Optional[str] = None) -> str:
    """Render the whole fleet from a list of (Character, state) pairs.

    ``notice`` shows an auto-cast explanation banner ("picked X because …").
    """
    banner = f'<div class="cast-notice">{_esc(notice)}</div>' if notice else ""
    if not items:
        return f'{banner}<div class="fleet empty">No models configured</div>'
    cards = "".join(render_character_card(c, s, dispatch_enabled=dispatch_enabled) for c, s in items)
    return f'{banner}<div class="fleet">{cards}</div>'


def render_character_detail(char, *, chat_enabled: bool) -> str:
    """Character detail panel: persona header + a chat conversation area."""
    rid = _esc(char.id)
    if chat_enabled:
        form = (
            f'<form class="chat-form" hx-post="/api/character/{rid}/chat" hx-target="#chat-log" '
            f'hx-swap="beforeend" hx-on::after-request="this.reset()">'
            f'<input class="task-in" name="message" placeholder="Message {_esc(char.name)}…" autocomplete="off">'
            '<button>Send</button></form>'
        )
    else:
        form = '<div class="disabled">chat disabled — enable dashboard.dispatch_enabled</div>'
    return (
        '<div class="char-detail">'
        f'<div class="cd-head">{char.avatar}<div><span class="char-name">{_esc(char.name)}</span>'
        f'<div class="char-role">{_esc(char.role)} · {_esc(char.provider)} · {_esc(char.model)}</div></div></div>'
        '<div id="chat-log" class="chat-log"></div>'
        f'{form}</div>'
    )


def render_chat_exchange(name: str, message: str, reply: str) -> str:
    """One user message + the character's reply, appended to the chat log."""
    return (
        f'<div class="chat-msg you"><span class="who">you</span> {_esc(message)}</div>'
        f'<div class="chat-msg them"><span class="who">{_esc(name)}</span> {_esc(reply)}</div>'
    )

from __future__ import annotations
"""Live dashboard aiohttp app: HTMX-driven panels + SSE event stream.

Server-renders HTML partials (see ``dashboard_render``) that HTMX swaps in, and
streams live run/agent events over SSE bridged from the in-process event bus.
Reuses the pure ``web_dashboard.compute_*`` aggregators (kept untouched).
"""
import asyncio
import logging
from typing import Optional

from aiohttp import web

from orchestra.monitoring import dashboard_render as render
from orchestra.monitoring import web_dashboard

logger = logging.getLogger(__name__)

_HTML = lambda body: web.Response(text=body, content_type="text/html")  # noqa: E731


# ── DB access helpers (thin, monkeypatchable in tests) ───────────────────────

def _runs_rows() -> list[dict]:
    """Most recent runs for the overview table."""
    from orchestra.storage.db import get_db
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT run_id, mode, status, total_cost_usd, created_at "
            "FROM runs ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        logger.warning("_runs_rows failed", exc_info=True)
        return []
    finally:
        conn.close()


def _incident_counts() -> dict:
    conn = web_dashboard._events_conn()
    try:
        return web_dashboard.count_incidents(conn)
    finally:
        conn.close()


# ── Shell page ───────────────────────────────────────────────────────────────

_SHELL = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><title>Orchestra — Live Dashboard</title>
<link rel="stylesheet" href="/static/style.css">
<script src="https://unpkg.com/htmx.org@1.9.12"></script>
<script src="https://unpkg.com/htmx.org@1.9.12/dist/ext/sse.js"></script>
</head><body>
<h1>Orchestra</h1>
<section class="panels">
  <div class="panel" hx-get="/api/metrics" hx-trigger="load, every 10s"></div>
  <div class="panel" hx-get="/api/budget" hx-trigger="load, every 10s"></div>
  <div class="panel" hx-get="/api/eval" hx-trigger="load, every 30s"></div>
  <div class="panel" hx-get="/api/incidents" hx-trigger="load, every 30s"></div>
</section>
<section class="panel" id="hitl" hx-get="/api/paused" hx-trigger="load, every 15s"></section>
<section class="live" hx-ext="sse" sse-connect="/api/stream">
  <h2>Live feed</h2>
  <div id="feed" sse-swap="run_update" hx-swap="afterbegin"></div>
</section>
<section class="split">
  <div class="panel runs-panel" hx-get="/api/runs" hx-trigger="load, every 10s"></div>
  <div class="panel" id="drill">Select a run…</div>
</section>
</body></html>"""


# ── Panel handlers ───────────────────────────────────────────────────────────

async def _index(request: web.Request) -> web.Response:
    return _HTML(_serve_shell())


def _serve_shell() -> str:
    """Serve the packaged index.html when present, else the inline shell."""
    try:
        from pathlib import Path
        p = Path(__file__).parent / "dashboard" / "index.html"
        if p.exists():
            return p.read_text(encoding="utf-8")
    except Exception:
        logger.warning("failed reading index.html; using inline shell", exc_info=True)
    return _SHELL


async def _metrics(request: web.Request) -> web.Response:
    return _HTML(render.render_metrics(web_dashboard.compute_metrics()))


async def _incidents(request: web.Request) -> web.Response:
    return _HTML(render.render_incidents(_incident_counts()))


async def _eval(request: web.Request) -> web.Response:
    return _HTML(render.render_eval(web_dashboard.compute_eval_summary()))


async def _runs(request: web.Request) -> web.Response:
    return _HTML(render.render_runs_table(_runs_rows()))


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", _index)
    app.router.add_get("/api/metrics", _metrics)
    app.router.add_get("/api/incidents", _incidents)
    app.router.add_get("/api/eval", _eval)
    app.router.add_get("/api/runs", _runs)
    return app


def run_app(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Run the live dashboard (blocking)."""
    web.run_app(make_app(), host=host, port=port)

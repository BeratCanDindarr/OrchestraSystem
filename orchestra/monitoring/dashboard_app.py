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

from orchestra import config, service
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


# ── HITL approvals + budget ──────────────────────────────────────────────────

def _paused_html() -> str:
    return render.render_paused_cards(
        web_dashboard.list_paused_runs(), enabled=config.dashboard_approvals_enabled()
    )


async def _paused(request: web.Request) -> web.Response:
    return _HTML(_paused_html())


async def _approve(request: web.Request) -> web.Response:
    if not config.dashboard_approvals_enabled():
        return web.Response(text="approvals disabled", status=403)
    data = await request.post()
    run_id = str(data.get("run_id", ""))
    decision = str(data.get("decision", "approve"))
    if run_id:
        try:
            web_dashboard.apply_approval(run_id, decision)
        except Exception:
            logger.warning("apply_approval failed for %s", run_id, exc_info=True)
    return _HTML(_paused_html())


async def _budget(request: web.Request) -> web.Response:
    return _HTML(render.render_budget(web_dashboard.compute_budget()))


# ── Run drill-down ───────────────────────────────────────────────────────────

def _run_events(run_id: str) -> list[dict]:
    """Seq-ordered event timeline for a run from the durable event log."""
    try:
        from orchestra.storage.event_log import EventLog
        return [e.to_dict() for e in EventLog().replay(run_id)]
    except Exception:
        logger.warning("_run_events failed for %s", run_id, exc_info=True)
        return []


def _trace_spans(run_id: str) -> list[dict]:
    """Best-effort OTel span list for a run (empty when tracing/in-memory unavailable)."""
    try:
        from orchestra.engine.tracer import OtelTracer
        tracer = OtelTracer.get_instance()
        exporter = getattr(tracer, "_exporter", None)
        if exporter is None or not hasattr(exporter, "get_spans"):
            return []
        spans = []
        for s in exporter.get_spans():
            attrs = dict(getattr(s, "attributes", {}) or {})
            if attrs.get("run_id") not in (None, run_id):
                continue
            spans.append({"name": getattr(s, "name", ""),
                          "total_latency_ms": attrs.get("total_latency_ms", 0.0)})
        return spans
    except Exception:
        logger.warning("_trace_spans failed for %s", run_id, exc_info=True)
        return []


async def _run_detail(request: web.Request) -> web.Response:
    rid = request.match_info["rid"]
    try:
        run = (service.get_run(rid) or {}).get("run") or {"run_id": rid, "status": "unknown", "agents": []}
    except Exception:
        logger.warning("get_run failed for %s", rid, exc_info=True)
        run = {"run_id": rid, "status": "unknown", "agents": []}
    return _HTML(render.render_run_detail(run))


async def _run_events_handler(request: web.Request) -> web.Response:
    return _HTML(render.render_timeline(_run_events(request.match_info["rid"])))


async def _run_trace_handler(request: web.Request) -> web.Response:
    return _HTML(render.render_trace(_trace_spans(request.match_info["rid"])))


# ── SSE live stream ──────────────────────────────────────────────────────────

def _make_bus_handler(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
    """Return a bus handler that forwards events onto ``queue`` from any thread.

    The event bus dispatches handlers on a thread pool, so we hop back onto the
    asyncio loop with ``call_soon_threadsafe`` before touching the queue.
    """
    def handler(run_id: str, event: dict) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, (run_id, event))
        except asyncio.QueueFull:
            pass  # drop under backpressure rather than block the bus thread
    return handler


async def _stream(request: web.Request) -> web.StreamResponse:
    from orchestra.events import bus
    resp = web.StreamResponse(headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    })
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    handler = _make_bus_handler(loop, queue)
    bus.subscribe_all(handler)
    try:
        await resp.prepare(request)
        while True:
            run_id, event = await queue.get()
            await resp.write(render.sse_pack("run_update", render.render_run_update(run_id, event)))
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    except Exception:
        logger.warning("SSE stream error", exc_info=True)
    finally:
        bus.unsubscribe_all(handler)
    return resp


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", _index)
    app.router.add_get("/api/metrics", _metrics)
    app.router.add_get("/api/incidents", _incidents)
    app.router.add_get("/api/eval", _eval)
    app.router.add_get("/api/runs", _runs)
    app.router.add_get("/api/runs/{rid}", _run_detail)
    app.router.add_get("/api/runs/{rid}/events", _run_events_handler)
    app.router.add_get("/api/runs/{rid}/trace", _run_trace_handler)
    app.router.add_get("/api/stream", _stream)
    app.router.add_get("/api/paused", _paused)
    app.router.add_post("/api/approve", _approve)
    app.router.add_get("/api/budget", _budget)
    return app


def run_app(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Run the live dashboard (blocking)."""
    web.run_app(make_app(), host=host, port=port)

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
from orchestra.monitoring import characters
from orchestra.monitoring import dashboard_render as render
from orchestra.monitoring import presence as presence_mod
from orchestra.monitoring import web_dashboard

logger = logging.getLogger(__name__)

_HTML = lambda body: web.Response(text=body, content_type="text/html")  # noqa: E731
_SSE_HEARTBEAT_SECS = 15.0  # idle-connection keep-alive / drop detection on the SSE stream


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


def _search_rows() -> list[dict]:
    """Recent runs including task text, for the global search index."""
    from orchestra.storage.db import get_db
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT run_id, mode, status, total_cost_usd, created_at, task "
            "FROM runs ORDER BY created_at DESC LIMIT 500"
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        logger.warning("_search_rows failed", exc_info=True)
        return []
    finally:
        conn.close()


def _audit_rows(limit: int = 80) -> list[dict]:
    """Most recent events across all runs (append-only log) for the audit view."""
    conn = web_dashboard._events_conn()
    try:
        rows = conn.execute(
            "SELECT event_type, run_id, ts, seq FROM events ORDER BY seq DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.debug("_audit_rows skipped: %s", exc)   # missing events table is benign
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


async def _charts(request: web.Request) -> web.Response:
    """Cost/latency/volume time series aggregated from recent runs."""
    return _HTML(render.render_charts(render.aggregate_timeseries(_runs_rows())))


async def _search(request: web.Request) -> web.Response:
    """Global run search — matches task/id/mode/status; ``field:value`` supported."""
    q = request.query.get("q", "")
    return _HTML(render.render_search_results(render.filter_runs(_search_rows(), q), q))


async def _audit(request: web.Request) -> web.Response:
    """Recent append-only event log entries."""
    return _HTML(render.render_audit(_audit_rows()))


async def _flow(request: web.Request) -> web.Response:
    """Room-to-room routing hand-off history."""
    from orchestra.monitoring import room_router
    return _HTML(render.render_flow(room_router.recent_flows(), _inbox_names()))


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


# ── Fleet: models as characters + task dispatch ──────────────────────────────

def _fleet_items() -> list:
    """(Character, presence-state) pairs for the whole roster."""
    roster = characters.build_roster()
    states = presence_mod.get_presence().snapshot([(c.id, c.available) for c in roster])
    return [(c, states.get(c.id, "idle")) for c in roster]


async def _fleet(request: web.Request) -> web.Response:
    return _HTML(render.render_fleet(
        _fleet_items(), dispatch_enabled=config.dashboard_dispatch_enabled()
    ))


def _active_workspace():
    """Ensure a workspace exists and orphan rooms are adopted; return the active one."""
    from orchestra.monitoring.room_store import get_room_store
    from orchestra.monitoring.workspace_store import get_workspace_store
    ws = get_workspace_store().ensure_default()
    store = get_room_store()
    orphans = [r for r in store.list() if not r.workspace_id]
    for r in orphans:                       # migrate pre-workspace rooms into the active one
        store.update(r.id, workspace_id=ws.id)
    return ws


def _office_html(*, notice: Optional[str] = None) -> str:
    from orchestra.monitoring.dispatch_inbox import get_inbox
    from orchestra.monitoring.room_store import get_room_store
    from orchestra.monitoring.workspace_store import get_workspace_store
    active = _active_workspace()
    roster = _fleet_items()
    char_by_id = {c.id: c for c, _s in roster}
    ws_rooms = [r for r in get_room_store().list() if r.workspace_id == active.id]
    rooms = render.rooms_with_members(ws_rooms, get_inbox().recent(200), char_by_id)
    return render.render_office(
        roster, rooms,
        dispatch_enabled=config.dashboard_dispatch_enabled(),
        notice=notice,
        workspaces=get_workspace_store().list(),
        active_ws=active.id,
    )


async def _office(request: web.Request) -> web.Response:
    return _HTML(_office_html())


# ── Task rooms: create / edit / delete (Phase 1) ─────────────────────────────

def _room_editor_html(room=None) -> str:
    from orchestra.monitoring import room_presets
    from orchestra.monitoring.room_store import get_room_store
    active = _active_workspace()
    # connection picker only lists rooms in the same workspace
    ws_id = room.workspace_id if room is not None else active.id
    peers = [r for r in get_room_store().list() if r.workspace_id == ws_id]
    models = [(c.id, c.name) for c in characters.build_roster()]
    return render.render_room_editor(room, presets=room_presets.preset_list(),
                                     all_rooms=peers, models=models)


# ── Workspaces: create / rename / delete / activate ──────────────────────────

async def _ws_create(request: web.Request) -> web.Response:
    from orchestra.monitoring.workspace_store import get_workspace_store
    get_workspace_store().create("New workspace")
    return _HTML(_office_html())


async def _ws_edit_form(request: web.Request) -> web.Response:
    from orchestra.monitoring.workspace_store import get_workspace_store
    ws = get_workspace_store().get(request.match_info["wid"])
    return _HTML(render.render_workspace_editor(ws) if ws else "")


async def _ws_rename(request: web.Request) -> web.Response:
    from orchestra.monitoring.workspace_store import get_workspace_store
    data = await request.post()
    get_workspace_store().rename(request.match_info["wid"], str(data.get("name", "")).strip())
    return _HTML(_office_html())


async def _ws_activate(request: web.Request) -> web.Response:
    from orchestra.monitoring.workspace_store import get_workspace_store
    get_workspace_store().set_active(request.match_info["wid"])
    return _HTML(_office_html())


async def _ws_delete(request: web.Request) -> web.Response:
    from orchestra.monitoring.room_store import get_room_store
    from orchestra.monitoring.workspace_store import get_workspace_store
    wid = request.match_info["wid"]
    if get_workspace_store().delete(wid):          # also drop the workspace's rooms
        store = get_room_store()
        for r in [r for r in store.list() if r.workspace_id == wid]:
            store.delete(r.id)
    return _HTML(_office_html())


async def _room_new(request: web.Request) -> web.Response:
    return _HTML(_room_editor_html())


async def _room_edit_form(request: web.Request) -> web.Response:
    from orchestra.monitoring.room_store import get_room_store
    room = get_room_store().get(request.match_info["rid"])
    if room is None:
        return _HTML("")
    return _HTML(_room_editor_html(room))


def _post_connections(data) -> list:
    return [str(c) for c in (data.getall("connections", []) if hasattr(data, "getall") else []) if c]


async def _room_create(request: web.Request) -> web.Response:
    from orchestra.monitoring.room_store import get_room_store
    data = await request.post()
    name = str(data.get("name", "")).strip()
    if name:
        store = get_room_store()
        room = store.create(name, preset=str(data.get("preset", "custom")),
                            rules=str(data.get("rules", "")),
                            workspace_id=_active_workspace().id)
        store.update(room.id, connections=_post_connections(data),
                     default_model=str(data.get("default_model", "")))
    return _HTML(_office_html())


async def _room_update(request: web.Request) -> web.Response:
    from orchestra.monitoring.room_store import get_room_store
    data = await request.post()
    get_room_store().update(request.match_info["rid"],
                            name=str(data.get("name", "")).strip() or "Untitled room",
                            preset=str(data.get("preset", "custom")),
                            rules=str(data.get("rules", "")),
                            connections=_post_connections(data),
                            default_model=str(data.get("default_model", "")))
    return _HTML(_office_html())


async def _room_delete(request: web.Request) -> web.Response:
    from orchestra.monitoring.room_store import get_room_store
    get_room_store().delete(request.match_info["rid"])
    return _HTML(_office_html())


def _status_data() -> dict:
    from orchestra import __version__
    from orchestra.monitoring.dispatch_inbox import get_inbox
    items = _fleet_items()
    return {
        "version": __version__,
        "total": len(items),
        "online": sum(1 for _c, s in items if s != "offline"),
        "busy": sum(1 for _c, s in items if s == "busy"),
        "active": sum(1 for d in get_inbox().recent() if d.status == "running"),
    }


async def _status(request: web.Request) -> web.Response:
    return _HTML(render.render_status(_status_data()))


def _health_data() -> list:
    from orchestra.providers.fallback import ALIAS_TABLE
    groups: dict = {}
    for _alias, (prov, _eff) in ALIAS_TABLE.items():
        name = getattr(prov, "name", type(prov).__name__)
        g = groups.setdefault(name, {"name": name, "total": 0, "online": 0})
        g["total"] += 1
        try:
            if prov.is_available():
                g["online"] += 1
        except Exception:
            pass
    return sorted(groups.values(), key=lambda x: x["name"])


async def _health(request: web.Request) -> web.Response:
    return _HTML(render.render_health(_health_data()))


def _settings_data() -> dict:
    return {
        "dispatch_enabled": config.dashboard_dispatch_enabled(),
        "approvals_enabled": config.dashboard_approvals_enabled(),
        "include_remote": config.fleet_include_remote_nodes(),
        "budget_usd": config.dashboard_budget_usd(),
    }


async def _settings(request: web.Request) -> web.Response:
    if request.method == "POST":
        import os
        data = await request.post()

        def _setbool(env_key: str, field: str) -> None:
            os.environ[env_key] = "true" if field in data else "false"

        _setbool("ORCHESTRA_DASHBOARD__DISPATCH_ENABLED", "dispatch_enabled")
        _setbool("ORCHESTRA_DASHBOARD__APPROVALS_ENABLED", "approvals_enabled")
        _setbool("ORCHESTRA_FLEET__INCLUDE_REMOTE_NODES", "include_remote")
        budget = data.get("budget_usd")
        if budget is not None:
            try:
                os.environ["ORCHESTRA_DASHBOARD__BUDGET_USD"] = str(float(str(budget)))
            except (TypeError, ValueError):
                pass
        config.reload_config()
    return _HTML(render.render_settings(_settings_data()))


def _office_states() -> dict:
    """Lightweight per-character state map (for animating existing sprites in place)."""
    from orchestra.monitoring.dispatch_inbox import get_inbox
    notify = get_inbox().unread_by_character()
    items = _fleet_items()
    out = {}
    for c, state in items:
        has_result = c.id in notify
        out[c.id] = {"state": state, "at_desk": state == "busy" or has_result,
                     "has_result": has_result}
    return out


async def _office_states_ep(request: web.Request) -> web.Response:
    return web.json_response(_office_states())


async def _dispatch(request: web.Request) -> web.Response:
    if not config.dashboard_dispatch_enabled():
        # 200 (not 403) so HTMX swaps the hint in — a silent 4xx looks like a dead button.
        return _HTML(_office_html())
    data = await request.post()
    character_id = str(data.get("character_id", "")).strip()
    task = str(data.get("task", "")).strip()
    room_id = str(data.get("room_id", "")).strip() or None
    notice: Optional[str] = None
    # A room contributes its role preset + rules as system context for the model.
    context = ""
    if room_id:
        from orchestra.monitoring import room_presets
        from orchestra.monitoring.room_store import get_room_store
        room = get_room_store().get(room_id)
        if room:
            context = room_presets.compose_context(room.preset, room.rules)
    # No character chosen → auto-cast: let the router pick the best free model.
    if task and (not character_id or character_id == "auto"):
        cast = characters.auto_cast(task, _fleet_items())
        if "error" in cast:
            notice = cast["error"]
            character_id = ""
        else:
            character_id = cast["character_id"]
            notice = f"Auto-cast → {cast['reason']}"
    if character_id and task:
        on_complete = None
        if room_id:                            # room dispatches can hand off to connected rooms
            from orchestra.monitoring import room_router
            on_complete = room_router.make_router(0)
        characters.dispatch_task(character_id, task, context=context, room_id=room_id,
                                 on_complete=on_complete)
    return _HTML(_office_html(notice=notice))


def _find_character(rid: str):
    return next((c for c in characters.build_roster() if c.id == rid), None)


def _inbox_names() -> dict:
    return {c.id: c.name for c in characters.build_roster()}


async def _inbox(request: web.Request) -> web.Response:
    from orchestra.monitoring.dispatch_inbox import get_inbox
    return _HTML(render.render_inbox(get_inbox().tray(), _inbox_names()))


async def _character_history(request: web.Request) -> web.Response:
    from orchestra.monitoring.dispatch_inbox import get_inbox
    rid = request.match_info["rid"]
    char = _find_character(rid)
    name = char.name if char else rid
    return _HTML(render.render_history(get_inbox().for_character(rid), name))


async def _council(request: web.Request) -> web.Response:
    if not config.dashboard_dispatch_enabled():
        return _HTML(_office_html())
    data = await request.post()
    task = str(data.get("task", "")).strip()
    ids = [i for i in (data.getall("character_id", []) if hasattr(data, "getall") else []) if i]
    if not ids:  # default council: the first 3 free models
        ids = [c.id for c, s in _fleet_items() if s in ("available", "idle")][:3]
    if task and ids:
        res = characters.dispatch_council(task, ids)
        notice = f"Council → sent to {res.get('count', 0)} models — compare their results in the inbox"
    else:
        notice = "Council needs a task and at least one available model"
    return _HTML(_office_html(notice=notice))


async def _dispatch_action(request: web.Request) -> web.Response:
    from orchestra.monitoring.dispatch_inbox import get_inbox
    did, action = request.match_info["did"], request.match_info["action"]
    box = get_inbox()
    {"pin": box.pin, "unpin": box.unpin, "archive": box.archive}.get(action, lambda _x: None)(did)
    return _HTML(render.render_inbox(box.tray(), _inbox_names()))


async def _dispatch_stream(request: web.Request) -> web.StreamResponse:
    """Stream a running dispatch's output live (polls the growing output, sends deltas)."""
    from orchestra.monitoring.dispatch_inbox import get_inbox
    did = request.match_info["did"]
    box = get_inbox()
    resp = web.StreamResponse(headers={
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive",
    })
    await resp.prepare(request)
    sent = 0
    try:
        while True:
            d = box.get(did)
            if d is None:
                break
            if len(d.output) > sent:
                await resp.write(render.sse_pack_stream("token", d.output[sent:]))
                sent = len(d.output)
            if d.status != "running":
                break
            await asyncio.sleep(0.15)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        try:
            await resp.write(render.sse_pack("done", ""))
        except Exception:
            pass
    return resp


async def _dispatch_output_view(request: web.Request) -> web.Response:
    from orchestra.monitoring.dispatch_inbox import get_inbox
    did = request.match_info["did"]
    box = get_inbox()
    d = box.get(did)
    if d is None:
        return web.Response(text="unknown dispatch", status=404)
    box.mark_read(did)  # reading the result marks it read → it drops from the tray
    name = _inbox_names().get(d.character_id, d.character_id)
    return _HTML(render.render_dispatch_output(d, name))


async def _character_detail(request: web.Request) -> web.Response:
    char = _find_character(request.match_info["rid"])
    if char is None:
        return web.Response(text="unknown character", status=404)
    return _HTML(render.render_character_detail(char, chat_enabled=config.dashboard_dispatch_enabled()))


async def _character_chat(request: web.Request) -> web.Response:
    rid = request.match_info["rid"]
    data = await request.post()
    message = str(data.get("message", "")).strip()
    char = _find_character(rid)
    name = char.name if char else rid
    if not config.dashboard_dispatch_enabled():
        # 200 with a reply bubble explaining how to enable — not a silent 403.
        return _HTML(render.render_chat_exchange(
            name, message,
            "Chat is disabled. Set dashboard.dispatch_enabled = true to talk to me."))
    res = characters.chat_with(rid, message)
    reply = str(res.get("reply") if res.get("ok") else res.get("error", "…"))
    return _HTML(render.render_chat_exchange(name, message, reply))


async def _character_chat_stream(request: web.Request) -> web.StreamResponse:
    """Stream a character's reply token-by-token over SSE (progressive chat)."""
    rid = request.match_info["rid"]
    message = request.query.get("message", "").strip()
    resp = web.StreamResponse(headers={
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive",
    })
    await resp.prepare(request)
    try:
        if not config.dashboard_dispatch_enabled():
            await resp.write(render.sse_pack(
                "token", "Chat is disabled. Set dashboard.dispatch_enabled = true to talk to me."))
        else:
            async for chunk in characters.chat_stream(rid, message):
                await resp.write(render.sse_pack("token", chunk))
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    except Exception:
        logger.warning("chat stream error", exc_info=True)
    finally:
        try:
            await resp.write(render.sse_pack("done", ""))
        except Exception:
            pass
    return resp


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
    def _put(item) -> None:
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            pass  # drop under backpressure rather than block the loop
    def handler(run_id: str, event: dict) -> None:
        # Runs on a bus thread; hop to the loop, and drop ON THE LOOP so a full
        # queue can't raise QueueFull in the loop's default exception handler.
        loop.call_soon_threadsafe(_put, (run_id, event))
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
            try:
                run_id, event = await asyncio.wait_for(queue.get(), timeout=_SSE_HEARTBEAT_SECS)
            except asyncio.TimeoutError:
                await resp.write(b": heartbeat\n\n")  # keep idle connections alive + detect drop
                continue
            await resp.write(render.sse_pack("run_update", render.render_run_update(run_id, event)))
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    except Exception:
        logger.warning("SSE stream error", exc_info=True)
    finally:
        bus.unsubscribe_all(handler)
    return resp


@web.middleware
async def _auth_middleware(request: web.Request, handler):
    """Optional HTTP Basic auth: when ``dashboard.auth_token`` is set, require it."""
    token = config.dashboard_auth_token()
    if token:
        import base64
        import hmac
        ok = False
        header = request.headers.get("Authorization", "")
        if header.startswith("Basic "):
            try:
                _user, _, pw = base64.b64decode(header[6:]).decode("utf-8", "ignore").partition(":")
                ok = hmac.compare_digest(pw.encode("utf-8"), token.encode("utf-8"))
            except Exception:
                ok = False
        if not ok:
            return web.Response(status=401, text="Authentication required",
                                headers={"WWW-Authenticate": 'Basic realm="Orchestra Dashboard"'})
    return await handler(request)


def make_app() -> web.Application:
    app = web.Application(middlewares=[_auth_middleware])
    app.router.add_get("/", _index)
    app.router.add_get("/api/metrics", _metrics)
    app.router.add_get("/api/incidents", _incidents)
    app.router.add_get("/api/charts", _charts)
    app.router.add_get("/api/search", _search)
    app.router.add_get("/api/audit", _audit)
    app.router.add_get("/api/flow", _flow)
    app.router.add_get("/api/eval", _eval)
    app.router.add_get("/api/runs", _runs)
    app.router.add_get("/api/runs/{rid}", _run_detail)
    app.router.add_get("/api/runs/{rid}/events", _run_events_handler)
    app.router.add_get("/api/runs/{rid}/trace", _run_trace_handler)
    app.router.add_get("/api/stream", _stream)
    app.router.add_get("/api/paused", _paused)
    app.router.add_post("/api/approve", _approve)
    app.router.add_get("/api/budget", _budget)
    app.router.add_get("/api/fleet", _fleet)
    app.router.add_get("/api/office", _office)
    app.router.add_post("/api/workspaces", _ws_create)
    app.router.add_get("/api/workspaces/{wid}/edit", _ws_edit_form)
    app.router.add_post("/api/workspaces/{wid}/activate", _ws_activate)
    app.router.add_post("/api/workspaces/{wid}/delete", _ws_delete)
    app.router.add_post("/api/workspaces/{wid}", _ws_rename)
    app.router.add_get("/api/rooms/new", _room_new)
    app.router.add_get("/api/rooms/{rid}/edit", _room_edit_form)
    app.router.add_post("/api/rooms", _room_create)
    app.router.add_post("/api/rooms/{rid}/delete", _room_delete)
    app.router.add_post("/api/rooms/{rid}", _room_update)
    app.router.add_get("/api/office/states", _office_states_ep)
    app.router.add_get("/api/status", _status)
    app.router.add_get("/api/health", _health)
    app.router.add_get("/api/settings", _settings)
    app.router.add_post("/api/settings", _settings)
    app.router.add_get("/api/inbox", _inbox)
    app.router.add_post("/api/dispatch", _dispatch)
    app.router.add_post("/api/council", _council)
    app.router.add_get("/api/dispatch/{did}", _dispatch_output_view)
    app.router.add_get("/api/dispatch/{did}/stream", _dispatch_stream)
    app.router.add_post("/api/dispatch/{did}/{action}", _dispatch_action)
    app.router.add_get("/api/character/{rid}/history", _character_history)
    app.router.add_get("/api/character/{rid}", _character_detail)
    app.router.add_post("/api/character/{rid}/chat", _character_chat)
    app.router.add_get("/api/character/{rid}/chat/stream", _character_chat_stream)
    from pathlib import Path
    static_dir = Path(__file__).parent / "dashboard"
    if static_dir.is_dir():
        app.router.add_static("/static/", static_dir)
    return app


def run_app(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Run the live dashboard (blocking)."""
    web.run_app(make_app(), host=host, port=port)

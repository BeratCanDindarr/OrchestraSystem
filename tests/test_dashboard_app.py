"""Tests for the live dashboard aiohttp app (Faz 6 Block B/D/E)."""
import pytest
from aiohttp.test_utils import TestClient, TestServer

from orchestra.monitoring import dashboard_app


async def _client():
    return TestClient(TestServer(dashboard_app.make_app()))


@pytest.mark.asyncio
async def test_index_shell_has_htmx_and_sse(monkeypatch):
    async with await _client() as client:
        resp = await client.get("/")
        assert resp.status == 200
        body = await resp.text()
    assert "htmx" in body
    assert 'sse-connect="/api/stream"' in body
    assert 'hx-get="/api/metrics"' in body


@pytest.mark.asyncio
async def test_metrics_panel(monkeypatch):
    monkeypatch.setattr(dashboard_app.web_dashboard, "compute_metrics", lambda *a, **k: {
        "total_runs": 5, "by_status": {"completed": 5}, "total_cost_usd": 1.25,
        "avg_cost_usd": 0.25, "failure_rate": 0.0, "by_mode": {"ask": 5},
    })
    async with await _client() as client:
        resp = await client.get("/api/metrics")
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/html")
        body = await resp.text()
    assert "$1.25" in body and "metric" in body


@pytest.mark.asyncio
async def test_runs_panel(monkeypatch):
    monkeypatch.setattr(dashboard_app, "_runs_rows", lambda: [
        {"run_id": "r1", "mode": "ask", "status": "completed", "total_cost_usd": 0.1, "created_at": "t"}
    ])
    async with await _client() as client:
        resp = await client.get("/api/runs")
        body = await resp.text()
    assert resp.status == 200
    assert 'hx-get="/api/runs/r1"' in body


@pytest.mark.asyncio
async def test_incidents_and_eval_panels(monkeypatch):
    monkeypatch.setattr(dashboard_app, "_incident_counts", lambda: {"run_failed": 2})
    monkeypatch.setattr(dashboard_app.web_dashboard, "compute_eval_summary", lambda *a, **k: {
        "total": 4, "passed": 3, "pass_rate": 0.75, "by_mode": {},
    })
    async with await _client() as client:
        inc = await (await client.get("/api/incidents")).text()
        ev = await (await client.get("/api/eval")).text()
    assert "run_failed" in inc
    assert "3/4" in ev


@pytest.mark.asyncio
async def test_run_detail_drilldown(monkeypatch):
    monkeypatch.setattr(dashboard_app.service, "get_run", lambda rid: {"run": {
        "run_id": "r1", "status": "completed", "summary": "done",
        "agents": [{"alias": "cdx", "status": "completed", "estimated_cost_usd": 0.1, "confidence": 0.9}],
    }})
    async with await _client() as client:
        body = await (await client.get("/api/runs/r1")).text()
    assert "cdx" in body
    assert 'hx-get="/api/runs/r1/events"' in body


@pytest.mark.asyncio
async def test_run_events_timeline(monkeypatch):
    monkeypatch.setattr(dashboard_app, "_run_events", lambda rid: [
        {"seq": 1, "event_type": "run_started", "ts": 0.0, "payload": {}}
    ])
    async with await _client() as client:
        body = await (await client.get("/api/runs/r1/events")).text()
    assert "run_started" in body


@pytest.mark.asyncio
async def test_run_trace_empty_state(monkeypatch):
    monkeypatch.setattr(dashboard_app, "_trace_spans", lambda rid: [])
    async with await _client() as client:
        resp = await client.get("/api/runs/r1/trace")
        body = await resp.text()
    assert resp.status == 200
    assert "trace" in body.lower()


@pytest.mark.asyncio
async def test_paused_cards_enabled(monkeypatch):
    monkeypatch.setattr(dashboard_app.web_dashboard, "list_paused_runs", lambda *a, **k: [
        {"run_id": "r1", "paused_by": "gate", "paused_at": "t", "status": "pending"}
    ])
    monkeypatch.setattr(dashboard_app.config, "dashboard_approvals_enabled", lambda: True)
    async with await _client() as client:
        body = await (await client.get("/api/paused")).text()
    assert 'hx-post="/api/approve"' in body and "r1" in body


@pytest.mark.asyncio
async def test_approve_blocked_when_gate_off(monkeypatch):
    monkeypatch.setattr(dashboard_app.config, "dashboard_approvals_enabled", lambda: False)
    async with await _client() as client:
        resp = await client.post("/api/approve", data={"run_id": "r1", "decision": "approve"})
    assert resp.status == 403


@pytest.mark.asyncio
async def test_approve_applies_when_gate_on(monkeypatch):
    monkeypatch.setattr(dashboard_app.config, "dashboard_approvals_enabled", lambda: True)
    monkeypatch.setattr(dashboard_app.web_dashboard, "apply_approval",
                        lambda run_id, decision, **k: {"ok": True, "run_id": run_id, "decision": decision})
    monkeypatch.setattr(dashboard_app.web_dashboard, "list_paused_runs", lambda *a, **k: [])
    async with await _client() as client:
        resp = await client.post("/api/approve", data={"run_id": "r1", "decision": "approve"})
        body = await resp.text()
    assert resp.status == 200
    assert "awaiting approval" in body.lower() or "hitl" in body.lower()


def test_compute_budget(monkeypatch):
    import sqlite3
    from orchestra.monitoring import web_dashboard
    from orchestra import config
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE runs (run_id TEXT, total_cost_usd REAL)")
    conn.executemany("INSERT INTO runs VALUES (?, ?)", [("r1", 0.3), ("r2", 0.2)])
    conn.commit()
    monkeypatch.setattr(config, "dashboard_budget_usd", lambda: 1.0)
    b = web_dashboard.compute_budget(conn)
    assert abs(b["spent"] - 0.5) < 1e-9
    assert b["budget"] == 1.0
    assert abs(b["remaining"] - 0.5) < 1e-9


@pytest.mark.asyncio
async def test_budget_panel(monkeypatch):
    monkeypatch.setattr(dashboard_app.web_dashboard, "compute_budget",
                        lambda *a, **k: {"spent": 0.5, "budget": 1.0, "remaining": 0.5})
    async with await _client() as client:
        body = await (await client.get("/api/budget")).text()
    assert "$" in body


@pytest.mark.asyncio
async def test_static_css_served():
    async with await _client() as client:
        resp = await client.get("/static/style.css")
    assert resp.status == 200


@pytest.mark.asyncio
async def test_shell_links_stylesheet_and_sse_ext():
    async with await _client() as client:
        body = await (await client.get("/")).text()
    assert '<link rel="stylesheet"' in body
    assert "ext/sse.js" in body or "sse.js" in body


@pytest.mark.asyncio
async def test_fleet_panel(monkeypatch):
    from orchestra.monitoring.characters import Character
    monkeypatch.setattr(dashboard_app.characters, "build_roster", lambda: [
        Character("cdx-deep", "Ada", "Deep Coder", "codex", "xhigh", True, "<svg></svg>")
    ])
    monkeypatch.setattr(dashboard_app.config, "dashboard_dispatch_enabled", lambda: True)
    async with await _client() as client:
        body = await (await client.get("/api/fleet")).text()
    assert "Ada" in body and "char-card" in body
    assert 'hx-post="/api/dispatch"' in body


@pytest.mark.asyncio
async def test_dispatch_blocked_when_disabled(monkeypatch):
    monkeypatch.setattr(dashboard_app.config, "dashboard_dispatch_enabled", lambda: False)
    monkeypatch.setattr(dashboard_app.characters, "build_roster", lambda: [])
    called = {"v": False}
    monkeypatch.setattr(dashboard_app.characters, "dispatch_task", lambda cid, task: called.update(v=True))
    async with await _client() as client:
        resp = await client.post("/api/dispatch", data={"character_id": "cdx-deep", "task": "x"})
        body = await resp.text()
    assert resp.status == 200            # 200 so HTMX shows feedback, not a silent 4xx
    assert called["v"] is False          # nothing dispatched
    assert "dispatch_enabled" in body    # actionable hint shown


@pytest.mark.asyncio
async def test_dispatch_invokes_when_enabled(monkeypatch):
    from orchestra.monitoring.characters import Character
    monkeypatch.setattr(dashboard_app.config, "dashboard_dispatch_enabled", lambda: True)
    monkeypatch.setattr(dashboard_app.characters, "build_roster", lambda: [
        Character("cdx-deep", "Ada", "Deep Coder", "codex", "xhigh", True, "<svg></svg>")
    ])
    seen = {}
    monkeypatch.setattr(dashboard_app.characters, "dispatch_task",
                        lambda cid, task: seen.update(cid=cid, task=task) or {"ok": True})
    async with await _client() as client:
        resp = await client.post("/api/dispatch", data={"character_id": "cdx-deep", "task": "build it"})
        body = await resp.text()
    assert resp.status == 200
    assert seen == {"cid": "cdx-deep", "task": "build it"}
    assert "workstation" in body  # returns the refreshed office view


@pytest.mark.asyncio
async def test_dispatch_auto_cast_when_no_character(monkeypatch):
    from orchestra.monitoring.characters import Character
    monkeypatch.setattr(dashboard_app.config, "dashboard_dispatch_enabled", lambda: True)
    monkeypatch.setattr(dashboard_app.characters, "build_roster", lambda: [
        Character("cdx-deep", "Ada", "Deep Coder", "codex", "xhigh", True, "<svg></svg>")
    ])
    monkeypatch.setattr(dashboard_app.characters, "auto_cast",
                        lambda task, roster: {"character_id": "cdx-deep", "name": "Ada", "reason": "Ada is free"})
    seen = {}
    monkeypatch.setattr(dashboard_app.characters, "dispatch_task",
                        lambda cid, task: seen.update(cid=cid, task=task))
    async with await _client() as client:
        resp = await client.post("/api/dispatch", data={"task": "refactor auth"})  # no character_id
        body = await resp.text()
    assert resp.status == 200
    assert seen == {"cid": "cdx-deep", "task": "refactor auth"}
    assert "Auto-cast" in body and "Ada is free" in body


@pytest.mark.asyncio
async def test_dispatch_auto_cast_no_available(monkeypatch):
    monkeypatch.setattr(dashboard_app.config, "dashboard_dispatch_enabled", lambda: True)
    monkeypatch.setattr(dashboard_app.characters, "build_roster", lambda: [])
    monkeypatch.setattr(dashboard_app.characters, "auto_cast",
                        lambda task, roster: {"error": "no models are available right now"})
    dispatched = {"called": False}
    monkeypatch.setattr(dashboard_app.characters, "dispatch_task",
                        lambda cid, task: dispatched.update(called=True))
    async with await _client() as client:
        resp = await client.post("/api/dispatch", data={"task": "x"})
        body = await resp.text()
    assert resp.status == 200
    assert dispatched["called"] is False           # nothing dispatched
    assert "no models are available" in body


@pytest.mark.asyncio
async def test_character_detail_and_chat(monkeypatch):
    from orchestra.monitoring.characters import Character
    monkeypatch.setattr(dashboard_app.config, "dashboard_dispatch_enabled", lambda: True)
    monkeypatch.setattr(dashboard_app.characters, "build_roster", lambda: [
        Character("cdx-deep", "Ada", "Deep Coder", "codex", "xhigh", True, "<svg></svg>")
    ])
    async with await _client() as client:
        detail = await (await client.get("/api/character/cdx-deep")).text()
        assert "Ada" in detail and 'data-chat-cid="cdx-deep"' in detail
        # unknown character → 404
        assert (await client.get("/api/character/nope")).status == 404
        # chat
        monkeypatch.setattr(dashboard_app.characters, "chat_with",
                            lambda cid, msg: {"ok": True, "reply": f"echo:{msg}"})
        body = await (await client.post("/api/character/cdx-deep/chat", data={"message": "hi"})).text()
    assert "chat-msg you" in body and "echo:hi" in body


@pytest.mark.asyncio
async def test_character_chat_gated(monkeypatch):
    from orchestra.monitoring.characters import Character
    monkeypatch.setattr(dashboard_app.config, "dashboard_dispatch_enabled", lambda: False)
    monkeypatch.setattr(dashboard_app.characters, "build_roster", lambda: [
        Character("cdx-deep", "Ada", "Deep Coder", "codex", "xhigh", True, "<svg></svg>")])
    async with await _client() as client:
        resp = await client.post("/api/character/cdx-deep/chat", data={"message": "hi"})
        body = await resp.text()
    assert resp.status == 200
    assert "dispatch_enabled" in body    # explains how to enable, in a reply bubble


@pytest.mark.asyncio
async def test_static_fleet_js_served():
    async with await _client() as client:
        resp = await client.get("/static/fleet.js")
        body = await resp.text()
    assert resp.status == 200
    assert "EventSource" in body and "drop" in body  # streaming + drag-drop client


@pytest.mark.asyncio
async def test_inbox_and_output_flow(monkeypatch):
    from orchestra.monitoring.characters import Character
    from orchestra.monitoring import dispatch_inbox
    box = dispatch_inbox.DispatchInbox()
    monkeypatch.setattr(dispatch_inbox, "get_inbox", lambda: box)
    monkeypatch.setattr(dashboard_app.characters, "build_roster", lambda: [
        Character("cdx-deep", "Ada", "Deep Coder", "codex", "xhigh", True, "<svg></svg>")
    ])
    did = box.add("cdx-deep", "refactor auth")
    box.complete(did, "refactored 3 files", ok=True)
    async with await _client() as client:
        tray = await (await client.get("/api/inbox")).text()
        assert "Ada" in tray and "refactor auth" in tray and "1 new" in tray
        assert f'hx-get="/api/dispatch/{did}"' in tray
        # read the output → marks read → drops from tray
        out = await (await client.get(f"/api/dispatch/{did}")).text()
        assert "refactored 3 files" in out and "ov-body" in out
        assert box.get(did).read is True
        tray2 = await (await client.get("/api/inbox")).text()
        assert "1 new" not in tray2      # no longer unread
        # unknown dispatch → 404
        assert (await client.get("/api/dispatch/nope")).status == 404


@pytest.mark.asyncio
async def test_office_view(monkeypatch):
    from orchestra.monitoring.characters import Character
    monkeypatch.setattr(dashboard_app.characters, "build_roster", lambda: [
        Character("cdx-deep", "Ada", "Deep Coder", "codex", "xhigh", True, "<svg></svg>")
    ])
    async with await _client() as client:
        body = await (await client.get("/api/office")).text()
    assert "office" in body and "workstation" in body and "Ada" in body


@pytest.mark.asyncio
async def test_character_history(monkeypatch):
    from orchestra.monitoring.characters import Character
    from orchestra.monitoring import dispatch_inbox
    box = dispatch_inbox.DispatchInbox()
    box.complete(box.add("cdx-deep", "past task"), "past result")
    monkeypatch.setattr(dispatch_inbox, "get_inbox", lambda: box)
    monkeypatch.setattr(dashboard_app.characters, "build_roster", lambda: [
        Character("cdx-deep", "Ada", "Deep Coder", "codex", "xhigh", True, "<svg></svg>")
    ])
    async with await _client() as client:
        body = await (await client.get("/api/character/cdx-deep/history")).text()
    assert "past task" in body and "history" in body


@pytest.mark.asyncio
async def test_council_fans_out(monkeypatch):
    monkeypatch.setattr(dashboard_app.config, "dashboard_dispatch_enabled", lambda: True)
    from orchestra.monitoring.characters import Character
    monkeypatch.setattr(dashboard_app.characters, "build_roster", lambda: [
        Character("a", "Ada", "Deep Coder", "codex", "x", True, "<svg></svg>"),
        Character("b", "Bo", "Fast Scout", "perplexity", "sonar", True, "<svg></svg>"),
    ])
    seen = {}
    monkeypatch.setattr(dashboard_app.characters, "dispatch_council",
                        lambda task, ids: seen.update(task=task, ids=ids) or {"ok": True, "count": len(ids)})
    async with await _client() as client:
        resp = await client.post("/api/council", data={"task": "compare"})
        body = await resp.text()
    assert resp.status == 200 and "Council" in body
    assert seen["task"] == "compare" and len(seen["ids"]) >= 1


@pytest.mark.asyncio
async def test_dispatch_pin_and_archive(monkeypatch):
    from orchestra.monitoring import dispatch_inbox
    box = dispatch_inbox.DispatchInbox()
    did = box.add("a", "t"); box.complete(did, "o")
    monkeypatch.setattr(dispatch_inbox, "get_inbox", lambda: box)
    monkeypatch.setattr(dashboard_app.characters, "build_roster", lambda: [])
    async with await _client() as client:
        await client.post(f"/api/dispatch/{did}/pin")
        assert box.get(did).pinned is True
        await client.post(f"/api/dispatch/{did}/archive")
        assert box.get(did).archived is True

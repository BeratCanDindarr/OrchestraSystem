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

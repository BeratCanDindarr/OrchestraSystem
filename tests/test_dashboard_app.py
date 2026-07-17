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

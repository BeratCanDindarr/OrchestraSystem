"""Tests for pure dashboard HTML partial renderers (Faz 6 Block A)."""
from orchestra.monitoring import dashboard_render as r


def test_render_metrics_has_totals_and_cost():
    html = r.render_metrics({
        "total_runs": 3, "by_status": {"completed": 2, "failed": 1},
        "total_cost_usd": 0.5, "avg_cost_usd": 0.1, "failure_rate": 0.33,
        "by_mode": {"ask": 3},
    })
    assert "3" in html
    assert "$0.50" in html
    assert "metric" in html


def test_render_runs_table_row_is_clickable():
    html = r.render_runs_table([
        {"run_id": "r1", "mode": "ask", "status": "completed", "total_cost_usd": 0.1, "created_at": "t"}
    ])
    assert "r1" in html
    assert 'hx-get="/api/runs/r1"' in html


def test_render_paused_cards_gate():
    row = [{"run_id": "r1", "paused_by": "gate", "paused_at": "t", "status": "pending"}]
    on = r.render_paused_cards(row, enabled=True)
    assert 'hx-post="/api/approve"' in on and "r1" in on
    off = r.render_paused_cards(row, enabled=False)
    assert 'hx-post="/api/approve"' not in off


def test_render_timeline_lists_events():
    html = r.render_timeline([{"seq": 1, "event_type": "run_started", "ts": 0.0, "payload": {}}])
    assert "run_started" in html
    assert "1" in html


def test_sparkline_svg():
    svg = r.sparkline_svg([1, 2, 3])
    assert svg.startswith("<svg")
    assert "<polyline" in svg


def test_sse_pack_format_and_flatten():
    assert r.sse_pack("run_update", "<tr>x</tr>") == b"event: run_update\ndata: <tr>x</tr>\n\n"
    packed = r.sse_pack("run_update", "line1\nline2")
    # data must be a single line (no bare newline inside the data payload)
    body = packed.decode()
    data_line = [ln for ln in body.split("\n") if ln.startswith("data: ")][0]
    assert "line1" in data_line and "line2" in data_line


def test_render_escapes_user_text():
    html = r.render_runs_table([
        {"run_id": "<b>x", "mode": "ask", "status": "completed", "total_cost_usd": 0.0, "created_at": "t"}
    ])
    assert "<b>x" not in html  # escaped
    assert "&lt;b&gt;x" in html

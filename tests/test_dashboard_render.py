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


def test_render_budget_over_limit():
    html = r.render_budget({"spent": 1.5, "budget": 1.0, "remaining": -0.5})
    assert "over budget" in html
    assert "$0.50" in html
    assert "$-0.50" not in html  # no misleading negative "left"


def test_render_budget_under_limit():
    html = r.render_budget({"spent": 0.4, "budget": 1.0, "remaining": 0.6})
    assert "$0.60 left" in html
    assert "over budget" not in html


def test_render_character_card_shows_persona_and_state():
    from orchestra.monitoring.characters import Character
    c = Character(id="cdx-deep", name="Ada", role="Deep Coder", provider="codex",
                  model="xhigh", available=True, avatar="<svg></svg>")
    html = r.render_character_card(c, "busy", dispatch_enabled=True)
    assert "Ada" in html and "Deep Coder" in html and "codex" in html
    assert "st-busy" in html
    assert "<svg" in html
    assert 'hx-post="/api/dispatch"' in html   # dispatch form present when enabled


def test_render_character_card_no_dispatch_when_disabled():
    from orchestra.monitoring.characters import Character
    c = Character(id="x", name="Bo", role="Fast Scout", provider="perplexity",
                  model="sonar", available=True, avatar="<svg></svg>")
    html = r.render_character_card(c, "available", dispatch_enabled=False)
    assert 'hx-post="/api/dispatch"' not in html


def test_render_fleet_lists_all():
    from orchestra.monitoring.characters import Character
    chars = [
        (Character("a", "Ada", "Deep Coder", "codex", "xhigh", True, "<svg></svg>"), "busy"),
        (Character("b", "Bo", "Fast Scout", "perplexity", "sonar", False, "<svg></svg>"), "offline"),
    ]
    html = r.render_fleet(chars, dispatch_enabled=True)
    assert "Ada" in html and "Bo" in html
    assert "st-busy" in html and "st-offline" in html


def test_render_character_escapes_user_persona():
    from orchestra.monitoring.characters import Character
    c = Character("x", "<b>evil", "role", "codex", "m", True, "<svg></svg>")
    html = r.render_character_card(c, "idle", dispatch_enabled=True)
    assert "<b>evil" not in html
    assert "&lt;b&gt;evil" in html


def test_render_fleet_shows_cast_notice():
    from orchestra.monitoring.characters import Character
    chars = [(Character("a", "Ada", "Deep Coder", "codex", "x", True, "<svg></svg>"), "available")]
    html = r.render_fleet(chars, dispatch_enabled=True, notice="Auto-cast → Ada is free")
    assert "cast-notice" in html and "Ada is free" in html


def test_render_character_detail_chat_gate():
    from orchestra.monitoring.characters import Character
    c = Character("cdx-deep", "Ada", "Deep Coder", "codex", "xhigh", True, "<svg></svg>")
    on = r.render_character_detail(c, chat_enabled=True)
    assert 'data-chat-cid="cdx-deep"' in on and "chat-log" in on
    off = r.render_character_detail(c, chat_enabled=False)
    assert "data-chat-cid" not in off and "disabled" in off


def test_render_chat_exchange_escapes():
    html = r.render_chat_exchange("Ada", "<b>hi", "<i>reply")
    assert "chat-msg you" in html and "chat-msg them" in html
    assert "<b>hi" not in html and "&lt;b&gt;hi" in html
    assert "&lt;i&gt;reply" in html


def test_render_inbox_running_and_unread():
    from orchestra.monitoring.dispatch_inbox import Dispatch
    ds = [
        Dispatch(id="d1", character_id="a", task="task one", status="running"),
        Dispatch(id="d2", character_id="b", task="task two", status="completed", output="ok"),
    ]
    html = r.render_inbox(ds, {"a": "Ada", "b": "Bex"})
    assert "Ada" in html and "Bex" in html
    assert "1 new" in html
    assert 'hx-get="/api/dispatch/d2"' in html   # completed is clickable
    assert "⏳" in html and "✅" in html


def test_render_inbox_empty():
    assert "No active work" in r.render_inbox([], {})


def test_render_dispatch_output_terminal():
    from orchestra.monitoring.dispatch_inbox import Dispatch
    d = Dispatch(id="d1", character_id="a", task="build <it>", status="completed", output="line1\nline2")
    html = r.render_dispatch_output(d, "Ada")
    assert "Ada" in html and "ov-body" in html and "$ build" in html
    assert "&lt;it&gt;" in html and "line1" in html and "line2" in html   # escaped, multiline


def test_render_office_desks_and_states():
    from orchestra.monitoring.characters import Character
    items = [
        (Character("cdx-deep", "Ada", "Deep Coder", "codex", "xhigh", True, "<svg></svg>"), "busy"),
        (Character("oll-fast", "Bo", "Fast Local", "ollama", "fast", True, "<svg></svg>"), "idle"),
        (Character("gmn-pro", "Cy", "Deep Researcher", "gemini", "pro", True, "<svg></svg>"), "available"),
    ]
    html = r.render_office(items, dispatch_enabled=True, notify_ids={"gmn-pro"})
    assert "office topdown" in html and html.count('class="desk"') == 3
    assert html.count('data-cid=') == 3
    assert 'hx-get="/api/character/cdx-deep"' in html
    assert "st-busy" in html and "st-idle" in html
    assert "at-desk" in html                            # busy/done workers walked to desks
    assert "has-result" in html and "done!" in html     # notify bubble on gmn-pro
    assert "typing…" in html and "zzz" in html          # busy + idle state bubbles
    assert "Lounge" in html and "Workstations" in html


def test_render_office_empty_and_hint():
    html = r.render_office([], dispatch_enabled=False, notify_ids=set())
    assert "No models configured" in html and "dispatch_enabled" in html


def test_render_dispatch_output_live_when_running():
    from orchestra.monitoring.dispatch_inbox import Dispatch
    d = Dispatch(id="d1", character_id="a", task="build", status="running", output="partial out")
    html = r.render_dispatch_output(d, "Ada")
    assert 'sse-connect="/api/dispatch/d1/stream"' in html and "ov-body live" in html
    assert "partial out" in html and "live" in html

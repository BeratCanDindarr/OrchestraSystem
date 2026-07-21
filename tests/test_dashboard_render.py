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


def _char(cid, name, role="Coder"):
    from orchestra.monitoring.characters import Character
    return Character(cid, name, role, "codex", "xhigh", True, "<svg></svg>")


class _Disp:
    def __init__(self, cid, status="running", room_id=None):
        self.character_id, self.status, self.room_id = cid, status, room_id


class _Room:
    def __init__(self, rid, name, preset="custom", rules="", connections=(),
                 node_type="agent", x=40.0, y=40.0, content="", default_model=""):
        self.id, self.name, self.preset, self.rules = rid, name, preset, rules
        self.connections = tuple(connections)
        self.node_type, self.x, self.y, self.content = node_type, x, y, content
        self.default_model = default_model


def test_rooms_with_members_attaches_and_dedups():
    chars = {"a": _char("a", "Ada"), "b": _char("b", "Bo"), "c": _char("c", "Cy")}
    store = [_Room("r1", "Research", "researcher"), _Room("r2", "Software", "senior_engineer")]
    disp = [
        _Disp("a", "running", "r1"),
        _Disp("b", "running", "r1"),        # two models collaborating in r1
        _Disp("a", "completed", "r1"),      # dedup: 'a' keeps the more-active 'running'
        _Disp("c", "completed", "r2"),
        _Disp("a", "running", None),        # untagged dispatch is ignored
    ]
    rooms = r.rooms_with_members(store, disp, chars)
    r1 = next(rm for rm in rooms if rm["id"] == "r1")
    assert {m["cid"] for m in r1["members"]} == {"a", "b"} and r1["running"] == 2
    assert next(m for m in r1["members"] if m["cid"] == "a")["status"] == "running"
    r2 = next(rm for rm in rooms if rm["id"] == "r2")
    assert [m["cid"] for m in r2["members"]] == ["c"] and r2["running"] == 0


def test_render_office_graph_canvas_nodes_and_palette():
    roster = [(_char("a", "Ada"), "busy"), (_char("b", "Bo"), "idle"), (_char("c", "Cy"), "available")]
    nodes = r.rooms_with_members(
        [_Room("s1", "Start", node_type="start", content="do the thing", connections=("a1",)),
         _Room("a1", "Research", "researcher", node_type="agent")],
        [_Disp("a", "running", "a1")], {c.id: c for c, _ in roster})
    html = r.render_office(roster, nodes, dispatch_enabled=True)
    assert "office graph" in html and "graph-canvas" in html
    # palette: add each node type + a Play button
    assert all(f'"type":"{t}"' in html for t in ("start", "agent", "prompt", "output"))
    assert "graph-play" in html and "/api/graph/play" in html
    # nodes positioned + typed; ports present; roster still there
    assert 'data-node="s1"' in html and 'data-node-type="start"' in html and 'data-connections="a1"' in html
    assert 'class="port in"' in html and 'class="port out"' in html
    assert "Research" in html and "Ada" in html and 'class="roster"' in html


def test_node_card_types_and_ports():
    make = lambda **k: r.rooms_with_members([_Room(**k)], [], {})[0]
    start = r._node_card(make(rid="s", name="Start", node_type="start", content="task text"), {}, {})
    assert "nt-start" in start and "task text" in start and 'class="port in"' not in start  # start has no input
    out = r._node_card(make(rid="o", name="Out", node_type="output", content="the result"), {}, {})
    assert "nt-output" in out and "the result" in out and 'class="port out"' not in out    # output has no output
    prompt = r._node_card(make(rid="p", name="P", node_type="prompt", content="be terse"), {}, {})
    assert "nt-prompt" in prompt and "be terse" in prompt


def test_node_card_agent_inline_controls_reflect_selection():
    from orchestra.monitoring import room_presets
    node = r.rooms_with_members(
        [_Room("a", "Dev", "researcher", node_type="agent", default_model="oll-fast")],
        [], {})[0]
    html = r._node_card(node, {}, {}, room_presets.preset_list(), [("cdx-deep", "Ada"), ("oll-fast", "Bo")])
    # inline model + preset dropdowns show the node's saved selection (not Auto-cast)
    assert 'value="oll-fast" selected' in html and 'value="researcher" selected' in html
    # fields are marked data-nf so the JS change→fetch handler persists them (not htmx)
    assert 'name="default_model" data-nf' in html and 'name="preset" data-nf' in html


def test_render_node_editor_by_type():
    from orchestra.monitoring import room_presets
    presets = room_presets.preset_list()
    start = r.render_node_editor(_Room("s", "Start", node_type="start", content="hi"), presets=presets)
    assert 'hx-post="/api/nodes/s"' in start and 'name="content"' in start and "Start" in start
    agent = r.render_node_editor(_Room("a", "Agent", "researcher", node_type="agent"),
                                 presets=presets, models=[("m1", "Ada")])
    assert 'name="preset"' in agent and 'name="default_model"' in agent and 'value="researcher" selected' in agent


def test_render_room_editor_create_and_edit():
    from orchestra.monitoring import room_presets
    presets = room_presets.preset_list()
    create = r.render_room_editor(None, presets=presets)
    assert 'hx-post="/api/rooms"' in create and "New room" in create
    assert '<textarea name="rules"' in create and '<select name="preset"' in create
    room = _Room("r9", "Decision Room", "decision_maker", "Pick one option.")
    edit = r.render_room_editor(room, presets=presets)
    assert 'hx-post="/api/rooms/r9"' in edit and "Edit room" in edit
    assert 'value="Decision Room"' in edit and "Pick one option." in edit
    assert 'value="decision_maker" selected' in edit
    assert "/api/rooms/r9/delete" in edit                # delete button when editing


def test_render_room_editor_connections_picker():
    from orchestra.monitoring import room_presets
    presets = room_presets.preset_list()
    decision = _Room("r1", "Decision Room", "decision_maker", connections=("r2",))
    others = [decision, _Room("r2", "Software Room"), _Room("r3", "Research Room")]
    html = r.render_room_editor(decision, presets=presets, all_rooms=others)
    # picker lists the OTHER rooms (self excluded), r2 pre-checked
    assert 'name="connections" value="r2" checked' in html
    assert 'name="connections" value="r3"' in html and 'value="r1"' not in html.split("Hand off")[1]
    assert "Software Room" in html and "Research Room" in html


def test_render_office_shows_connections_and_router_badge():
    roster = [(_char("a", "Ada"), "idle")]
    store = [_Room("r1", "Decision Room", "decision_maker", connections=("r2",)),
             _Room("r2", "Software Room", "senior_engineer")]
    rooms = r.rooms_with_members(store, [], {"a": _char("a", "Ada")})
    html = r.render_office(roster, rooms, dispatch_enabled=True)
    assert "room-router" in html and "router" in html           # decision agent node flagged
    assert "Software Room" in html                               # target name shown
    assert 'data-connections="r2"' in html                      # edge data for the wire overlay
    assert 'class="node-wires"' in html                         # svg overlay present


def test_render_office_empty_and_hint():
    html = r.render_office([], [], dispatch_enabled=False)
    assert '"type":"start"' in html and "graph-play" in html and "dispatch_enabled" in html


class _WS:
    def __init__(self, wid, name):
        self.id, self.name = wid, name


def test_render_office_workspace_tabs():
    roster = [(_char("a", "Ada"), "idle")]
    wss = [_WS("w1", "Research"), _WS("w2", "Shipping")]
    html = r.render_office(roster, [], dispatch_enabled=True, workspaces=wss, active_ws="w2")
    assert "ws-tabs" in html and "Research" in html and "Shipping" in html
    # active tab flagged; switch/new/delete controls present
    assert '/api/workspaces/w2/activate' in html and '/api/workspaces/w1/activate' in html
    assert 'hx-post="/api/workspaces"' in html                 # new workspace
    assert '/api/workspaces/w2/delete' in html                 # delete active (2 workspaces)


def test_render_office_single_workspace_no_delete():
    html = r.render_office([], [], dispatch_enabled=True, workspaces=[_WS("w1", "Default")], active_ws="w1")
    assert "/api/workspaces/w1/delete" not in html             # never delete the last workspace


def test_render_workspace_editor():
    html = r.render_workspace_editor(_WS("w1", "Research"))
    assert 'hx-post="/api/workspaces/w1"' in html and 'value="Research"' in html and "Rename workspace" in html


def test_render_office_sorts_inactive_models_to_bottom():
    roster = [(_char("off1", "Zed"), "offline"), (_char("busy1", "Ada"), "busy"),
              (_char("idle1", "Bo"), "idle"), (_char("off2", "Yuki"), "offline")]
    html = r.render_office(roster, [], dispatch_enabled=True)
    # active models (busy/idle) come before the "Inactive" separator; offline after
    assert html.index('data-cid="busy1"') < html.index("roster-sep")
    assert html.index('data-cid="idle1"') < html.index("roster-sep")
    assert html.index("roster-sep") < html.index('data-cid="off1"')
    assert html.index("roster-sep") < html.index('data-cid="off2"')
    assert "Inactive" in html


def test_render_flow_history_and_empty():
    assert "No hand-offs yet" in r.render_flow([])
    html = r.render_flow([{"from_name": "Decision Room", "to_name": "Research Room", "cid": "m1"}],
                         names={"m1": "Ada"})
    assert "Decision Room" in html and "Research Room" in html and "Ada" in html and "flow-row" in html


def test_render_dispatch_output_live_when_running():
    from orchestra.monitoring.dispatch_inbox import Dispatch
    d = Dispatch(id="d1", character_id="a", task="build", status="running", output="partial out")
    html = r.render_dispatch_output(d, "Ada")
    assert 'data-stream-did="d1"' in html and "ov-body live" in html   # JS EventSource hook
    assert "partial out" in html and "live" in html


def test_sse_pack_stream_preserves_newlines():
    packed = r.sse_pack_stream("token", "line1\nline2").decode("utf-8")
    assert packed == "event: token\ndata: line1\ndata: line2\n\n"   # EventSource rejoins as line1\nline2


def test_render_status_chips():
    html = r.render_status({"online": 5, "total": 26, "busy": 2, "active": 1, "version": "1.0.0"})
    assert "5</b>/26 online" in html and "2</b> working" in html
    assert "1</b> active" in html and "v1.0.0" in html


def test_render_health_up_down():
    html = r.render_health([
        {"name": "codex", "online": 2, "total": 2},
        {"name": "ollama", "online": 0, "total": 6},
    ])
    assert "hz-dot up" in html and "hz-dot down" in html
    assert "codex" in html and "ollama" in html and "1/2 up" in html


def test_render_settings_reflects_config():
    html = r.render_settings({"dispatch_enabled": True, "approvals_enabled": False,
                              "include_remote": False, "budget_usd": 2.5})
    assert 'name="dispatch_enabled" checked' in html
    assert 'name="approvals_enabled" ' in html and 'name="approvals_enabled" checked' not in html
    assert 'value="2.5"' in html and 'hx-post="/api/settings"' in html


# ── time-series charts (Functional Power cluster) ──────────────────────────────

def test_aggregate_timeseries_buckets_cost_latency_count():
    rows = [
        {"created_at": "2026-07-20T10:00:00+00:00", "updated_at": "2026-07-20T10:00:10+00:00",
         "total_cost_usd": 0.10, "status": "completed"},
        {"created_at": "2026-07-20T10:05:00+00:00", "updated_at": "2026-07-20T10:05:30+00:00",
         "total_cost_usd": 0.20, "status": "failed"},
        {"created_at": "2026-07-20T11:00:00+00:00", "updated_at": "2026-07-20T11:00:20+00:00",
         "total_cost_usd": 0.40, "status": "completed"},
    ]
    s = r.aggregate_timeseries(rows, buckets=2)
    assert len(s["buckets"]) == 2
    # first bucket holds the two 10:xx runs, second holds the 11:00 run
    assert abs(s["buckets"][0]["cost"] - 0.30) < 1e-9 and s["buckets"][0]["count"] == 2
    assert abs(s["buckets"][1]["cost"] - 0.40) < 1e-9 and s["buckets"][1]["count"] == 1
    assert abs(s["totals"]["cost"] - 0.70) < 1e-9 and s["totals"]["runs"] == 3
    assert s["totals"]["failures"] == 1
    # avg latency of first bucket = (10 + 30) / 2 = 20s
    assert abs(s["buckets"][0]["avg_latency_s"] - 20.0) < 1e-9


def test_aggregate_timeseries_empty():
    s = r.aggregate_timeseries([], buckets=6)
    assert s["buckets"] == [] and s["totals"]["runs"] == 0


def test_render_charts_has_cost_and_latency():
    s = r.aggregate_timeseries([
        {"created_at": "2026-07-20T10:00:00+00:00", "updated_at": "2026-07-20T10:00:10+00:00",
         "total_cost_usd": 0.10, "status": "completed"},
    ], buckets=3)
    html = r.render_charts(s)
    assert "charts" in html and "svg" in html.lower()
    assert "$0.10" in html and "Cost" in html and "Latency" in html


def test_render_charts_empty():
    assert "No run history" in r.render_charts(r.aggregate_timeseries([], buckets=6))


# ── global search / filter ─────────────────────────────────────────────────────

_SEARCH_ROWS = [
    {"run_id": "aaa1", "mode": "ask", "status": "completed", "total_cost_usd": 0.1,
     "created_at": "t", "task": "refactor the parser module"},
    {"run_id": "bbb2", "mode": "dual", "status": "failed", "total_cost_usd": 0.2,
     "created_at": "t", "task": "write unit tests for auth"},
    {"run_id": "ccc3", "mode": "ask", "status": "running", "total_cost_usd": 0.0,
     "created_at": "t", "task": "summarize the parser design"},
]


def test_filter_runs_free_text_matches_task_and_id():
    assert {r_["run_id"] for r_ in r.filter_runs(_SEARCH_ROWS, "parser")} == {"aaa1", "ccc3"}
    assert {r_["run_id"] for r_ in r.filter_runs(_SEARCH_ROWS, "bbb2")} == {"bbb2"}


def test_filter_runs_field_tokens():
    assert {r_["run_id"] for r_ in r.filter_runs(_SEARCH_ROWS, "status:failed")} == {"bbb2"}
    assert {r_["run_id"] for r_ in r.filter_runs(_SEARCH_ROWS, "mode:ask parser")} == {"aaa1", "ccc3"}
    assert r.filter_runs(_SEARCH_ROWS, "status:failed parser") == []


def test_filter_runs_blank_returns_empty():
    assert r.filter_runs(_SEARCH_ROWS, "   ") == []


def test_render_search_results():
    html = r.render_search_results(r.filter_runs(_SEARCH_ROWS, "parser"), "parser")
    assert "aaa1" in html and "ccc3" in html and "bbb2" not in html
    assert 'hx-get="/api/runs/aaa1"' in html and "refactor the parser" in html


def test_render_search_results_empty_states():
    assert "Type to search" in r.render_search_results([], "")
    assert "No matches" in r.render_search_results([], "zzz")


# ── audit log ──────────────────────────────────────────────────────────────────

def test_render_audit_lists_events():
    html = r.render_audit([
        {"event_type": "agent_completed", "run_id": "abc12345", "ts": "2026-07-20T10:00:00+00:00", "seq": 9},
        {"event_type": "run_failed", "run_id": "def67890", "ts": "2026-07-20T10:01:00+00:00", "seq": 10},
    ])
    assert "agent_completed" in html and "run_failed" in html
    assert "abc12345" in html and "audit" in html
    # a failure event is flagged so it can be styled distinctly
    assert "ev-fail" in html


def test_render_audit_escapes_and_empty():
    assert "No events" in r.render_audit([])
    html = r.render_audit([{"event_type": "<script>", "run_id": "x", "ts": "t", "seq": 1}])
    assert "<script>" not in html and "&lt;script&gt;" in html

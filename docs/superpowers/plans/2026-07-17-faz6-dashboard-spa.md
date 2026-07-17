# Faz 6 — Live Dashboard (HTMX + SSE + aiohttp) Implementation Plan

> REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** Statik dashboard → canlı HTMX+SSE aiohttp dashboard: metrics/incidents/eval/budget panelleri, run drill-down (agents/timeline/trace), etkileşimli HITL kartları, SSE canlı akış. Backend tamamen pytest-testli.

**Architecture:** Bkz `docs/superpowers/specs/2026-07-17-faz6-dashboard-spa-design.md`. Saf `dashboard_render.py` (HTML string) + `dashboard_app.py` (aiohttp) + static shell. Mevcut `web_dashboard.compute_*` korunur.

## Global Constraints

- Render fn'ler **saf** (str döner, I/O yok) → unit-testable. App route'ları partial HTML döner (HTMX swap), JSON değil.
- Mevcut `web_dashboard.py` compute fn'lerine + testlerine dokunma (yalnız `compute_budget` ekle).
- HTMX CDN referansı (air-gapped için `/static/htmx.min.js` route vendored kopya varsa servis eder).
- Conventional commits. Scratch dizininden test. aiohttp `TestClient(TestServer(...))` deseni (test_gateway/test_agent_card).

---

### Task A: `dashboard_render.py` — pure HTML partials + sse_pack

**Files:** Create `orchestra/monitoring/dashboard_render.py`; Test `tests/test_dashboard_render.py`.

**Interfaces:** Produces `render_metrics(m)->str`, `render_incidents(d)->str`, `render_eval(e)->str`, `render_runs_table(rows)->str`, `render_run_detail(run)->str`, `render_timeline(events)->str`, `render_paused_cards(rows, *, enabled)->str`, `render_budget(b)->str`, `render_run_update(run_id, event)->str`, `sparkline_svg(values)->str`, `sse_pack(event, data)->bytes`.

- [ ] **A1** Write failing tests `tests/test_dashboard_render.py`:
  - `render_metrics({"total_runs":3,"by_status":{"completed":2},"total_cost_usd":0.5,"avg_cost_usd":0.1,"failure_rate":0.0,"by_mode":{"ask":3}})` → contains "3" and "$0.50" and a `metric` card class.
  - `render_runs_table([{"run_id":"r1","mode":"ask","status":"completed","total_cost_usd":0.1,"created_at":"t"}])` → contains "r1" and `hx-get="/api/runs/r1"`.
  - `render_paused_cards([{"run_id":"r1","paused_by":"gate","paused_at":"t","status":"pending"}], enabled=True)` → contains `hx-post="/api/approve"` and "r1"; `enabled=False` → no `hx-post`, shows disabled note.
  - `render_timeline([{"seq":1,"event_type":"run_started","ts":0.0,"payload":{}}])` → contains "run_started" and "1".
  - `sparkline_svg([1,2,3])` → starts with "<svg" and has a `<polyline`.
  - `sse_pack("run_update","<tr>x</tr>")` == `b"event: run_update\ndata: <tr>x</tr>\n\n"`; multiline data flattened (no bare `\n` in data line).
- [ ] **A2** Run → FAIL (module missing).
- [ ] **A3** Create `dashboard_render.py`: each render fn builds an HTML string via f-strings with `html.escape` on user text; `sparkline_svg` maps values→points polyline; `sse_pack(event,data)` → `f"event: {event}\ndata: {data.replace(chr(10),' ')}\n\n".encode()`.
- [ ] **A4** Run → PASS.
- [ ] **A5** Commit `feat(dashboard): pure HTML partial renderers + sse_pack`.

---

### Task B: `dashboard_app.py` — aiohttp app + core panels + shell

**Files:** Create `orchestra/monitoring/dashboard_app.py`; Test `tests/test_dashboard_app.py`.

**Interfaces:** Consumes render fns (A), `web_dashboard.compute_metrics/count_incidents/compute_eval_summary/list_paused_runs/_events_conn`. Produces `dashboard_app.make_app() -> web.Application` with routes `/`, `/api/metrics`, `/api/incidents`, `/api/eval`, `/api/runs`.

- [ ] **B1** Write failing tests (`tests/test_dashboard_app.py`, aiohttp TestClient):
  - GET `/` → 200, body contains `htmx` (script) and `sse-connect="/api/stream"` and `hx-get="/api/metrics"`.
  - GET `/api/metrics` → 200, `Content-Type` text/html, body has a metric card (monkeypatch `compute_metrics` → fixed dict).
  - GET `/api/runs` → 200, body has runs table (monkeypatch the runs query source → fixed rows).
  - GET `/api/eval`, `/api/incidents` → 200 html.
- [ ] **B2** Run → FAIL.
- [ ] **B3** Create `dashboard_app.py`: `make_app()` builds `web.Application`; handlers call compute fns + render fns, return `web.Response(text=..., content_type="text/html")`. `/` returns the shell HTML (loads htmx + sse ext, panels with `hx-get ... hx-trigger="load"`, `hx-ext="sse" sse-connect="/api/stream"`). Runs list: a `_runs_rows()` helper reading `runs` table via `get_db()` (last 100), injectable for tests.
- [ ] **B4** Run → PASS.
- [ ] **B5** Commit `feat(dashboard): aiohttp app with live panels + HTMX shell`.

---

### Task C: SSE live stream (`/api/stream`) + bus bridge

**Files:** Modify `orchestra/monitoring/dashboard_app.py`; Test `tests/test_dashboard_sse.py`.

**Interfaces:** Produces `dashboard_app._event_to_fragment(run_id, event)->str`; route `GET /api/stream`; `dashboard_app.subscribe_bus(loop, queue)` bridge helper.

- [ ] **C1** Write failing tests:
  - unit: a bridge `_make_bus_handler(loop, queue)` → returns a handler that on call does `loop.call_soon_threadsafe(queue.put_nowait, (run_id, event))`; test by calling handler with a real loop + queue, assert item lands.
  - SSE: TestClient GET `/api/stream` with a timeout; emit via `bus.emit("r1", {"event":"agent_completed"})`; read one chunk; assert `event: run_update` and `agent_completed` in bytes. (Use `asyncio.wait_for` with short timeout; close response.)
- [ ] **C2** Run → FAIL.
- [ ] **C3** Add `/api/stream` handler (see spec: StreamResponse, subscribe_all bridge to asyncio.Queue, write `sse_pack("run_update", render_run_update(...))`, `finally: unsubscribe_all`). Register route in `make_app`.
- [ ] **C4** Run → PASS.
- [ ] **C5** Commit `feat(dashboard): SSE live event stream bridged from the event bus`.

---

### Task D: Run drill-down (detail / events / trace)

**Files:** Modify `orchestra/monitoring/dashboard_app.py`, `dashboard_render.py`; Test extends `tests/test_dashboard_app.py`.

**Interfaces:** Produces routes `GET /api/runs/{rid}`, `GET /api/runs/{rid}/events`, `GET /api/runs/{rid}/trace`; render `render_run_detail`, `render_timeline`, `render_trace(spans)`.

- [ ] **D1** Write failing tests:
  - `/api/runs/{rid}` → monkeypatch `service.get_run` → `{"run":{"run_id":"r1","status":"completed","agents":[{"alias":"a","status":"completed","estimated_cost_usd":0.1,"confidence":0.9}],"summary":"s"}}`; body has "a" and `hx-get="/api/runs/r1/events"`.
  - `/api/runs/{rid}/events` → monkeypatch `EventLog.replay` → 1 event; body has event_type.
  - `/api/runs/{rid}/trace` → no spans → body contains "trace" (empty state) and 200.
- [ ] **D2** Run → FAIL.
- [ ] **D3** Add routes + `render_run_detail(run)` (agents table + summary + timeline hx-get), `render_trace(spans)` (span hierarchy or empty state). Trace source: `tracer` in-memory exporter `get_spans()` filtered by run_id if available, else [].
- [ ] **D4** Run → PASS.
- [ ] **D5** Commit `feat(dashboard): run drill-down — detail, event timeline, OTel trace`.

---

### Task E: HITL cards + budget panel + config

**Files:** Modify `orchestra/monitoring/dashboard_app.py`, `orchestra/monitoring/web_dashboard.py` (compute_budget), `orchestra/config.py`; Test extends `tests/test_dashboard_app.py`.

**Interfaces:** Produces routes `GET /api/paused`, `POST /api/approve`, `GET /api/budget`; `web_dashboard.compute_budget(conn=None)->dict`; `config.dashboard_budget_usd()->float`.

- [ ] **E1** Write failing tests:
  - `/api/paused` → monkeypatch `list_paused_runs` → 1 row; `dashboard_approvals_enabled → True` → card has `hx-post="/api/approve"`.
  - `POST /api/approve` gate off (`dashboard_approvals_enabled → False`) → 403.
  - `POST /api/approve` gate on + monkeypatch `apply_approval` → `{"ok":True,...}` → 200 html (updated queue).
  - `compute_budget(conn)` in-memory runs conn (total_cost_usd sum) + monkeypatch `config.dashboard_budget_usd → 1.0` → `{"spent":X,"budget":1.0,"remaining":1.0-X,...}`.
  - `/api/budget` → 200 html with "$".
- [ ] **E2** Run → FAIL.
- [ ] **E3** Add `config.dashboard_budget_usd()` (default 0.0); `web_dashboard.compute_budget(conn=None)` (sum runs.total_cost_usd; remaining = budget-spent if budget>0 else None); dashboard_app `/api/paused`, `/api/approve` (gate → 403 else apply_approval), `/api/budget` routes + render.
- [ ] **E4** Run tests + `tests/test_web_dashboard_hitl.py` → PASS.
- [ ] **E5** Commit `feat(dashboard): interactive HITL cards + live budget panel`.

---

### Task F: static shell assets + CLI wiring

**Files:** Create `orchestra/monitoring/dashboard/index.html`, `style.css`; Modify `orchestra/monitoring/dashboard_app.py` (serve shell + `/static/*`), `orchestra/cli.py`; Test extends `tests/test_dashboard_app.py`.

**Interfaces:** `dashboard_app.run_app(host, port)`; CLI `orchestra dashboard` → new app.

- [ ] **F1** Write failing tests:
  - `/static/style.css` → 200 (served from package dir).
  - `/` shell includes `<link rel="stylesheet"` and htmx sse ext script.
- [ ] **F2** Run → FAIL.
- [ ] **F3** Add `index.html` (full shell: head loads htmx + htmx sse ext from CDN + `/static/style.css`; body panels), `style.css`; serve `/` from index.html (fallback to inline if missing) + `app.router.add_static("/static/", <dashboard dir>)`; `run_app(host,port)` via `web.run_app(make_app(),...)`; repoint `cli.py dashboard` command to `dashboard_app.run_app`.
- [ ] **F4** Run tests → PASS. Smoke: `python -c "from orchestra.monitoring import dashboard_app; app=dashboard_app.make_app(); print(len(app.router.routes()))"`.
- [ ] **F5** Commit `feat(dashboard): static HTMX shell assets + wire orchestra dashboard command`.

---

## Final Verification

```bash
.venv/bin/python -m pytest tests/test_dashboard_render.py tests/test_dashboard_app.py tests/test_dashboard_sse.py \
  tests/test_web_dashboard_metrics.py tests/test_web_dashboard_hitl.py tests/test_web_dashboard_eval.py \
  tests/test_web_dashboard_incidents.py -p no:cacheprovider --no-cov -q
.venv/bin/mypy orchestra/monitoring/dashboard_render.py orchestra/monitoring/dashboard_app.py orchestra/config.py
```
Smoke: `orchestra dashboard --port 8137` → GET `/` renders; `/api/metrics` partial; `/api/stream` streams on a live run.

## Progress Log

| Date | Tasks | Notes |
|------|-------|-------|

## Self-Review

- Coverage: SSE canlı akış(C)✓ HTMX SPA+grafik(A/B/F)✓ drill-down(D)✓ HITL(E)✓ observability+budget(B/E)✓.
- Placeholders: none. Types: render fn adları + `make_app`/`run_app`/`compute_budget`/`dashboard_budget_usd`/`sse_pack` uniform across tasks.

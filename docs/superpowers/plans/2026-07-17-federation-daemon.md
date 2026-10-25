# Federation Sync Daemon + Token Refresh Implementation Plan

> REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** Periyodik federation sync daemon (DI'lı) + token-refresh sinyali + HTTP app wiring — federation'ı uçtan uca aktive et, config-gate default-off.

**Architecture:** Bkz `docs/superpowers/specs/2026-07-17-federation-daemon-design.md`.

## Global Constraints

- DI: `sleep_fn` / `should_continue` / `now_fn` → testte gerçek uyku / asyncio-forever yok.
- Gate default-off (`federation.enabled`=false → daemon başlamaz). Conventional commits. Scratch dizininden test.

---

### Task A: config gates + sync_tick + run_sync_loop

**Files:** Create `orchestra/engine/federation_daemon.py`; Modify `orchestra/config.py`; Test `tests/test_federation_daemon.py`.

**Interfaces:** Produces `federation_daemon.sync_tick(*, now_fn)`, `federation_daemon.run_sync_loop(*, interval, sleep_fn, should_continue, now_fn)`; `config.federation_sync_enabled()`, `config.federation_token_refresh_margin_seconds()`.

- [ ] **A1** Write failing tests (async):
  - `sync_tick`: patch `federation_client.sync_all_peers` (AsyncMock → `{"p":1}`) + `FederationBus.cleanup_expired` (→2); `await sync_tick(now_fn=lambda:0.0)` → `{"sync":{"p":1},"cleaned":2}`.
  - `run_sync_loop`: counter `should_continue` (True×3 then False); `sleep_fn` async recorder; patch `sync_tick` (AsyncMock); `await run_sync_loop(interval=5, sleep_fn=rec, should_continue=cnt)` → sync_tick awaited 3×, sleeps==[5,5,5].
  - `run_sync_loop` swallows tick error: `sync_tick` raises once; loop still completes its ticks.
  - `federation_sync_enabled`: monkeypatch `config.federation_config` → `{"enabled":True,"sync_interval_seconds":10}`→True; `{"enabled":False,...}`→False; `{"enabled":True,"sync_interval_seconds":0}`→False.
- [ ] **A2** Run → FAIL.
- [ ] **A3** Add config `federation_sync_enabled()` + `federation_token_refresh_margin_seconds()` (see spec). Create `federation_daemon.py` with `sync_tick`, `run_sync_loop`, `_warn_peers_needing_refresh` (stub calling needs_refresh — full in Task B).
- [ ] **A4** Run → PASS.
- [ ] **A5** Commit `feat(federation): background sync daemon (tick + loop) + config gates`.

---

### Task B: token refresh signal

**Files:** Modify `orchestra/engine/federation.py`, `orchestra/engine/federation_daemon.py`; Test extends `tests/test_federation_daemon.py`.

**Interfaces:** Produces `FederationPeer.needs_refresh(now, margin_seconds) -> bool`.

- [ ] **B1** Write failing tests:
  - `needs_refresh`: `FederationPeer(peer_id="p", endpoint_url="u", token_expires_at=0.0).needs_refresh(999, 10)` → False (no TTL); `token_expires_at=100` → `needs_refresh(95, 10)` True (95 > 90); `needs_refresh(80, 10)` False.
- [ ] **B2** Run → FAIL.
- [ ] **B3** Add `FederationPeer.needs_refresh` (see spec). Complete `_warn_peers_needing_refresh(now)` in daemon: build peers via `federation_client._peer_from_config` over `config.federation_config().get("peer_urls",[])`, WARN-log those with `needs_refresh(now, config.federation_token_refresh_margin_seconds())`.
- [ ] **B4** Run → PASS.
- [ ] **B5** Commit `feat(federation): pre-expiry token-refresh warning signal`.

---

### Task C: wire daemon into federation HTTP app

**Files:** Modify `orchestra/federation_server/federation_http.py`; Test extends `tests/test_federation_daemon.py`.

**Interfaces:** Produces `federation_http.attach_sync_daemon(app) -> app`.

- [ ] **C1** Write failing tests:
  - `attach_sync_daemon(web.Application())` → `len(app.on_startup)>=1` and `len(app.on_cleanup)>=1`.
  - startup gate: build app, call its on_startup[-1] handler with `config.federation_sync_enabled → False` (monkeypatch) + patched `run_sync_loop` → task NOT created (`"_sync_task" not in app`).
- [ ] **C2** Run → FAIL.
- [ ] **C3** Add `attach_sync_daemon(app)` (on_startup/on_cleanup, gate-checked) + call it at end of `create_app()` (see spec).
- [ ] **C4** Run tests + `tests/test_federation_crdt.py` → PASS.
- [ ] **C5** Commit `feat(federation): auto-start sync daemon on federation server startup`.

---

## Final Verification

```bash
.venv/bin/python -m pytest tests/test_federation_daemon.py tests/test_federation_robustness.py \
  tests/test_federation_crdt.py -p no:cacheprovider --no-cov -q
.venv/bin/mypy orchestra/engine/federation_daemon.py orchestra/engine/federation.py orchestra/config.py
```

## Progress Log

| Date | Tasks | Notes |
|------|-------|-------|
| 2026-07-17 | A-C | Dal `feat/faz5-federation-daemon` (mcp-client üstüne stack), 4 commit. A: `engine/federation_daemon.py` (`sync_tick`=sync_all_peers+cleanup_expired, `run_sync_loop` DI'lı loop, tick-error yutar) + `config.federation_sync_enabled` (enabled && interval>0) + `federation_token_refresh_margin_seconds`. B: `FederationPeer.needs_refresh(now,margin)` + daemon `_warn_peers_needing_refresh` (config-driven rotation: `_peer_from_config` her tick config'i okur → operatör token'ı döndürünce otomatik alınır). C: `federation_http.attach_sync_daemon(app)` → `create_app` on_startup/on_cleanup (gate'li auto-start). 30 test yeşil (daemon+robustness+crdt); yeni modüller mypy temiz (federation_client:90 query_peer no-any-return **pre-existing**). Faz 5 TAMAMEN bitti. |

## Self-Review

- Coverage: daemon loop✓ tick✓ gates✓ token-refresh✓ wiring✓. Placeholders: none. Types: `sync_tick`, `run_sync_loop`, `federation_sync_enabled`, `federation_token_refresh_margin_seconds`, `needs_refresh`, `attach_sync_daemon` uniform.

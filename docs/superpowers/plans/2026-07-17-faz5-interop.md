# Faz 5 — Interop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** A2A Agent Card (server+client) + MCP config gate + Federation delivery-retry/TTL-cleanup — testable, no new network infra.

**Architecture:** Bkz `docs/superpowers/specs/2026-07-17-faz5-interop-design.md`.

## Global Constraints

- Agent Card = read-only discovery metadata (auth'suz, `/health` gibi). MCP gate default **on** (davranış aynı — server zaten çalışıyordu). Federation retry/TTL default korunur.
- DI-testli: `sleep_fn` inject, `now` parametre, aiohttp `TestClient`. Gerçek network/sleep yok.
- Conventional commits. Scratch dizininden test.

---

### Task A: A2A Agent Card (builder + gateway route)

**Files:** Create `orchestra/engine/agent_card.py`; Modify `orchestra/gateway.py`; Modify `orchestra/config.py` (mcp_enabled). Test `tests/test_agent_card.py` (create).

**Interfaces:** Produces `agent_card.build_agent_card(*, host, version, url, aliases, mcp_tools=None) -> dict`; `config.mcp_enabled() -> bool`; gateway `GET /.well-known/agent.json`.

- [ ] **A1** Write failing test `tests/test_agent_card.py`:
  - `build_agent_card(host="h", version="1.0.0", url="http://h/", aliases=["cdx-deep","oll-fast"], mcp_tools=[{"name":"orchestra_run","description":"d"}])` → assert `card["name"]=="orchestra@h"`, `card["protocolVersion"]=="0.2.0"`, `len(card["skills"])==2`, `card["skills"][0]["id"]=="cdx-deep"`, `card["mcpTools"][0]["name"]=="orchestra_run"`.
  - No mcp_tools → `"mcpTools" not in card`.
- [ ] **A2** Run → FAIL (module missing).
- [ ] **A3** Create `orchestra/engine/agent_card.py` with `build_agent_card` (see spec) + `fetch_agent_card(base_url, token="")` (urllib GET `/.well-known/agent.json`, optional Bearer).
- [ ] **A4** Add `config.mcp_enabled()` (default True) after `federation_config` region.
- [ ] **A5** Run builder tests → PASS.
- [ ] **A6** Add gateway `_agent_card` handler + `app.router.add_get("/.well-known/agent.json", _agent_card)` (see spec). Write failing test using `aiohttp.test_utils.TestClient(TestServer(gateway.make_app()))`: GET `/.well-known/agent.json` → 200, JSON has `name` startswith "orchestra@", `skills` is list.
- [ ] **A7** Run gateway route test → PASS (async test; use `pytest.mark.asyncio` or the existing test_gateway pattern).
- [ ] **A8** Commit `feat(interop): A2A Agent Card endpoint + MCP config gate`.

---

### Task B: Agent Card advertises MCP tools by gate

**Files:** Test extends `tests/test_agent_card.py`.

- [ ] **B1** Write failing test: gateway route with `monkeypatch config.mcp_enabled → False` → card has no `mcpTools`; with True (default) + patched `REGISTRY.list_tools` returning `[{"name":"t","description":"d"}]` → card `mcpTools` present. (Verifies the gateway handler honors the gate.)
- [ ] **B2** Run → FAIL if handler doesn't gate.
- [ ] **B3** Ensure `_agent_card` handler builds `mcp_tools` only when `config.mcp_enabled()` (already in A6 spec code — adjust if needed).
- [ ] **B4** Run → PASS.
- [ ] **B5** Commit `feat(interop): gate MCP tool advertisement in Agent Card`.

---

### Task C: Federation delivery retry + TTL cleanup

**Files:** Modify `orchestra/federation_server/federation_client.py`, `orchestra/engine/federation.py`. Test `tests/test_federation_robustness.py` (create).

**Interfaces:** Produces `federation_client.push_with_retry(peer, summaries, *, max_retries=None, sleep_fn=None)`; `FederationBus.cleanup_expired(now: float) -> int`.

- [ ] **C1** Write failing tests (`tests/test_federation_robustness.py`), async where needed:
  - `push_with_retry`: patch `push_to_peer` with a side-effect list `[{"error":"x"},{"error":"x"},{"accepted":1,"total":1}]`; `sleep_fn` = async recorder; call with `max_retries=3` → returns `{"accepted":1,...}`; sleep_fn called 2 times. All-error (max_retries=2) → returns `{"error":...}`, sleep called 2 times.
  - `cleanup_expired`: `bus = FederationBus(root=tmp_path/"fed")`; write two summaries via `enqueue_outbox` — one with `timestamp = now - 10*3600, ttl_hours=1` (expired), one fresh (`timestamp=now`); `bus.cleanup_expired(now)` returns 1; expired gone, fresh kept.
- [ ] **C2** Run → FAIL.
- [ ] **C3** Add `push_with_retry` to federation_client.py (see spec); change `sync_with_peer` to call `push_with_retry` instead of `push_to_peer`.
- [ ] **C4** Add `FederationBus.cleanup_expired(now)` to federation.py (see spec).
- [ ] **C5** Run tests + `tests/test_federation_integration.py` → PASS (federation_integration is DI/tmp-based; federation_http needs aiohttp server — may pre-fail in env, unrelated).
- [ ] **C6** Commit `feat(federation): push retry/backoff (delivery guarantee) + TTL cleanup`.

---

## Final Verification

```bash
.venv/bin/python -m pytest tests/test_agent_card.py tests/test_federation_robustness.py \
  tests/test_gateway.py tests/test_federation_integration.py tests/test_federation_crdt.py \
  -p no:cacheprovider --no-cov -q
.venv/bin/mypy orchestra/engine/agent_card.py orchestra/config.py orchestra/engine/federation.py
```
Confirm pre-existing federation_http failures unchanged (env/network).

## Progress Log

| Date | Tasks | Notes |
|------|-------|-------|

## Self-Review

- Coverage: A2A card builder+route✓ MCP gate✓ MCP advertise✓ federation retry✓ TTL cleanup✓.
- Placeholders: none.
- Types: `build_agent_card`, `fetch_agent_card`, `mcp_enabled`, `push_with_retry`, `cleanup_expired` uniform across tasks + spec.

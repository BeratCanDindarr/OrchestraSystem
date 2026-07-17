# External MCP Client Implementation Plan

> REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** `engine/mcp_client.py` — transport-injected JSON-RPC client to consume external MCP servers (initialize/list_tools/call_tool), newline-delimited stdio, DI-testable.

**Architecture:** Bkz `docs/superpowers/specs/2026-07-17-mcp-client-design.md`.

## Global Constraints

- Transport-injected core → no real subprocess in unit tests (FakeTransport).
- StdioTransport framing = `json.dumps(obj)+"\n"` / `readline()` — birebir server (`mcp_server.py`) ile uyumlu.
- Conventional commits. Scratch dizininden test.

---

### Task A: McpClient core + Transport protocol

**Files:** Create `orchestra/engine/mcp_client.py`; Test `tests/test_mcp_client.py`.

**Interfaces:** Produces `McpClient(transport)` with `initialize()`, `list_tools()->list`, `call_tool(name,arguments=None)->dict`, `close()`, ctx-mgr; `McpClientError`.

- [ ] **A1** Write failing tests `tests/test_mcp_client.py` with a `FakeTransport(recv_script)` (sent list + scripted recv):
  - `list_tools`: recv `{"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"t"}]}}` → returns `[{"name":"t"}]`; sent[0] method=="tools/list", id==1.
  - `call_tool("t",{"a":1})`: recv result `{"ok":true}` (id 1) → returns `{"ok":True}`; sent params=={"name":"t","arguments":{"a":1}}.
  - id-correlation: recv a notification `{"method":"x"}` (no id) THEN the real `{"id":1,"result":{}}` → returns `{}` (notification skipped).
  - error: recv `{"id":1,"error":{"code":-1,"message":"boom"}}` → raises `McpClientError`.
  - closed: recv `None` → raises `McpClientError`.
- [ ] **A2** Run → FAIL (module missing).
- [ ] **A3** Create `orchestra/engine/mcp_client.py`: `McpClientError`, `McpClient` (see spec; `_request` id-correlation loop, initialize/list_tools/call_tool/close/`__enter__`/`__exit__`).
- [ ] **A4** Run → PASS.
- [ ] **A5** Commit `feat(mcp): add transport-injected external MCP client core`.

---

### Task B: StdioTransport (subprocess) + from_command

**Files:** Modify `orchestra/engine/mcp_client.py`; Test extends `tests/test_mcp_client.py`.

**Interfaces:** Produces `StdioTransport(command)` with `start/send/recv/close`; `McpClient.from_command(command)`.

- [ ] **B1** Write failing tests: patch `orchestra.engine.mcp_client.subprocess.Popen` with a fake proc (stdin=io.StringIO-like recorder, stdout=io.StringIO with a JSON line). `StdioTransport(["x"])` → `start()`; `send({"a":1})` writes `'{"a": 1}\n'` to stdin; `recv()` reads+parses the stdout line; empty stdout line → `recv()` returns None.
- [ ] **B2** Run → FAIL.
- [ ] **B3** Add `StdioTransport` (Popen text-mode, send=write+flush, recv=readline→json/None, close=terminate) + `McpClient.from_command`.
- [ ] **B4** Run → PASS.
- [ ] **B5** Commit `feat(mcp): add StdioTransport + from_command factory`.

---

### Task C: config + connect_from_config

**Files:** Modify `orchestra/config.py`, `orchestra/engine/mcp_client.py`; Test extends `tests/test_mcp_client.py`.

**Interfaces:** Produces `config.mcp_client_servers()->dict`; `mcp_client.connect_from_config(name)->McpClient`.

- [ ] **C1** Write failing test: monkeypatch `config.mcp_client_servers` → `{"fs":["cmd","--x"]}`; patch `McpClient.from_command` recorder; `connect_from_config("fs")` → calls `from_command(["cmd","--x"])`; unknown name → `KeyError`/`McpClientError`.
- [ ] **C2** Run → FAIL.
- [ ] **C3** Add `config.mcp_client_servers()` (section("mcp").get("client_servers", {})); add `connect_from_config(name)` to mcp_client.py.
- [ ] **C4** Run → PASS.
- [ ] **C5** Commit `feat(mcp): config-driven external MCP server connect`.

---

## Final Verification

```bash
.venv/bin/python -m pytest tests/test_mcp_client.py tests/test_mcp_server_unit.py -p no:cacheprovider --no-cov -q
.venv/bin/mypy orchestra/engine/mcp_client.py orchestra/config.py
```

## Progress Log

| Date | Tasks | Notes |
|------|-------|-------|

## Self-Review

- Coverage: core client✓ transport✓ config factory✓. Placeholders: none. Types: `McpClient`, `Transport`, `StdioTransport`, `McpClientError`, `from_command`, `connect_from_config`, `mcp_client_servers` uniform.

# Design: Remote node agent gateway (use another machine's agents from this one)

## Status
Approved (brainstorming complete, 2026-07-13)

## Problem

The user runs Orchestra on a MacBook and has a Windows machine (Ryzen 9 9955HX, RTX 5070Ti)
that has its own GPU-accelerated Ollama models AND logged-in CLI agent accounts (Codex/ChatGPT,
Claude, Gemini). They want to use ALL of the Windows machine's resources from the MacBook:
not just its Ollama models (already solved by the multi-backend-Ollama feature via `base_url`),
but its logged-in CLI accounts — e.g. run the Codex account that is authenticated on Windows
from the Mac.

Ollama is reachable remotely because it is an HTTP server. The CLI agents are NOT: `codex`,
`claude`, `gemini` are authenticated on the Windows machine and the Mac cannot reach those
sessions. The fix is to run Orchestra itself on Windows as an **agent gateway** the Mac
delegates tasks to: the Mac says "run this task with your `cdx-deep`", Windows runs its own
logged-in Codex and returns the result.

This is the natural second sub-project after multi-backend Ollama. Same pattern one layer up:
multi-backend Ollama routes `model@backend` to an Ollama `base_url`; this routes `alias@node`
to a remote Orchestra node.

## Goals

- Run Orchestra on Windows as a small HTTP **agent gateway** (`orchestra serve`) exposing
  `POST /run`, `GET /aliases`, `GET /health`.
- Address any of a remote node's ALIASES with **`alias@node`** (e.g. `cdx-deep@windows` runs the
  Windows Codex account; `oll-fast@windows` runs the Windows Ollama through the same node). The
  left side is a Windows *alias* (what `run_ask` takes), not a raw model name.
- A Mac-side `RemoteNodeProvider` that POSTs the task to the node and returns the result as a
  normal provider response (cost recorded on the Mac).
- **Auto-discovery + auto-add:** `orchestra nodes scan` probes the LAN in parallel, finds
  Orchestra gateways (validated by shared token), and writes them to config automatically.
- Token-based auth; safe-by-default (no `0.0.0.0` without a token).

## Non-Goals (deferred / out of scope)

- **Token-by-token live streaming** of the remote agent's output (watching remote Codex type in
  real time) — needs SSE/WS; v1 returns the final result synchronously. The run's status
  (started→completed, cost) still shows in the Mac statusline because the run is managed locally.
- **Symmetric peer** (Windows also delegating to the Mac) — v1 is one-directional
  (Windows = server, Mac = client).
- TLS, user management, rate-limiting — YAGNI for a single-user trusted-network setup.
- VPN setup (Tailscale) — documentation only; the code already accepts any node URL.

## Architecture

```
Mac (client)                              Windows (agent gateway)
────────────                              ───────────────────────
orchestra ask cdx-deep@windows "..."
  → resolve() @node fast-path
  → RemoteNodeProvider(windows)           orchestra serve --token X  (HTTP :7070)
      POST /run {alias, task} + Bearer ─►   → run_ask("cdx-deep", task)
                                              → Windows's logged-in Codex runs
      {stdout, cost_usd, status}       ◄─   → result + cost returned
```

### 1. Windows server — `orchestra serve`

A small purpose-built aiohttp server (`orchestra/gateway.py`, ~150 lines) — NOT the existing
`server.py` WebSocket daemon (which is UI-specific: slot tracking, approval, streaming). It wraps
`service.run_task`/`run_ask` and adds nothing else. Endpoints:
- `POST /run` — body `{alias, task}`; runs `run_ask(alias, task)` locally; returns
  `{stdout, model, cost_usd, status}`.
- `GET /aliases` — the node's `available_aliases()` (what this node can run) — for discovery.
- `GET /health` — liveness + hostname + alias count.

### 2. Mac provider — `RemoteNodeProvider`

New provider (`providers/remote_node.py`, `name = "remote"`), configured with a node's URL + token:
- `native_run(task, alias)` → `POST {url}/run {alias, task}` with `Authorization: Bearer <token>`;
  parses `{stdout, cost_usd, status}` into a `(stdout, returncode)` result; cost → `self._last_usage`.
- `is_available()` → `GET {url}/health`.
- `model_label(alias)` → `remote/{node}/{alias}` so output shows the node + alias.

### 3. Node registry + config

Like `ollama_backends`, a small registry (`remote_nodes.py`) over config:
```toml
[remote_nodes.windows]
url = "http://192.168.1.42:7070"
token = "shared-secret"
```
`list_nodes() -> dict[str, dict]`, `get_node(name) -> RemoteNodeProvider` (cached, ValueError on unknown).

### 4. `@node` resolution + the `@node` vs `@backend` rule

`fallback.resolve()` grows a second `@` fast-path. Since both remote nodes and Ollama backends
use `@`, resolution for `left@X` checks **`[remote_nodes.X]` first, then
`[providers.ollama.backends.X]`**. If a name is defined in both, the remote node wins (with a
logged warning).

The two layers differ in what the LEFT side means:
- `@backend` (Ollama): left side is a **raw model name** — `qwen2.5:7b@windows-gpu` (goes
  straight to that Ollama host's `/api/generate`).
- `@node` (remote): left side is a **Windows alias** that `run_ask` resolves on the node —
  `cdx-deep@windows` (Windows Codex), `oll-fast@windows` (Windows Ollama's `qwen2.5:7b`, since
  `oll-fast` maps to it on Windows).

So one `@windows` node gives access to both the CLI accounts (`cdx-deep@windows`) and Ollama
(`oll-fast@windows`, `oll-coder@windows`) through Windows's own alias table.

### 5. Auto-discovery + auto-add — `orchestra nodes scan`

1. Auto-detect the subnet (from the Mac's own IP → `/24`) or take `--subnet`.
2. Probe all IPs in parallel with `GET :7070/health` (carrying the token) — responders are nodes.
3. `/health` returns the node's hostname + alias count.
4. **Auto-add**: discovered nodes are written to config as `[remote_nodes.<hostname>]`
   (the user chose auto-add). The token both authenticates and filters: only gateways with the
   correct token respond 200 and get added; a random LAN service (or someone else's gateway with
   a different token) returns 401 and is skipped.

This satisfies "auto-scan + self-add" on the pull side: run `orchestra serve --token X` once on
Windows, `orchestra nodes scan --token X` on the Mac, and Windows is found and added — no
registration software on Windows.

## Data Flow

1. Mac: `ask cdx-deep@windows "..."` → a local `OrchestraRun` is created (shows in the
   statusline), provider resolves to `RemoteNodeProvider`.
2. Provider → Windows `:7070/run` HTTP POST (+ Bearer token).
3. Windows: `run_ask("cdx-deep", task)` → logged-in Codex runs → result + cost returned.
4. Mac: processes the result like a normal run — manifest, events, cost recorded on the Mac.

## Security

The gateway grants a token holder the ability to **run agents on Windows** — and agents like
`codex`/`claude` can write and execute code. So the token effectively grants remote code
execution on the Windows machine. Therefore:
- Token must be strong and secret; only trusted networks (LAN / VPN).
- `--host 0.0.0.0` is refused without a token (bind `127.0.0.1` only until a token is set).
- Windows's existing workspace-guard applies to gateway-run agents too.
- No TLS/user-management/rate-limiting (out of scope for single-user trusted-network use).

## Error Handling (no silent swallowing)

- Node offline: `ask ...@windows` → clear error (`cannot reach node 'windows': http://... — is
  'orchestra serve' running?`); other nodes unaffected.
- Wrong token: clear "authentication failed" (401).
- Remote alias error: surfaced back to the Mac with the node's message.
- Scan: non-responding IPs time out and are skipped silently; only real gateways are added.

## Testing Strategy (aiohttp test client + mocked HTTP, no real network)

- Gateway: `/run`/`/aliases`/`/health`; missing/wrong token → 401; `run_ask` mocked.
- `RemoteNodeProvider`: mocked HTTP POST + token header + result parse; `is_available`.
- Node registry + config accessor.
- `@node` resolution: `cdx-deep@windows` → `RemoteNodeProvider`; `@node` vs `@backend` precedence;
  no-`@` backward compatible.
- Scan: mocked subnet, parallel health probes, auto-add to a tmp config.
- CLI: `serve`/`nodes`/`scan`.

## Change Surface (new, isolated)

- New: `orchestra/gateway.py` (serve server), `orchestra/providers/remote_node.py`
  (RemoteNodeProvider), `orchestra/remote_nodes.py` (registry + scan), tests.
- `orchestra/config.py` — accessor for `[remote_nodes.*]`.
- `orchestra/providers/fallback.py` — `@node` fast-path (before the `@backend` one).
- `orchestra/cli.py` — `serve`, `nodes`, `nodes scan` commands.

## Alternatives Considered

| Approach | Why rejected |
|---|---|
| **Copy API keys to the Mac** instead of a gateway | Works for API-key providers (OpenAI/DeepSeek) but NOT for logged-in CLI sessions (Codex/ChatGPT/Claude auth lives on Windows). The gateway is the only way to use those accounts remotely. |
| **Extend the existing `server.py` WebSocket daemon** | It is UI-specific (slots, approval, streaming) and heavy; a purpose-built ~150-line gateway is cleaner and independently testable. |
| **Symmetric peer from the start** | More setup + security surface on both machines; the user needs Mac→Windows first. Symmetry is a later phase. |
| **mDNS/zeroconf auto-discovery** | Orchestra gateways don't advertise via mDNS and adding it is extra moving parts; a parallel LAN scan of the gateway port is simpler and token-filtered. |

## Implementation Notes (post-build)

- **Remote-run cost/model shown as `$0` on the client (v1).** The gateway returns
  `{stdout, model, cost_usd, status}`, but `RemoteNodeProvider.native_run` currently reads only
  `stdout`. So a remote run records the label `remote/<node>/<alias>` with zero cost on the Mac —
  the node's own logs have the real figure. Wiring the returned `cost_usd` into `_last_usage` is a
  later refinement (v1 non-goal, kept simple).
- **`config set` from two separate processes in quick succession can race** (read-modify-write of
  `.orchestra/config.toml`). `nodes-scan` auto-add is unaffected — it writes all found nodes in a
  single process, serially. Avoid firing concurrent `orchestra config set` commands.

## Open Questions

None — all resolved during brainstorming.

# Design: OpenAI-compatible gateway (`/v1/chat/completions`)

## Status
Approved (brainstorming complete, 2026-07-13)

## Problem

The built-in `orchestra chat` REPL is minimal. The user wants to drive Orchestra from a mature
open-source coding UI (opencode, aider, Continue, Cline, LibreChat, …). Those tools all speak the
OpenAI API: you point them at a `base_url` and they call `POST /v1/chat/completions` + `GET
/v1/models`. Orchestra exposes no such endpoint today (its servers are a UI WebSocket, the
remote-node gateway's bespoke `/run`, and MCP stdio).

Adding an OpenAI-compatible surface turns Orchestra into a backend for any of those tools:
the tool provides the polished UI; Orchestra provides its models AND its orchestration
(multi-backend Ollama, remote-node agents, dual/critical routing) behind a standard API.

## Goals

- `POST /v1/chat/completions` (stream + non-stream) and `GET /v1/models`, added to the EXISTING
  aiohttp gateway (`gateway.py`), reusing its bearer-token auth.
- Two kinds of "model" in the tool's picker:
  - **Single aliases** as direct model ids — `oll-fast`, `cdx-deep`, `cdx-deep@windows`,
    `oll-coder@windows-gpu` — resolved to one provider.
  - **Orchestration modes** prefixed `orchestra-` — `orchestra-dual`, `orchestra-critical`,
    `orchestra-auto` — run Orchestra's parallel-agent + review/synthesis flow, returning one
    consolidated assistant reply.
- Multi-turn context: the tool sends the whole `messages` array; the gateway concatenates it into
  one prompt (same portable strategy as the chat REPL).
- Live token streaming for streaming-capable single-model aliases (Ollama, Claude SDK); a single
  chunk otherwise.

## Non-Goals (deferred)

- Multimodal content (image/array `content`) — v1 is plain-text content.
- Function calling / tools API; `/v1/embeddings`.
- Accurate `usage` token accounting — best-effort (0 when unknown).
- Token-by-token streaming for orchestration modes (dual/critical) — they emit one chunk.

## Architecture

Two routes added to `gateway.make_app`; the translation logic lives in a new isolated module
`orchestra/openai_compat.py` so `gateway.py` stays a thin router.

```
opencode → POST /v1/chat/completions {model, messages, stream}
  → auth (existing bearer check)
  → openai_compat.resolve_target(model): alias vs "orchestra-<mode>"
  → openai_compat.messages_to_prompt(messages): system prefix + User/Assistant history
  → run:
       alias  → resolve(model) → provider.stream()/run()  (DIRECT — the chat REPL's light path,
                NOT run_ask; keeps live streaming, no manifest/event machinery)
       mode   → run_dual_async / run_critical_async / run_task(auto)  (full orchestration)
  → openai_compat.to_response(...) or to_sse_stream(...)
  → opencode
```

### 1. Endpoints & model naming

- `POST /v1/chat/completions` — the chat request.
- `GET /v1/models` — `{"object":"list","data":[{"id":..,"object":"model"}, ...]}`: the available
  single aliases (`available_aliases()` rows marked ✓) plus the three `orchestra-*` modes.
- Model id = an Orchestra alias verbatim (`oll-fast`, `cdx-deep@windows`) OR an `orchestra-<mode>`
  string. `resolve_target(model)` returns either `("alias", model)` or `("mode", "dual"|"critical"|"auto")`.

### 2. Request translation (`openai_compat.py`)

- `messages_to_prompt(messages) -> str`: a leading `system` message becomes a `System: …` prefix;
  `user`/`assistant` turns render as `User: …\n\nAssistant: …\n\nUser: …\n\nAssistant:` (reuse the
  chat REPL's rendering approach, extended to prepend system). Plain-text `content` only.
- Dispatch:
  - `orchestra-dual` → `run_dual_async(prompt)`; `orchestra-critical` → `run_critical_async(prompt)`;
    `orchestra-auto` → `run_task(mode="auto", task=prompt)` (wrapped in `asyncio.to_thread` if no
    async variant exists).
  - any other model → treat as a single alias: `resolve(model)` then stream/run (the chat REPL's
    stream-or-run path).
- Reply extraction:
  - single alias → the provider's returned text.
  - orchestration mode → the synthesized result from the `OrchestraRun` (`run.summary` if present,
    else the winning agent's stdout) — one consolidated answer.

### 3. Streaming + response format

- `stream=false`: run to completion; return a single `chat.completion` object:
  `{"id","object":"chat.completion","created","model","choices":[{"index":0,"message":{"role":"assistant","content":..},"finish_reason":"stop"}],"usage":{...}}`.
- `stream=true`: aiohttp `StreamResponse`, `Content-Type: text/event-stream`; each part
  `data: {chat.completion.chunk with choices[0].delta.content}\n\n`, then a final
  `data: {..delta:{}, finish_reason:"stop"}\n\n` and `data: [DONE]\n\n`.
  - single alias + real streaming provider (Ollama, Claude SDK) → one delta chunk per token (live).
  - single alias + one-shot provider (CLI, remote node) → the full reply as one delta chunk.
  - orchestration mode → run completes, synthesis sent as one delta chunk.
- `usage`: filled from the run's token/cost data when available, else zeros. Most tools treat it as
  informational.

### 4. Auth & error handling

- Reuse the gateway's `Authorization: Bearer <token>` check (`hmac.compare_digest`) on
  `/v1/chat/completions` AND `/v1/models`. Same token as `/run`. opencode config:
  `base_url: http://<mac-ip>:7070/v1`, `api_key: <gateway-token>`.
- Errors in OpenAI shape `{"error":{"message","type","code"}}` (tools depend on this):
  unknown model → 404 `invalid_request_error`; invalid JSON / missing `messages` → 400;
  auth fail → 401; run failure → 500 (message surfaced, not swallowed). A mid-stream error is
  emitted as a final delta chunk then `[DONE]` (connection not left hanging).

### 5. Testing (aiohttp test client + mocked Orchestra runs — no real model calls)

- `openai_compat` pure functions: `messages_to_prompt` (system prefix, multi-turn concat);
  `resolve_target` (alias vs `orchestra-<mode>`); `to_response` / `to_sse_stream` shape.
- `/v1/chat/completions`: non-stream success (run mocked); stream success (provider.stream mocked →
  SSE deltas + `[DONE]`); unknown model → 404; missing messages → 400; auth → 401.
- `/v1/models`: lists aliases + `orchestra-*` modes; auth required.

## Change Surface (isolated)

- New: `orchestra/openai_compat.py` (translation + response/SSE formatting), tests.
- `orchestra/gateway.py` — add `/v1/chat/completions` + `/v1/models` routes to `make_app`.
- Reuses `resolve` + `provider.stream/run` directly for single aliases (the chat REPL's light
  path), `run_dual_async`/`run_critical_async`/`run_task` for orchestration modes,
  `available_aliases`, and the chat REPL's history-rendering approach.

## Alternatives Considered

| Approach | Why rejected |
|---|---|
| **Polish the built-in `orchestra chat` REPL** | Reinvents a mature UI poorly; an OpenAI-compatible API lets any existing tool (opencode/aider/Continue) be the UI with near-zero UI code on our side. |
| **A separate `openai-serve` server** | More surface (second server + command); the existing gateway already has aiohttp + auth. One `orchestra serve` is simpler. |
| **Only expose single aliases (no modes)** | Loses Orchestra's core value (dual/critical orchestration). Exposing `orchestra-*` modes as models is the differentiator. |
| **Only expose `orchestra-auto`** | Simplest menu but removes user control over model/backend. Both aliases and modes chosen. |

## Open Questions

None — all resolved during brainstorming.

# Design: Anthropic-compatible gateway (`/v1/messages`)

## Status
Approved (brainstorming complete, 2026-07-14)

## Problem

The just-shipped OpenAI-compatible gateway (`/v1/chat/completions`, `/v1/models`) lets any
OpenAI-API tool (opencode, aider, Continue) drive Orchestra. Claude Code — the tool the user runs
daily — speaks the **Anthropic Messages API** (`POST /v1/messages`), not the OpenAI shape, so it
cannot connect to the current gateway. Pointing Claude Code at Orchestra requires an
Anthropic-compatible surface.

Adding `POST /v1/messages` to the same gateway turns Orchestra into a backend for Claude Code (and
any Anthropic-API tool): the tool provides the UI; Orchestra provides its models AND its
orchestration (multi-backend Ollama, remote-node agents, dual/critical routing) behind the
standard Anthropic API.

## Goals

- `POST /v1/messages` (stream + non-stream) added to the EXISTING aiohttp gateway (`gateway.py`).
- `POST /v1/messages/count_tokens` — best-effort token count so Claude Code's context management
  does not error.
- Auth accepts BOTH `x-api-key: <token>` (Anthropic's native header, sent by Claude Code's
  `ANTHROPIC_API_KEY`) AND `Authorization: Bearer <token>` (same token as `/run` and `/v1/*`).
- Two kinds of "model", mirroring the OpenAI endpoint:
  - **Single aliases** — `oll-fast`, `cdx-deep`, `cdx-deep@windows`, resolved to one provider.
  - **Orchestration modes** — `orchestra-dual`, `orchestra-critical`, `orchestra-auto`.
- **Lenient model handling:** a model that is neither a known alias nor an `orchestra-<mode>`
  (e.g. Claude Code's built-in background model `claude-3-5-haiku`) maps to a configurable default
  alias (`oll-fast`) instead of 404ing, so Claude Code works out of the box (main + background
  calls). This differs from the OpenAI endpoint's strict 404 and is intentional (see Rationale).
- Multi-turn context: the tool sends `system` + the whole `messages` array; the gateway
  concatenates it into one prompt (the portable strategy shared with the chat REPL and OpenAI
  endpoint) via `render_history`.
- Live token streaming for streaming-capable single-model aliases (Ollama, Claude SDK); a single
  delta otherwise.

## Non-Goals (deferred)

- **Tool use / function calling.** Claude Code is agentic — it sends `tools` and expects
  `tool_use` blocks to edit files / run bash. Orchestra models return plain text. v1 ignores the
  `tools` field and returns plain text; Claude Code's agentic file/bash tools will NOT function
  through Orchestra models. v1 is a chat/orchestration surface only. Tool-use translation is a
  separate, much larger project.
- Multimodal content (image/document blocks) — v1 extracts text blocks only; other block types are
  ignored.
- Accurate `usage` / `count_tokens` — best-effort char-based approximation (`len // 4`).
- Token-by-token streaming for orchestration modes (dual/critical) — one delta.
- `GET /v1/models` (Anthropic shape) — YAGNI; Claude Code sends a configured model name and the
  lenient handling accepts any string.

## Architecture

Two routes added to `gateway.make_app`; the Anthropic-specific translation lives in a new
`orchestra/anthropic_compat.py`. The provider-agnostic dispatch is EXTRACTED from `openai_compat.py`
into a neutral `orchestra/gateway_dispatch.py` that both endpoints import, so `gateway.py` stays a
thin router and neither compat module depends on the other.

```
Claude Code → POST /v1/messages {model, system, messages, stream}
  → auth (x-api-key OR Bearer)
  → anthropic_compat.request_to_prompt(system, messages): System prefix + User/Assistant history
  → anthropic_compat.resolve_anthropic_model(model): known alias/mode, else default alias
  → gateway_dispatch.run_target / stream_target   (SHARED with the OpenAI endpoint)
  → anthropic_compat.to_message_response(...) or the Anthropic SSE event sequence
  → Claude Code
```

### 1. Module layout

- **New `orchestra/gateway_dispatch.py`** (extracted from `openai_compat.py`, behaviour-preserving):
  `resolve_target`, `run_target`, `stream_target`, `_reply_from_run`, `_reply_from_manifest`,
  `_start_bg`, `_STREAM_ERROR_TEXT`, module `logger`. `openai_compat.py` imports these from it and
  keeps only OpenAI-specific code (`messages_to_prompt`, `to_response`, `to_sse_chunk`, `sse_done`,
  `list_models`). Existing OpenAI tests must still pass unchanged (import paths updated where they
  patch these names).
- **New `orchestra/anthropic_compat.py`** (Anthropic-specific only):
  - `request_to_prompt(system, messages) -> str`
  - `resolve_anthropic_model(model) -> str` (lenient; returns a normalized model string that
    `run_target`/`stream_target` accept — a known alias or `orchestra-<mode>`; never raises)
  - `to_message_response(model, text, _id=None) -> dict`
  - SSE event formatters: `sse_message_start`, `sse_content_block_start`, `sse_text_delta`,
    `sse_content_block_stop`, `sse_message_delta`, `sse_message_stop` (each returns a
    `event: <type>\ndata: <json>\n\n` string)
  - `approx_tokens(text) -> int` (`len(text) // 4`, min 0)
  - `_DEFAULT_ALIAS = "oll-fast"`
- **`gateway.py`**: `POST /v1/messages`, `POST /v1/messages/count_tokens`; `_authorized` extended
  to accept `x-api-key`.

### 2. Request translation (`anthropic_compat.py`)

Anthropic request: `{model, system?, messages[], max_tokens (required), stream?, tools?, ...}`.

- `system`: a `str` OR a list of `{type:"text", text}` blocks → a leading `System: …` line.
- `messages[].content`: a `str` OR a list of blocks. Extract text from `str` content and from
  `{type:"text", text}` blocks; IGNORE `image`/`tool_use`/`tool_result`/other block types.
- `request_to_prompt` builds the conversation with `render_history` (the shared renderer, already
  tolerant of missing keys) and prepends the `System:` line — identical shape to the OpenAI
  endpoint's `messages_to_prompt`, differing only in the Anthropic top-level `system` and content
  blocks.
- `tools`, `max_tokens`, `metadata` are accepted and ignored.

### 3. Dispatch & reply extraction

- `resolve_anthropic_model(model) -> str`: returns `model` unchanged if it is a valid
  `orchestra-<mode>` OR a known alias (`fallback.known_aliases`, base before any `@`); otherwise
  returns `_DEFAULT_ALIAS`. Never raises.
- The `/v1/messages` handler normalizes the model with `resolve_anthropic_model`, then passes the
  resulting known model string to `gateway_dispatch.run_target` / `stream_target`. Because the
  string is already a known alias/mode, the strict `resolve_target` inside those functions never
  raises (never falls back to the OpenAI-endpoint's 404 path either — that path is unreachable here).
- Reply extraction is inside `run_target`/`stream_target` (unchanged): alias → provider text; mode
  → `run.summary` or last-agent stdout.

### 4. Response & streaming

- **Non-stream** → an Anthropic `message`:
  ```json
  {"id":"msg_<hex>","type":"message","role":"assistant","model":<model>,
   "content":[{"type":"text","text":<reply>}],
   "stop_reason":"end_turn","stop_sequence":null,
   "usage":{"input_tokens":<approx>,"output_tokens":<approx>}}
  ```
- **Stream** → `Content-Type: text/event-stream`, the full Anthropic event sequence (each part
  `event: <type>\ndata: <json>\n\n`):
  1. `message_start` — `{type:"message_start", message:{id,type:"message",role:"assistant",model,
     content:[],stop_reason:null,stop_sequence:null,usage:{input_tokens:<approx>,output_tokens:0}}}`
  2. `content_block_start` — `{type:"content_block_start", index:0, content_block:{type:"text",text:""}}`
  3. per chunk: `content_block_delta` — `{type:"content_block_delta", index:0, delta:{type:"text_delta", text:<chunk>}}`
     - single alias + streaming provider (Ollama, Claude SDK) → one delta per token (live)
     - single alias + one-shot provider, or orchestration mode → the full reply as one delta
  4. `content_block_stop` — `{type:"content_block_stop", index:0}`
  5. `message_delta` — `{type:"message_delta", delta:{stop_reason:"end_turn", stop_sequence:null}, usage:{output_tokens:<approx>}}`
  6. `message_stop` — `{type:"message_stop"}`
- **Streaming discipline (reuse the hardened OpenAI path):** the handler consumes
  `gateway_dispatch.stream_target` (which already stops its producer thread on `aclose` and emits a
  generic `[stream error]` text, never raw exceptions). The handler wraps the loop so it ALWAYS
  emits `content_block_stop` + `message_delta` + `message_stop` (terminal events) even on a
  provider error, and on client disconnect suppresses writes and `aclose()`s the generator
  (deterministic producer stop). This mirrors the C1/C2 fixes from the OpenAI endpoint.

### 5. Auth & error handling

- `_authorized(request, token)` extended: authorized if `token` is empty, OR the `x-api-key` header
  equals `token`, OR `Authorization: Bearer <token>` matches. Byte comparison (`hmac.compare_digest`
  on UTF-8 bytes, as fixed for the Bearer path). Applied to `/v1/messages` and
  `/v1/messages/count_tokens`. The `anthropic-version` header is accepted and ignored. Existing
  `/run`, `/aliases`, `/v1/*` behaviour is unchanged (they keep working via the same helper).
- Errors in Anthropic shape `{"type":"error","error":{"type":<class>,"message":<msg>}}`:
  - bad/missing JSON, missing `messages` → 400 `invalid_request_error`
  - auth fail → 401 `authentication_error`
  - run failure → 500 `api_error`, message GENERIC (`"internal error running the model"`), full
    exception logged server-side only (no PII/internal leak — mirrors OpenAI I1)
  - There is NO 404 model path (lenient fallback).
- `count_tokens`: `POST /v1/messages/count_tokens` → `{"input_tokens": approx_tokens(prompt)}`.
- `usage`: `input_tokens = approx_tokens(prompt)`, `output_tokens = approx_tokens(reply)`.

## Testing (aiohttp TestClient/TestServer + mocked dispatch — no real model calls)

- `anthropic_compat` pure functions: `request_to_prompt` (system str + text-block list; content str
  + text-block list; ignored image/tool blocks); `resolve_anthropic_model` (known alias → itself,
  known mode, unknown → default); `to_message_response` shape; each SSE formatter's
  `event:`/`data:` shape; `approx_tokens`.
- `gateway_dispatch` extraction: existing `test_openai_compat.py` / `test_openai_gateway.py` pass
  unchanged (dispatch behaviour preserved); patch targets updated to `gateway_dispatch` where they
  patched `resolve`/`run_target`/`stream_target`.
- `/v1/messages`: non-stream success (mock `run_target`) → message shape; stream success (mock
  `stream_target`) → the ordered event sequence (`message_start` … `message_stop`); auth x-api-key
  200 / missing 401 / Bearer 200; bad JSON 400; missing `messages` 400; run failure 500 (generic
  body, secret string ABSENT); lenient unknown model → `run_target` called with `_DEFAULT_ALIAS`.
- `/v1/messages/count_tokens`: returns `input_tokens`.
- Live smoke: Claude Code with `ANTHROPIC_BASE_URL=http://127.0.0.1:7070`,
  `ANTHROPIC_API_KEY=orchestra-lan`, `ANTHROPIC_MODEL=oll-fast`; `claude -p "…"` headless → a reply
  flows through Orchestra → Ollama with no traceback.

## Change Surface (isolated)

- New: `orchestra/gateway_dispatch.py` (extracted dispatch), `orchestra/anthropic_compat.py`
  (translation + response/SSE), tests.
- Modified: `orchestra/openai_compat.py` (import dispatch from `gateway_dispatch`),
  `orchestra/gateway.py` (add `/v1/messages` + `count_tokens` routes; extend `_authorized`),
  and OpenAI tests' patch targets.
- Reuses: `gateway_dispatch.run_target`/`stream_target`, `fallback.known_aliases`, `render_history`.

## Rationale: lenient model handling (vs the OpenAI endpoint's strict 404)

The OpenAI endpoint 404s an unknown model because the tool shows a model PICKER populated from
`/v1/models`, so an unknown id is a genuine user error. Claude Code sends model names
PROGRAMMATICALLY — a main model plus a built-in background/"small fast" model
(`claude-3-5-haiku`) — that the user does not control per-call. 404ing those would break Claude
Code's background tasks. Mapping unknown names to a configurable default alias makes Claude Code
work out of the box. The two endpoints differ deliberately; each matches how its clients address
models.

## Alternatives Considered

| Approach | Why rejected |
|---|---|
| **`anthropic_compat` imports dispatch from `openai_compat`** | Conceptually backwards (Anthropic depending on the OpenAI module). Extracting a neutral `gateway_dispatch` is the correct boundary and keeps both endpoints symmetric. |
| **A separate Anthropic server/port** | More surface (second server + command); the existing gateway already has aiohttp + auth. |
| **Strict 404 on unknown model (parity with OpenAI)** | Breaks Claude Code's background `claude-3-5-haiku` calls unless the user sets both `ANTHROPIC_MODEL` and `ANTHROPIC_SMALL_FAST_MODEL`. Lenient default is better UX for this client. |
| **Implement tool_use translation in v1** | Orchestra models return plain text, not structured tool calls; faithful translation is a large, separate project. v1 is a text/orchestration surface. |

## Open Questions

None — all resolved during brainstorming.

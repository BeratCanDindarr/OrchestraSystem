# Design: Tool use on the Anthropic gateway (Ollama native messages)

## Status
Approved (brainstorming complete, 2026-07-14)

## Problem

The Anthropic-compatible gateway (`/v1/messages`) is text-only: the `tools` field is ignored, so
Claude Code pointed at Orchestra cannot edit files or run bash — its agentic loop needs the model
to return structured `tool_use` content blocks and to receive `tool_result` blocks back.

The originally-planned first provider (claude_sdk passthrough) requires a real `ANTHROPIC_API_KEY`,
which the user does not have (a Claude Code subscription is NOT an API key, cannot legitimately
authorize raw `/v1/messages` calls, and the `claude` CLI runs its own agentic loop — it never
returns `tool_use` blocks to a caller). The API-key-free path to a fully agentic Claude Code is
**Ollama's `/api/chat` tool calling**: `oll-kimi`/`oll-minimax` (`:cloud`, strong tool callers,
zero local RAM) and local 3B models.

## Goals

- Claude Code pointed at the gateway becomes FULLY AGENTIC (file edits, bash) when the model is a
  tool-capable Ollama alias: `tool_use` blocks flow out, `tool_result` blocks round-trip back.
- A general, provider-agnostic capability mechanism so future providers (claude_sdk when a key
  exists, openai_sdk) plug in WITHOUT gateway changes.
- Anthropic-shaped payloads as the INTERNAL COMMON LANGUAGE: providers receive an Anthropic-style
  request dict and return an Anthropic-style response dict; all translation lives inside the
  provider.
- Graceful degradation: `tools` + non-capable provider → the existing text path (tools ignored),
  exactly today's behavior. Orchestration modes (`orchestra-*`) always take the text path.
- Existing text-path behavior is UNCHANGED (all current gateway/compat/dispatch tests keep passing).

## Non-Goals (deferred)

- True incremental `input_json_delta` streaming — v1 buffers: the provider is called
  non-streaming and the result is expanded into a spec-compliant synthetic SSE sequence (whole
  tool input as ONE `input_json_delta`). Valid per the Messages API stream contract.
- claude_sdk / openai_sdk `native_messages` implementations (the capability mechanism leaves the
  door open; each is a later provider-only change).
- `tool_choice` enforcement beyond passthrough; image blocks inside `tool_result` (Claude Code
  screenshots); `/v1/chat/completions` (OpenAI endpoint) tool calling.
- Making orchestration modes tool-capable (they synthesize multi-agent output; structurally
  cannot emit tool calls).

## Architecture

```
Claude Code → POST /v1/messages {model, system, messages, tools, stream}
  → auth (unchanged)
  → tools present AND resolved provider supports_native_messages?
      YES → gateway_dispatch.run_native(model, payload)
              → provider.native_messages(payload)   # Anthropic-in, Anthropic-out
                  OllamaProvider: translate → POST /api/chat (messages+tools+num_ctx)
                                  ← message.content + message.tool_calls → content blocks
              → non-stream: normalized Anthropic message response
              → stream: buffered synthetic SSE (text + tool_use blocks, stop_reason preserved)
      NO  → existing text path (tools ignored) — unchanged
```

### 1. Capability mechanism (`providers/base.py`)

Mirrors the existing optional-`stream()` pattern:

```python
supports_native_messages: bool = False   # class attribute

def native_messages(self, payload: dict, timeout: int = 180, **kwargs) -> dict:
    """Anthropic-shaped request in, Anthropic-shaped response dict out.
    Only providers that can produce structured tool calls implement this."""
    raise NotImplementedError(f"{self.name} does not support native messages")
```

`payload` is the (sanitized) client request: `{model?, system?, messages, tools?, tool_choice?,
max_tokens?}`. The return dict carries at minimum: `content` (list of Anthropic content blocks —
`{"type":"text",...}` and/or `{"type":"tool_use","id","name","input"}`), `stop_reason`
(`"end_turn"` | `"tool_use"`), and optional `usage` (`input_tokens`/`output_tokens`).

### 2. Ollama translation (inside `providers/ollama.py`)

`OllamaProvider.supports_native_messages = True`; `native_messages(payload)`:

- **Request translation (Anthropic → Ollama `/api/chat`):**
  - `system` (str or text blocks) → leading `{"role":"system","content":...}` message.
  - Per message: str content → passthrough; block-list content →
    - `text` blocks → concatenated `content`,
    - assistant `tool_use` blocks → `tool_calls: [{"function":{"name","arguments":input}}]`,
    - user `tool_result` blocks → separate `{"role":"tool","content":<result text>}` messages
      (result content: str, or text-block list concatenated). Correlation is by ORDER (Ollama has
      no tool-call ids): the Nth `tool_result` answers the Nth emitted `tool_call` of the previous
      assistant turn.
  - `tools` → Ollama tools: `{"type":"function","function":{"name","description",
    "parameters": <input_schema>}}` (mechanical rename).
  - `options.num_ctx` set from config (`[providers.ollama] num_ctx`, default **32768**) — Claude
    Code's system prompt + tool schemas overflow Ollama's small default context, which would make
    the model blind to its tools.
  - `stream: false` (v1 buffers).
- **Response translation (Ollama → Anthropic):**
  - `message.content` (non-empty) → one `{"type":"text","text":...}` block.
  - each `message.tool_calls[i]` → `{"type":"tool_use","id":"toolu_<uuid-hex>","name":
    fn.name,"input": fn.arguments}` (synthetic ids, since Ollama provides none).
  - `stop_reason`: `"tool_use"` if any tool_calls else `"end_turn"`.
  - `usage` from `prompt_eval_count`/`eval_count` when present.
- **Errors:** HTTP/model errors (including "model does not support tools") raise; the gateway maps
  them to a GENERIC Anthropic 500 (`api_error`) with the full exception in server logs only —
  same PII discipline as the text path. No silent fallback to the text path mid-session.

### 3. Dispatch (`gateway_dispatch.py`)

```python
async def run_native(model: str, payload: dict) -> dict:
    """Run an Anthropic-shaped payload through a native-messages-capable provider."""
    kind, _ = resolve_target(model)          # modes never reach here (gateway guards)
    provider, effort = resolve(model)
    if not getattr(provider, "supports_native_messages", False):
        raise NotImplementedError(...)        # gateway pre-checks; this is a safety net
    # resolve() returns (provider, effort_or_model); the provider's own MODELS mapping
    # turns the effort string into the concrete model name (existing pattern) — so we
    # pass the effort through and let native_messages resolve it internally.
    resolved_payload = {**payload, "model": effort}
    return await asyncio.to_thread(provider.native_messages, resolved_payload)
```

No fallback chain on the native path: a mid-session swap to a text-only provider would break
Claude Code's tool loop.

### 4. Gateway routing + buffered SSE (`gateway.py` + `anthropic_compat.py`)

- `_v1_messages`: after auth/validation and lenient model resolution, IF `payload.get("tools")`
  AND the resolved provider `supports_native_messages` (checked via a small
  `gateway_dispatch.native_capable(model) -> bool` helper) → native path; ELSE existing text path
  (tools ignored). Orchestration modes always text path.
- **Non-stream native:** `run_native` result → normalized full Anthropic `message` envelope
  (id `msg_…`, `role:"assistant"`, provider's content blocks + stop_reason + usage passthrough,
  zeros when absent).
- **Stream native** (`_v1_messages_stream_native`): call `run_native` (buffered), then emit the
  synthetic sequence:
  1. `message_start`
  2. per content block i: `content_block_start(index=i, block)` — text block starts with
     `{"type":"text","text":""}` + one `text_delta`; tool_use block starts with
     `{"type":"tool_use","id","name","input":{}}` + ONE `input_json_delta`
     (`partial_json=json.dumps(input)`) — then `content_block_stop(index=i)`.
  3. `message_delta` with the PROVIDER'S `stop_reason` (`"tool_use"` when tools were called —
     Claude Code's loop keys off this) + usage.
  4. `message_stop`.
  Same hardening as the text path: terminal events always emitted, writes in
  `contextlib.suppress(ConnectionResetError, ClientError)`, errors logged server-side + generic
  `[stream error]` text delta, no client-facing exception text.
- `anthropic_compat.py` gains multi-block/typed SSE formatters (current ones are text-index-0
  only): `sse_content_block_start(index, block)`, `sse_input_json_delta(index, partial_json)`,
  `sse_content_block_stop(index)`, `sse_message_delta(stop_reason, output_tokens)` — the existing
  text-path formatters remain for the text path (unchanged behavior).

## Testing

- **Translation pure functions (ollama):** Anthropic→Ollama messages (system prefix; text
  passthrough; tool_use→tool_calls; tool_result→role:"tool" with order correlation; mixed
  text+tool_result user turn), tools schema rename, num_ctx injection; Ollama→Anthropic
  (text block; tool_calls→tool_use with synthetic `toolu_` ids; stop_reason mapping; usage).
- **Capability:** base flag False + native_messages raises; OllamaProvider flag True.
- **Dispatch:** run_native happy path (mock provider); non-capable provider → NotImplementedError;
  no fallback-chain invocation.
- **Gateway:** tools + capable → native path used (mock `run_native`), response envelope correct;
  tools + non-capable → text path (existing behavior, `run_target` called); no tools → text path;
  native stream → golden event sequence with a text block AND a tool_use block, `stop_reason:
  "tool_use"` in message_delta, `[DONE]`-less Anthropic termination (message_stop), terminal
  events on error; native run failure → generic 500, secret text ABSENT.
- **Regression:** the full existing gateway/compat/dispatch suites pass unchanged.
- **Live smoke (acceptance):**
  `ANTHROPIC_BASE_URL=http://127.0.0.1:7070 ANTHROPIC_API_KEY=orchestra-lan ANTHROPIC_MODEL=oll-kimi
  claude -p "create hello.txt containing the word merhaba"` run in a scratch dir → the file is
  ACTUALLY CREATED (tool_use round-trip proven). Second smoke with local `qwen2.5:3b` — if the 3B
  model's tool quality is poor, that is documented; acceptance rides on `oll-kimi`.

## Change Surface

- `orchestra/providers/base.py` — capability flag + `native_messages()` (optional, like `stream()`).
- `orchestra/providers/ollama.py` — `native_messages` + translation helpers (module may warrant a
  small `ollama_native.py` sibling if it pushes past size limits).
- `orchestra/gateway_dispatch.py` — `run_native`, `native_capable`.
- `orchestra/anthropic_compat.py` — typed/multi-block SSE formatters.
- `orchestra/gateway.py` — native-path branch in `_v1_messages` + `_v1_messages_stream_native`.
- `orchestra/config.py` accessor for `[providers.ollama] num_ctx` if not already readable via
  `providers_config()` (it is — no new accessor needed; read with a 32768 default).
- Tests: new `tests/test_ollama_native.py`, `tests/test_gateway_native.py`; additions to
  `tests/test_anthropic_compat.py`.

## Alternatives Considered

| Approach | Why rejected |
|---|---|
| **claude_sdk passthrough first (original plan)** | Requires a real ANTHROPIC_API_KEY the user doesn't have; a Claude Code subscription cannot legitimately authorize raw API calls, and the `claude` CLI never returns tool_use blocks (it runs its own loop). Door stays open via the capability mechanism. |
| **Translation in the gateway, providers speak raw formats** | The gateway would need to know every provider's wire format; the second provider (openai_sdk) would pile a second translation into gateway code. Provider-internal translation scales. |
| **True incremental tool streaming in v1** | Fiddly (`input_json_delta` chunking) for marginal UX; buffered synthetic SSE is spec-valid and Claude Code-compatible. Deferred. |
| **Silent fallback to text path on native errors** | A mid-session downgrade breaks Claude Code's tool loop invisibly. Fail loud (generic 500, logged) instead. |

## Open Questions

None — all resolved during brainstorming.

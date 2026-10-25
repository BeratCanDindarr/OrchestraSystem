# Gateway Tool Use (Ollama native messages) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Claude Code pointed at the Orchestra gateway becomes fully agentic (file edits, bash) through tool-capable Ollama models: `tool_use` blocks flow out of `/v1/messages`, `tool_result` blocks round-trip back.

**Architecture:** An optional provider capability (`supports_native_messages` + `native_messages(payload)->dict`, Anthropic-shaped in/out) mirroring the existing optional `stream()` pattern; Ollama implements it against `/api/chat` with all translation inside the provider layer (pure helpers in a new `ollama_native.py`); the gateway routes `tools`-bearing requests to a native path with buffered synthetic SSE.

**Tech Stack:** Python 3.11, urllib (existing Ollama HTTP pattern), aiohttp gateway. Tests: pytest + `aiohttp.test_utils.TestClient/TestServer` + `@pytest.mark.asyncio`, mocked `urllib.request.urlopen` for provider HTTP.

## Global Constraints

- Pure Python, no new dependencies.
- Anthropic-shaped payloads are the INTERNAL COMMON LANGUAGE: `native_messages` receives an Anthropic-style request dict and returns an Anthropic-style response dict (`content` blocks + `stop_reason` + optional `usage`). All Ollama translation lives in the provider layer, NOT the gateway.
- Graceful degradation: `tools` + non-capable provider → existing text path (tools ignored). Orchestration modes (`orchestra-*`) ALWAYS take the text path. No-tools requests are untouched.
- NO fallback chain on the native path (a mid-session swap to a text-only provider breaks Claude Code's tool loop). Native errors → generic Anthropic 500 `api_error`, full exception server-side logs only (no raw `str(exc)` to clients).
- Streaming is BUFFERED synthetic SSE in v1: provider called non-streaming, result expanded to a spec-compliant event sequence (whole tool input as ONE `input_json_delta`); the provider's `stop_reason` (esp. `"tool_use"`) MUST be preserved in `message_delta`.
- Synthetic tool ids: `toolu_<uuid4-hex-24>`; `tool_result`→`role:"tool"` correlation is by ORDER (Ollama has no ids).
- `options.num_ctx` from `[providers.ollama] num_ctx` config, default **32768**.
- Existing text-path behavior UNCHANGED — all current gateway/compat/dispatch suites keep passing.
- Functions <50 lines; files <800 lines. No silent `except: pass`. Conventional Commits, NO Co-Authored-By.
- Run pytest from a scratch dir: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest <abs paths> -q --no-cov -p no:cacheprovider -o addopts=""`. If `tests/baselines/performance_baseline.json` shows modified before a commit, `git checkout --` it.

---

### Task 1: Provider capability flag (`base.py`)

**Files:**
- Modify: `orchestra/providers/base.py` (append after `stream()`, line 57)
- Test: `tests/test_native_messages_capability.py` (create)

**Interfaces:**
- Produces: `BaseProvider.supports_native_messages: bool = False` (class attribute) and `BaseProvider.native_messages(payload: dict, timeout: int = 180, **kwargs) -> dict` raising NotImplementedError. Tasks 3-6 rely on these exact names.

- [ ] **Step 1: Write the failing test** — create `tests/test_native_messages_capability.py`:

```python
import pytest


def _concrete_base_provider():
    from orchestra.providers.base import BaseProvider

    class _P(BaseProvider):
        name = "stub"
        def is_available(self): return True
        def model_label(self, e): return f"stub/{e}"
        def build_command(self, p, e): return ["echo", p]
    return _P()


def test_base_provider_not_native_capable_by_default():
    p = _concrete_base_provider()
    assert p.supports_native_messages is False


def test_base_native_messages_raises_not_implemented():
    p = _concrete_base_provider()
    with pytest.raises(NotImplementedError):
        p.native_messages({"messages": [{"role": "user", "content": "hi"}]})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_native_messages_capability.py -q --no-cov -p no:cacheprovider -o addopts=""`
Expected: FAIL (`AttributeError: ... 'supports_native_messages'`).

- [ ] **Step 3: Implement** — in `orchestra/providers/base.py`, add a class attribute next to `_last_usage` (line 12) and a method after `stream()`:

```python
    supports_native_messages: bool = False
```

```python
    def native_messages(self, payload: dict, timeout: int = 180, **kwargs) -> dict:
        """Anthropic-shaped request in, Anthropic-shaped response dict out.

        `payload`: {model?, system?, messages, tools?, tool_choice?, max_tokens?}.
        Returns at minimum: {"content": [<Anthropic content blocks>],
        "stop_reason": "end_turn"|"tool_use", "usage": {...}?}.
        Only providers that can emit structured tool calls implement this.
        """
        raise NotImplementedError(f"Provider {self.name} does not support native messages")
```

- [ ] **Step 4: Run to verify it passes** (same command). Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add orchestra/providers/base.py tests/test_native_messages_capability.py
git commit -m "feat(providers): optional native_messages capability (Anthropic-shaped in/out)"
```

---

### Task 2: Ollama translation pure functions (`ollama_native.py`)

**Files:**
- Create: `orchestra/providers/ollama_native.py`
- Test: `tests/test_ollama_native.py` (create)

**Interfaces:**
- Produces (Task 3 consumes): `to_ollama_messages(system, messages) -> list[dict]`, `to_ollama_tools(tools) -> list[dict]`, `from_ollama_response(message: dict) -> tuple[list[dict], str]`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_ollama_native.py`:

```python
from orchestra.providers.ollama_native import (
    to_ollama_messages, to_ollama_tools, from_ollama_response,
)


def test_to_ollama_messages_system_and_text():
    msgs = to_ollama_messages("be terse", [{"role": "user", "content": "hi"}])
    assert msgs == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]


def test_to_ollama_messages_tool_use_becomes_tool_calls():
    messages = [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "ok"},
            {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"cmd": "ls"}},
        ]},
    ]
    out = to_ollama_messages(None, messages)
    assert out[0] == {"role": "user", "content": "list files"}
    assert out[1]["role"] == "assistant" and out[1]["content"] == "ok"
    assert out[1]["tool_calls"] == [{"function": {"name": "bash", "arguments": {"cmd": "ls"}}}]


def test_to_ollama_messages_tool_result_becomes_role_tool():
    messages = [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file1\nfile2"},
            {"type": "text", "text": "now summarize"},
        ]},
    ]
    out = to_ollama_messages(None, messages)
    # tool results come FIRST (they answer the previous assistant turn), then user text
    assert out[0] == {"role": "tool", "content": "file1\nfile2"}
    assert out[1] == {"role": "user", "content": "now summarize"}


def test_to_ollama_messages_tool_result_block_content():
    messages = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t", "content": [{"type": "text", "text": "out"}]},
    ]}]
    assert to_ollama_messages(None, messages) == [{"role": "tool", "content": "out"}]


def test_to_ollama_tools_schema_rename():
    tools = [{"name": "bash", "description": "run", "input_schema": {"type": "object"}}]
    assert to_ollama_tools(tools) == [{
        "type": "function",
        "function": {"name": "bash", "description": "run", "parameters": {"type": "object"}},
    }]


def test_from_ollama_response_text_only():
    blocks, stop = from_ollama_response({"content": "hello"})
    assert blocks == [{"type": "text", "text": "hello"}]
    assert stop == "end_turn"


def test_from_ollama_response_tool_calls():
    blocks, stop = from_ollama_response({
        "content": "",
        "tool_calls": [{"function": {"name": "bash", "arguments": {"cmd": "ls"}}}],
    })
    assert stop == "tool_use"
    assert len(blocks) == 1
    b = blocks[0]
    assert b["type"] == "tool_use" and b["name"] == "bash" and b["input"] == {"cmd": "ls"}
    assert b["id"].startswith("toolu_")


def test_from_ollama_response_string_arguments_parsed():
    blocks, _ = from_ollama_response({
        "content": "", "tool_calls": [{"function": {"name": "f", "arguments": "{\"a\": 1}"}}],
    })
    assert blocks[0]["input"] == {"a": 1}
```

- [ ] **Step 2: Run to verify they fail** (ModuleNotFoundError expected).

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_ollama_native.py -q --no-cov -p no:cacheprovider -o addopts=""`

- [ ] **Step 3: Create `orchestra/providers/ollama_native.py`:**

```python
"""Pure translation between Anthropic Messages payloads and Ollama /api/chat.

Kept as pure functions so the wire translation is unit-testable without HTTP.
Correlation note: Ollama tool calls carry no ids, so tool_use ids are synthesized
(`toolu_<hex>`) and tool_result -> role:"tool" correlation is by ORDER."""
from __future__ import annotations

import json
import uuid


def _block_text(content) -> str:
    """Text of a str-or-block-list content value (text blocks only)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _system_text(system) -> str:
    if not system:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n\n".join(
            b.get("text", "") for b in system
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def to_ollama_tools(tools: list[dict]) -> list[dict]:
    """Anthropic tool definitions -> Ollama (OpenAI-style) function tools."""
    return [{
        "type": "function",
        "function": {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {}),
        },
    } for t in tools]


def _convert_block_message(role: str, blocks: list) -> list[dict]:
    """One Anthropic block-list message -> one or more Ollama chat messages."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    tool_results: list[str] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        btype = b.get("type")
        if btype == "text":
            text_parts.append(b.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append({"function": {
                "name": b.get("name", ""), "arguments": b.get("input", {}),
            }})
        elif btype == "tool_result":
            tool_results.append(_block_text(b.get("content", "")))
    out: list[dict] = []
    if role == "assistant":
        msg: dict = {"role": "assistant", "content": "".join(text_parts)}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        out.append(msg)
        return out
    # user turn: tool results answer the PREVIOUS assistant tool_calls -> emit first
    for result_text in tool_results:
        out.append({"role": "tool", "content": result_text})
    if text_parts:
        out.append({"role": "user", "content": "".join(text_parts)})
    return out


def to_ollama_messages(system, messages: list[dict]) -> list[dict]:
    """Anthropic system+messages -> Ollama /api/chat messages list."""
    out: list[dict] = []
    sys_txt = _system_text(system)
    if sys_txt:
        out.append({"role": "system", "content": sys_txt})
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            out.extend(_convert_block_message(role, content))
        else:
            out.append({"role": role, "content": content if isinstance(content, str) else ""})
    return out


def from_ollama_response(message: dict) -> tuple[list[dict], str]:
    """Ollama /api/chat response `message` -> (Anthropic content blocks, stop_reason)."""
    blocks: list[dict] = []
    text = message.get("content") or ""
    if text:
        blocks.append({"type": "text", "text": text})
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {"_raw": args}
        blocks.append({
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:24]}",
            "name": fn.get("name", ""),
            "input": args if isinstance(args, dict) else {"_raw": args},
        })
    stop_reason = "tool_use" if tool_calls else "end_turn"
    return blocks, stop_reason
```

- [ ] **Step 4: Run to verify all pass** (same command). Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add orchestra/providers/ollama_native.py tests/test_ollama_native.py
git commit -m "feat(ollama): pure Anthropic<->Ollama chat translation helpers"
```

---

### Task 3: `OllamaProvider.native_messages` (HTTP `/api/chat`)

**Files:**
- Modify: `orchestra/providers/ollama.py` (add after `stream()`, ~line 180)
- Test: append to `tests/test_ollama_native.py`

**Interfaces:**
- Consumes: Task 2 helpers; existing `MODELS`, `self._base_url()`, `config.providers_config()`.
- Produces (Task 4 consumes): `OllamaProvider.supports_native_messages = True`; `native_messages(payload, timeout=180, **kwargs) -> dict` with keys `content`, `stop_reason`, `usage`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_ollama_native.py`:

```python
import io
import json as _j
from unittest.mock import patch

from orchestra.providers.ollama import OllamaProvider


def _fake_urlopen_factory(response_dict, captured):
    class _Resp(io.BytesIO):
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def _fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = _j.loads(req.data.decode())
        return _Resp(_j.dumps(response_dict).encode())
    return _fake_urlopen


def test_ollama_provider_is_native_capable():
    assert OllamaProvider.supports_native_messages is True


def test_native_messages_posts_api_chat_and_translates():
    captured: dict = {}
    fake = _fake_urlopen_factory({
        "message": {"content": "", "tool_calls": [
            {"function": {"name": "bash", "arguments": {"cmd": "ls"}}}]},
        "prompt_eval_count": 12, "eval_count": 3,
    }, captured)
    p = OllamaProvider()
    with patch("orchestra.providers.ollama.urllib.request.urlopen", fake):
        result = p.native_messages({
            "model": "fast",
            "system": "be terse",
            "messages": [{"role": "user", "content": "list files"}],
            "tools": [{"name": "bash", "description": "run", "input_schema": {"type": "object"}}],
        })
    assert captured["url"].endswith("/api/chat")
    assert captured["body"]["model"] == "qwen2.5:3b"          # effort resolved via MODELS
    assert captured["body"]["stream"] is False
    assert captured["body"]["options"]["num_ctx"] == 32768     # default
    assert captured["body"]["messages"][0] == {"role": "system", "content": "be terse"}
    assert captured["body"]["tools"][0]["function"]["name"] == "bash"
    assert result["stop_reason"] == "tool_use"
    assert result["content"][0]["type"] == "tool_use"
    assert result["usage"] == {"input_tokens": 12, "output_tokens": 3}


def test_native_messages_http_error_raises():
    import urllib.error
    def _boom(req, timeout=0):
        raise urllib.error.URLError("connection refused")
    p = OllamaProvider()
    with patch("orchestra.providers.ollama.urllib.request.urlopen", _boom):
        import pytest
        with pytest.raises(Exception):
            p.native_messages({"model": "fast", "messages": [{"role": "user", "content": "x"}]})
```

- [ ] **Step 2: Run to verify they fail** (AttributeError: no `supports_native_messages` True / no `native_messages` behavior).

- [ ] **Step 3: Implement** — in `orchestra/providers/ollama.py`, inside `OllamaProvider`, add the class attribute right under `name = "ollama"` and the method after `stream()`:

```python
    supports_native_messages = True
```

```python
    def native_messages(self, payload: dict, timeout: int = 180, **kwargs) -> dict:
        """Anthropic-shaped request -> POST /api/chat -> Anthropic-shaped response.

        Errors RAISE (no silent text fallback): the gateway maps them to a generic
        Anthropic 500 and logs the detail server-side.
        """
        from orchestra.providers import ollama_native as _native
        model = MODELS.get(payload.get("model", ""), payload.get("model", ""))
        num_ctx = int(config.providers_config().get("ollama", {}).get("num_ctx", 32768))
        body: dict = {
            "model": model,
            "messages": _native.to_ollama_messages(payload.get("system"), payload.get("messages", [])),
            "stream": False,
            "options": {"num_ctx": num_ctx},
        }
        if payload.get("tools"):
            body["tools"] = _native.to_ollama_tools(payload["tools"])
        req = urllib.request.Request(
            f"{self._base_url()}/api/chat",
            data=_json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = _json.loads(resp.read())
        except urllib.error.HTTPError as exc:  # surface body detail to SERVER logs only
            detail = exc.read().decode(errors="replace")[:500]
            raise RuntimeError(f"ollama /api/chat {exc.code}: {detail}") from exc
        blocks, stop_reason = _native.from_ollama_response(data.get("message", {}) or {})
        return {
            "content": blocks,
            "stop_reason": stop_reason,
            "usage": {
                "input_tokens": int(data.get("prompt_eval_count") or 0),
                "output_tokens": int(data.get("eval_count") or 0),
            },
        }
```

- [ ] **Step 4: Run to verify all pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_ollama_native.py -q --no-cov -p no:cacheprovider -o addopts=""`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add orchestra/providers/ollama.py tests/test_ollama_native.py
git commit -m "feat(ollama): native_messages via /api/chat with tool calling"
```

---

### Task 4: Dispatch — `run_native` + `native_capable` (`gateway_dispatch.py`)

**Files:**
- Modify: `orchestra/gateway_dispatch.py` (append after `run_target`)
- Test: append to `tests/test_gateway_dispatch.py`

**Interfaces:**
- Consumes: existing `resolve`, `resolve_target`.
- Produces (Task 6 consumes): `native_capable(model: str) -> bool`; `async run_native(model: str, payload: dict) -> dict`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_gateway_dispatch.py` (match the file's existing import style; it already has `import asyncio`, `patch`, `gd` alias):

```python
class _NativeProvider:
    supports_native_messages = True
    name = "native-stub"
    def native_messages(self, payload, timeout=180, **kwargs):
        return {"content": [{"type": "text", "text": f"ran {payload['model']}"}],
                "stop_reason": "end_turn", "usage": {}}


class _TextOnlyProvider:
    supports_native_messages = False
    name = "text-stub"


def test_native_capable_true_for_capable_alias():
    with patch("orchestra.gateway_dispatch.resolve", return_value=(_NativeProvider(), "fast")):
        assert gd.native_capable("oll-fast") is True


def test_native_capable_false_for_text_provider_and_modes():
    with patch("orchestra.gateway_dispatch.resolve", return_value=(_TextOnlyProvider(), "x")):
        assert gd.native_capable("cdx-deep") is False
    assert gd.native_capable("orchestra-dual") is False


def test_run_native_passes_effort_as_model():
    with patch("orchestra.gateway_dispatch.resolve", return_value=(_NativeProvider(), "fast")):
        result = asyncio.run(gd.run_native("oll-fast", {"messages": [{"role": "user", "content": "hi"}]}))
    assert result["content"][0]["text"] == "ran fast"


def test_run_native_rejects_non_capable():
    import pytest
    with patch("orchestra.gateway_dispatch.resolve", return_value=(_TextOnlyProvider(), "x")):
        with pytest.raises(NotImplementedError):
            asyncio.run(gd.run_native("cdx-deep", {"messages": []}))
```

- [ ] **Step 2: Run to verify they fail** (AttributeError: no `native_capable`).

- [ ] **Step 3: Implement** — append to `orchestra/gateway_dispatch.py`:

```python
def native_capable(model: str) -> bool:
    """True when `model` is a single alias whose provider can run Anthropic-shaped
    payloads natively (structured tool calls). Modes and unknown models -> False."""
    try:
        kind, _ = resolve_target(model)
    except ValueError:
        return False
    if kind != "alias":
        return False
    provider, _ = resolve(model)
    return bool(getattr(provider, "supports_native_messages", False))


async def run_native(model: str, payload: dict) -> dict:
    """Run an Anthropic-shaped payload through a native-messages-capable provider.

    NO fallback chain here: a mid-session swap to a text-only provider would
    silently break the client's tool loop. Non-capable providers raise."""
    provider, effort = resolve(model)
    if not getattr(provider, "supports_native_messages", False):
        raise NotImplementedError(
            f"provider {getattr(provider, 'name', '?')} does not support native messages"
        )
    resolved = {**payload, "model": effort}
    return await asyncio.to_thread(provider.native_messages, resolved)
```

- [ ] **Step 4: Run to verify all pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_gateway_dispatch.py -q --no-cov -p no:cacheprovider -o addopts=""`
Expected: 15 passed (11 existing + 4 new).

- [ ] **Step 5: Commit**

```bash
git add orchestra/gateway_dispatch.py tests/test_gateway_dispatch.py
git commit -m "feat(dispatch): run_native + native_capable (no fallback on the native path)"
```

---

### Task 5: Typed/multi-block SSE formatters + native response envelope (`anthropic_compat.py`)

**Files:**
- Modify: `orchestra/anthropic_compat.py` (append; existing formatters unchanged)
- Test: append to `tests/test_anthropic_compat.py`

**Interfaces:**
- Consumes: existing `_sse`, `new_message_id`.
- Produces (Task 6 consumes): `sse_content_block_start_typed(index, block)`, `sse_text_delta_at(index, text)`, `sse_input_json_delta(index, partial_json)`, `sse_content_block_stop_at(index)`, `sse_message_delta_reason(stop_reason, output_tokens=0)`, `to_native_message_response(model, result: dict) -> dict`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_anthropic_compat.py` (the file already defines `_parse_sse`):

```python
from orchestra.anthropic_compat import (
    sse_content_block_start_typed, sse_text_delta_at, sse_input_json_delta,
    sse_content_block_stop_at, sse_message_delta_reason, to_native_message_response,
)


def test_typed_sse_tool_use_block():
    ev, data = _parse_sse(sse_content_block_start_typed(
        1, {"type": "tool_use", "id": "toolu_x", "name": "bash", "input": {}}))
    assert ev == "content_block_start" and data["index"] == 1
    assert data["content_block"]["type"] == "tool_use" and data["content_block"]["name"] == "bash"

    ev, data = _parse_sse(sse_input_json_delta(1, '{"cmd": "ls"}'))
    assert ev == "content_block_delta"
    assert data["delta"] == {"type": "input_json_delta", "partial_json": '{"cmd": "ls"}'}

    ev, data = _parse_sse(sse_text_delta_at(0, "hi"))
    assert data["index"] == 0 and data["delta"]["type"] == "text_delta"

    ev, data = _parse_sse(sse_content_block_stop_at(1))
    assert ev == "content_block_stop" and data["index"] == 1

    ev, data = _parse_sse(sse_message_delta_reason("tool_use", output_tokens=5))
    assert data["delta"]["stop_reason"] == "tool_use"
    assert data["usage"] == {"output_tokens": 5}


def test_to_native_message_response_envelope():
    result = {"content": [{"type": "text", "text": "ok"},
                          {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"c": 1}}],
              "stop_reason": "tool_use", "usage": {"input_tokens": 7, "output_tokens": 2}}
    r = to_native_message_response("oll-fast", result)
    assert r["type"] == "message" and r["role"] == "assistant" and r["model"] == "oll-fast"
    assert r["content"] == result["content"]
    assert r["stop_reason"] == "tool_use"
    assert r["usage"] == {"input_tokens": 7, "output_tokens": 2}
    assert r["id"].startswith("msg_")
```

- [ ] **Step 2: Run to verify they fail** (ImportError).

- [ ] **Step 3: Implement** — append to `orchestra/anthropic_compat.py`:

```python
# --- typed/multi-block formatters for the native (tool-use) path -------------

def sse_content_block_start_typed(index: int, block: dict) -> str:
    return _sse("content_block_start", {
        "type": "content_block_start", "index": index, "content_block": block,
    })


def sse_text_delta_at(index: int, text: str) -> str:
    return _sse("content_block_delta", {
        "type": "content_block_delta", "index": index,
        "delta": {"type": "text_delta", "text": text},
    })


def sse_input_json_delta(index: int, partial_json: str) -> str:
    return _sse("content_block_delta", {
        "type": "content_block_delta", "index": index,
        "delta": {"type": "input_json_delta", "partial_json": partial_json},
    })


def sse_content_block_stop_at(index: int) -> str:
    return _sse("content_block_stop", {"type": "content_block_stop", "index": index})


def sse_message_delta_reason(stop_reason: str, output_tokens: int = 0) -> str:
    return _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })


def to_native_message_response(model: str, result: dict) -> dict:
    """Anthropic message envelope around a provider's native_messages result."""
    usage = result.get("usage") or {}
    return {
        "id": new_message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": result.get("content") or [],
        "stop_reason": result.get("stop_reason", "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
        },
    }
```

- [ ] **Step 4: Run to verify all pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_anthropic_compat.py -q --no-cov -p no:cacheprovider -o addopts=""`
Expected: 11 passed (9 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add orchestra/anthropic_compat.py tests/test_anthropic_compat.py
git commit -m "feat(anthropic): typed multi-block SSE formatters + native response envelope"
```

---

### Task 6: Gateway native path — routing + buffered SSE (`gateway.py`)

**Files:**
- Modify: `orchestra/gateway.py` (inside `make_app`, next to `_v1_messages`)
- Test: `tests/test_gateway_native.py` (create)

**Interfaces:**
- Consumes: `gateway_dispatch.native_capable/run_native` (Task 4), `anthropic_compat` typed formatters + `to_native_message_response` (Task 5), existing `_authorized`, `_anthropic_error`.
- Produces: `tools`-bearing requests to capable providers served natively (non-stream + stream).

- [ ] **Step 1: Write the failing tests** — create `tests/test_gateway_native.py`:

```python
from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock
from aiohttp.test_utils import TestClient, TestServer

from orchestra import gateway

_TOOLS = [{"name": "bash", "description": "run", "input_schema": {"type": "object"}}]
_NATIVE_RESULT = {
    "content": [{"type": "text", "text": "running"},
                {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"cmd": "ls"}}],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 9, "output_tokens": 4},
}


def _body(stream=False):
    b = {"model": "oll-fast", "max_tokens": 64, "tools": _TOOLS,
         "messages": [{"role": "user", "content": "list files"}]}
    if stream:
        b["stream"] = True
    return b


@pytest.mark.asyncio
async def test_tools_capable_uses_native_path():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.native_capable", return_value=True), \
             patch("orchestra.gateway_dispatch.run_native", new=AsyncMock(return_value=_NATIVE_RESULT)) as mock_native:
            resp = await client.post("/v1/messages", json=_body())
            assert resp.status == 200
            body = await resp.json()
        assert mock_native.call_args.args[0] == "oll-fast"
        assert mock_native.call_args.args[1]["tools"] == _TOOLS
        assert body["stop_reason"] == "tool_use"
        assert body["content"][1]["type"] == "tool_use"


@pytest.mark.asyncio
async def test_tools_non_capable_falls_back_to_text_path():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.native_capable", return_value=False), \
             patch("orchestra.gateway_dispatch.run_target", new=AsyncMock(return_value="plain")) as mock_text:
            resp = await client.post("/v1/messages", json=_body())
            assert resp.status == 200
            body = await resp.json()
        mock_text.assert_called_once()
        assert body["content"][0]["text"] == "plain"


@pytest.mark.asyncio
async def test_no_tools_never_checks_native():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.native_capable") as mock_cap, \
             patch("orchestra.gateway_dispatch.run_target", new=AsyncMock(return_value="ok")):
            b = _body(); del b["tools"]
            resp = await client.post("/v1/messages", json=b)
            assert resp.status == 200
        mock_cap.assert_not_called()


@pytest.mark.asyncio
async def test_native_stream_event_sequence_with_tool_use():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.native_capable", return_value=True), \
             patch("orchestra.gateway_dispatch.run_native", new=AsyncMock(return_value=_NATIVE_RESULT)):
            resp = await client.post("/v1/messages", json=_body(stream=True))
            assert resp.status == 200
            text = await resp.text()
    events = [ln[len("event: "):] for ln in text.splitlines() if ln.startswith("event: ")]
    assert events == [
        "message_start",
        "content_block_start", "content_block_delta", "content_block_stop",   # text block 0
        "content_block_start", "content_block_delta", "content_block_stop",   # tool_use block 1
        "message_delta", "message_stop",
    ]
    assert '"input_json_delta"' in text and '"partial_json"' in text
    assert '"stop_reason": "tool_use"' in text


@pytest.mark.asyncio
async def test_native_run_failure_500_generic():
    async def _boom(model, payload):
        raise RuntimeError("secret /path token=abc")
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.native_capable", return_value=True), \
             patch("orchestra.gateway_dispatch.run_native", _boom):
            resp = await client.post("/v1/messages", json=_body())
            assert resp.status == 500
            raw = await resp.text()
        assert "secret" not in raw and "token=abc" not in raw
```

- [ ] **Step 2: Run to verify they fail** (native path missing → text path answers / wrong shapes).

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_gateway_native.py -q --no-cov -p no:cacheprovider -o addopts=""`

- [ ] **Step 3: Implement** — in `orchestra/gateway.py` inside `make_app`:

(a) In `_v1_messages`, right AFTER the lenient model resolution line
(`model = anthropic_compat.resolve_anthropic_model(...)`) and BEFORE the prompt build, insert:

```python
        if payload.get("tools") and gateway_dispatch.native_capable(model):
            return await _v1_messages_native(request, model, payload)
```

(also add `gateway_dispatch` to that handler's lazy import line: `from orchestra import gateway_dispatch, anthropic_compat`).

(b) Add the two native handlers next to `_v1_messages_stream`:

```python
    async def _v1_messages_native(request: web.Request, model: str, payload: dict):
        from orchestra import gateway_dispatch, anthropic_compat
        native_payload = {k: payload[k] for k in
                          ("system", "messages", "tools", "tool_choice", "max_tokens")
                          if k in payload}
        if payload.get("stream"):
            return await _v1_messages_stream_native(request, model, native_payload)
        try:
            result = await gateway_dispatch.run_native(model, native_payload)
        except Exception as exc:
            logger.warning("anthropic native run failed: %s", exc)
            return _anthropic_error("internal error running the model", "api_error", 500)
        return web.json_response(anthropic_compat.to_native_message_response(model, result))

    async def _v1_messages_stream_native(request: web.Request, model: str, native_payload: dict):
        from orchestra import gateway_dispatch, anthropic_compat
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(request)
        try:
            result = await gateway_dispatch.run_native(model, native_payload)
        except Exception as exc:
            logger.warning("anthropic native stream failed: %s", exc)
            result = {"content": [{"type": "text", "text": "[stream error]"}],
                      "stop_reason": "end_turn", "usage": {}}
        usage = result.get("usage") or {}
        with contextlib.suppress(ConnectionResetError, ClientError):
            await resp.write(anthropic_compat.sse_message_start(
                model, anthropic_compat.new_message_id(), int(usage.get("input_tokens") or 0)).encode())
            for i, block in enumerate(result.get("content") or []):
                if block.get("type") == "tool_use":
                    start = {"type": "tool_use", "id": block.get("id", ""),
                             "name": block.get("name", ""), "input": {}}
                    await resp.write(anthropic_compat.sse_content_block_start_typed(i, start).encode())
                    await resp.write(anthropic_compat.sse_input_json_delta(
                        i, json.dumps(block.get("input", {}), ensure_ascii=False)).encode())
                else:
                    await resp.write(anthropic_compat.sse_content_block_start_typed(
                        i, {"type": "text", "text": ""}).encode())
                    await resp.write(anthropic_compat.sse_text_delta_at(i, block.get("text", "")).encode())
                await resp.write(anthropic_compat.sse_content_block_stop_at(i).encode())
            await resp.write(anthropic_compat.sse_message_delta_reason(
                result.get("stop_reason", "end_turn"), int(usage.get("output_tokens") or 0)).encode())
            await resp.write(anthropic_compat.sse_message_stop().encode())
            await resp.write_eof()
        return resp
```

- [ ] **Step 4: Run the new tests + full regression sweep**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_gateway_native.py /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_anthropic_gateway.py /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_anthropic_compat.py /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_openai_gateway.py /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_gateway.py /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_gateway_dispatch.py /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_ollama_native.py /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_native_messages_capability.py -q --no-cov -p no:cacheprovider -o addopts=""`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add orchestra/gateway.py tests/test_gateway_native.py
git commit -m "feat(gateway): native tool-use path on /v1/messages (buffered synthetic SSE)"
```

---

### Task 7: docs + live smoke (Claude Code fully agentic)

**Files:**
- Modify: `CLAUDE.md`, `docs/architecture.md`
- No new test; live smoke with real Ollama + Claude Code.

- [ ] **Step 1: `CLAUDE.md`** — update the Anthropic-gateway comment block in `## Setup & Commands` (added by the previous sub-project) to reflect tool support; replace its last line:

FROM: `#   (unknown model names fall back to oll-fast; tool use is unsupported — text only)`
TO:
```
#   (unknown model names fall back to oll-fast; tool use WORKS for Ollama models —
#   Claude Code is fully agentic via e.g. ANTHROPIC_MODEL=oll-kimi; orchestra-* modes stay text-only)
```

- [ ] **Step 2: `docs/architecture.md`** — in the Providers row, extend the gateway sentence: `tools`-bearing `/v1/messages` requests to native-capable providers (Ollama via `/api/chat`, `ollama_native.py` translation) return real `tool_use` blocks — Claude Code runs fully agentic; other providers/modes degrade to text.

- [ ] **Step 3: Live smoke (real Ollama + real Claude Code)**

```bash
# server (scratch port, real code):
.venv/bin/orchestra serve --host 127.0.0.1 --port 7099 --token smoke &
sleep 3
# 1) curl: tools request returns a tool_use block (qwen2.5:3b is tool-nominal):
curl -s http://127.0.0.1:7099/v1/messages -H "x-api-key: smoke" -d '{
  "model":"oll-fast","max_tokens":128,
  "tools":[{"name":"write_file","description":"write a file","input_schema":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}],
  "messages":[{"role":"user","content":"use the write_file tool to create hello.txt containing merhaba"}]}' | head -c 600
# expect: "type": "tool_use", "name": "write_file", stop_reason "tool_use"
# 2) Claude Code end-to-end IN A SCRATCH DIR (the file must actually appear):
cd <scratch-dir> && ANTHROPIC_BASE_URL=http://127.0.0.1:7099 ANTHROPIC_API_KEY=smoke \
  ANTHROPIC_MODEL=oll-kimi claude -p "create hello.txt containing exactly the word merhaba" \
  --output-format json | head -c 600
ls hello.txt && cat hello.txt      # ACCEPTANCE: file exists, contains merhaba
# 3) repeat 2) with ANTHROPIC_MODEL=oll-fast (local 3B) — document quality if it fails; acceptance rides on oll-kimi
# kill the server afterward
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/architecture.md
git commit -m "docs(tool-use): document agentic Claude Code via Ollama native messages"
```

---

## Verification (end-to-end)

1. All 8 test files green from a scratch dir; full fast suite shows no new regressions.
2. Live: curl `tools` request → `tool_use` block + `stop_reason:"tool_use"` (stream & non-stream).
3. Live acceptance: `ANTHROPIC_MODEL=oll-kimi claude -p "create hello.txt..."` in a scratch dir → the file EXISTS afterwards (full tool_use → tool_result → completion loop).
4. Degradation: `ANTHROPIC_MODEL=orchestra-dual` (or a CLI alias) with tools → plain-text answer, no crash.
5. Text path untouched: existing opencode/Claude Code text chats keep working.

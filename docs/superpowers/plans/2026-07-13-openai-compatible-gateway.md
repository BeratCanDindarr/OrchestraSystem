# OpenAI-compatible Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /v1/chat/completions` + `GET /v1/models` to the existing agent gateway so any OpenAI-compatible tool (opencode/aider/Continue/…) can drive Orchestra's aliases and orchestration modes.

**Architecture:** A new isolated `orchestra/openai_compat.py` does translation (messages→prompt, model→target, response/SSE formatting, run dispatch); `gateway.make_app` gains two routes that call it, reusing the existing bearer auth. Single aliases use the light `resolve()`+`provider.stream/run` path; `orchestra-<mode>` models use `run_dual_async`/`run_critical_async`/`run_task`.

**Tech Stack:** Python 3.11+, aiohttp (existing), stdlib (json/uuid/time/asyncio). No new deps. No Rust. Reuses `fallback.resolve`, `available_aliases`, the async runners, and the chat REPL's history-render approach.

## Global Constraints

- Pure Python; NO new dependencies; NO Rust.
- Two routes added to `gateway.make_app`: `POST /v1/chat/completions`, `GET /v1/models`. Reuse `_authorized(request, token)` (bearer, hmac.compare_digest) on BOTH; `/health` stays open.
- Model id is an Orchestra alias verbatim (`oll-fast`, `cdx-deep@windows`) OR `orchestra-<mode>` where mode ∈ {dual, critical, auto}.
- messages → one prompt: a leading `system` message becomes a `System: …` prefix; user/assistant turns render `User: …\n\nAssistant: …\n\nUser: …\n\nAssistant:` (same approach as `chat/history.render_history`). Plain-text content only (multimodal is non-goal).
- Single alias: `resolve(model)` → `provider.stream()` (live) / `provider.run()` — the light path, NOT run_ask. Mode: `run_dual_async(prompt)` / `run_critical_async(prompt)` / `run_task(mode="auto", task=prompt)` (wrap the sync `run_task` in `asyncio.to_thread`).
- Reply from a mode = `run.summary` if present else the last/winning agent's `stdout_log`.
- OpenAI response shapes are exact (see tasks). Errors in OpenAI shape `{"error":{"message","type","code"}}`: unknown model → 404 invalid_request_error; invalid JSON / missing messages → 400; auth → 401; run failure → 500.
- `usage` best-effort (zeros when unknown). No silent exception swallowing; functions <50 lines; files <400 typical.
- Run pytest FROM A SCRATCH DIR: `cd /tmp && /abs/.venv/bin/pytest <abs paths> -q --no-cov -p no:cacheprovider`.
- Conventional Commits; NO Co-Authored-By lines.

## File Structure

- `orchestra/openai_compat.py` — CREATE: `messages_to_prompt`, `resolve_target`, `list_models`, `to_response`, `to_sse_chunk`, `sse_done`, `run_target` (async, non-stream), `stream_target` (async generator).
- `orchestra/gateway.py` — MODIFY: add the two routes to `make_app`.
- Tests: `tests/test_openai_compat.py`, `tests/test_openai_gateway.py`.

---

### Task 1: translation — messages_to_prompt + resolve_target

**Files:**
- Create: `orchestra/openai_compat.py`
- Test: `tests/test_openai_compat.py`

**Interfaces:**
- Produces: `messages_to_prompt(messages: list[dict]) -> str`; `resolve_target(model: str) -> tuple[str, str]` returning `("mode", "dual"|"critical"|"auto")` for `orchestra-<mode>` (unknown `orchestra-x` → ValueError) else `("alias", model)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_openai_compat.py
import pytest
from orchestra.openai_compat import messages_to_prompt, resolve_target


def test_messages_to_prompt_system_prefix_and_history():
    msgs = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again?"},
    ]
    out = messages_to_prompt(msgs)
    assert out == (
        "System: be terse\n\n"
        "User: hi\n\n"
        "Assistant: hello\n\n"
        "User: again?\n\n"
        "Assistant:"
    )


def test_messages_to_prompt_no_system():
    out = messages_to_prompt([{"role": "user", "content": "hi"}])
    assert out == "User: hi\n\nAssistant:"


def test_resolve_target_alias():
    assert resolve_target("oll-fast") == ("alias", "oll-fast")
    assert resolve_target("cdx-deep@windows") == ("alias", "cdx-deep@windows")


def test_resolve_target_modes():
    assert resolve_target("orchestra-dual") == ("mode", "dual")
    assert resolve_target("orchestra-critical") == ("mode", "critical")
    assert resolve_target("orchestra-auto") == ("mode", "auto")


def test_resolve_target_unknown_mode_raises():
    with pytest.raises(ValueError):
        resolve_target("orchestra-bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_openai_compat.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`No module named 'orchestra.openai_compat'`).

- [ ] **Step 3: Create the module with the two functions**

```python
# orchestra/openai_compat.py
"""OpenAI-compatible translation for the Orchestra gateway (/v1/chat/completions)."""
from __future__ import annotations

_MODES = {"dual", "critical", "auto"}
_ROLE_LABEL = {"system": "System", "user": "User", "assistant": "Assistant"}


def messages_to_prompt(messages: list[dict]) -> str:
    """Concatenate an OpenAI messages array into one prompt ending in 'Assistant:'."""
    lines = []
    for m in messages:
        label = _ROLE_LABEL.get(m.get("role", "user"), "User")
        lines.append(f"{label}: {m.get('content', '')}")
    return "\n\n".join(lines) + "\n\nAssistant:"


def resolve_target(model: str) -> tuple[str, str]:
    """('mode', <mode>) for orchestra-<mode>, else ('alias', <model>)."""
    if model.startswith("orchestra-"):
        mode = model[len("orchestra-"):]
        if mode not in _MODES:
            raise ValueError(f"unknown orchestration mode {mode!r}")
        return "mode", mode
    return "alias", model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_openai_compat.py -q --no-cov -p no:cacheprovider`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add orchestra/openai_compat.py tests/test_openai_compat.py
git commit -m "feat(openai): messages_to_prompt + resolve_target translation"
```

---

### Task 2: response + SSE formatting

**Files:**
- Modify: `orchestra/openai_compat.py`
- Test: `tests/test_openai_compat.py` (extend)

**Interfaces:**
- Produces: `to_response(model, content, prompt_tokens=0, completion_tokens=0) -> dict` (a `chat.completion` object); `to_sse_chunk(model, delta_content="", finish_reason=None) -> str` (`data: {chat.completion.chunk}\n\n`); `sse_done() -> "data: [DONE]\n\n"`. IDs use a fixed prefix `chatcmpl-` + a caller-independent token — tests pass `_id`/`_created` via optional kwargs to stay deterministic.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_openai_compat.py
import json
from orchestra.openai_compat import to_response, to_sse_chunk, sse_done


def test_to_response_shape():
    r = to_response("oll-fast", "hello", prompt_tokens=3, completion_tokens=1, _id="chatcmpl-x", _created=111)
    assert r["object"] == "chat.completion"
    assert r["id"] == "chatcmpl-x"
    assert r["created"] == 111
    assert r["model"] == "oll-fast"
    assert r["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
    assert r["choices"][0]["finish_reason"] == "stop"
    assert r["usage"] == {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}


def test_to_sse_chunk_delta():
    line = to_sse_chunk("oll-fast", "hel", _id="chatcmpl-x", _created=111)
    assert line.startswith("data: ")
    assert line.endswith("\n\n")
    obj = json.loads(line[len("data: "):].strip())
    assert obj["object"] == "chat.completion.chunk"
    assert obj["choices"][0]["delta"] == {"content": "hel"}
    assert obj["choices"][0]["finish_reason"] is None


def test_to_sse_chunk_final():
    line = to_sse_chunk("oll-fast", "", finish_reason="stop", _id="chatcmpl-x", _created=111)
    obj = json.loads(line[len("data: "):].strip())
    assert obj["choices"][0]["finish_reason"] == "stop"
    assert obj["choices"][0]["delta"] == {}


def test_sse_done():
    assert sse_done() == "data: [DONE]\n\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_openai_compat.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`cannot import name 'to_response'`).

- [ ] **Step 3: Add the formatters**

Append to `orchestra/openai_compat.py` (add `import json`, `import time`, `import uuid` at the top):

```python
def _new_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def to_response(model: str, content: str, prompt_tokens: int = 0,
                completion_tokens: int = 0, _id: str | None = None, _created: int | None = None) -> dict:
    return {
        "id": _id or _new_id(),
        "object": "chat.completion",
        "created": _created if _created is not None else int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def to_sse_chunk(model: str, delta_content: str = "", finish_reason: str | None = None,
                 _id: str | None = None, _created: int | None = None) -> str:
    delta = {"content": delta_content} if delta_content else {}
    obj = {
        "id": _id or _new_id(),
        "object": "chat.completion.chunk",
        "created": _created if _created is not None else int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def sse_done() -> str:
    return "data: [DONE]\n\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_openai_compat.py -q --no-cov -p no:cacheprovider`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add orchestra/openai_compat.py tests/test_openai_compat.py
git commit -m "feat(openai): chat.completion + SSE chunk formatters"
```

---

### Task 3: list_models

**Files:**
- Modify: `orchestra/openai_compat.py`
- Test: `tests/test_openai_compat.py` (extend)

**Interfaces:**
- Consumes: `available_aliases()` (returns lines like `"✓ oll-fast    $0.0/1k"` and mode lines like `"  auto ..."`).
- Produces: `list_models() -> list[dict]` = `[{"id","object":"model","owned_by":"orchestra"}]` for each available (✓) single alias plus `orchestra-dual`/`orchestra-critical`/`orchestra-auto`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_openai_compat.py
from unittest.mock import patch
from orchestra.openai_compat import list_models


def test_list_models_includes_aliases_and_modes():
    fake = ["✓ oll-fast    $0.0/1k", "✗ gmn-pro    $0.006/1k", "  auto     (router)"]
    with patch("orchestra.openai_compat.available_aliases", return_value=fake):
        models = list_models()
    ids = [m["id"] for m in models]
    assert "oll-fast" in ids            # available single alias
    assert "gmn-pro" not in ids         # unavailable, skipped
    assert "orchestra-dual" in ids
    assert "orchestra-critical" in ids
    assert "orchestra-auto" in ids
    assert all(m["object"] == "model" for m in models)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_openai_compat.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`cannot import name 'list_models'`).

- [ ] **Step 3: Add list_models**

Add to `orchestra/openai_compat.py` (add at top: `from orchestra.providers.fallback import available_aliases`):

```python
def list_models() -> list[dict]:
    """Available single aliases (✓ rows) + the orchestra-<mode> models."""
    models: list[dict] = []
    for row in available_aliases():
        if "✓" in row:
            # rows look like "✓ oll-fast    $0.0/1k" — the alias is the 2nd token
            parts = row.split()
            if len(parts) >= 2:
                models.append({"id": parts[1], "object": "model", "owned_by": "orchestra"})
    for mode in ("dual", "critical", "auto"):
        models.append({"id": f"orchestra-{mode}", "object": "model", "owned_by": "orchestra"})
    return models
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_openai_compat.py -q --no-cov -p no:cacheprovider`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add orchestra/openai_compat.py tests/test_openai_compat.py
git commit -m "feat(openai): /v1/models listing (aliases + orchestration modes)"
```

---

### Task 4: run dispatch — run_target (non-stream) + stream_target

**Files:**
- Modify: `orchestra/openai_compat.py`
- Test: `tests/test_openai_compat.py` (extend)

**Interfaces:**
- Consumes: `resolve_target` (Task 1), `fallback.resolve`, `run_dual_async`/`run_critical_async` (`orchestra.engine.runners.common`), `run_task` (`orchestra.service`), `messages_to_prompt`.
- Produces:
  - `async run_target(model, prompt) -> str` — full reply. alias → `resolve(model)` then `provider.run(prompt, effort)` returning stdout; mode → the async runner (auto via `asyncio.to_thread(run_task, mode="auto", task=prompt)` → its dict's `run` manifest) → reply text (`summary` or first agent stdout).
  - `stream_target(model, prompt) -> asyncio generator[str]` — for a streaming-capable single alias, yields provider.stream chunks (run in a thread, bridged via `asyncio.Queue`); otherwise yields the full `run_target` reply once.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_openai_compat.py
import asyncio
from unittest.mock import patch, MagicMock
from orchestra import openai_compat as oc


class _RunProvider:
    def stream(self, prompt, effort, timeout=180, **k):
        raise NotImplementedError   # like base/CLI/remote providers
    def run(self, prompt, effort, timeout=180, **k):
        assert "User: hi" in prompt
        return "one-shot", 0


def test_run_target_alias_uses_provider_run():
    with patch("orchestra.openai_compat.resolve", return_value=(_RunProvider(), "fast")):
        out = asyncio.run(oc.run_target("oll-fast", "User: hi\n\nAssistant:"))
    assert out == "one-shot"


def test_run_target_mode_dual_uses_summary():
    fake_run = MagicMock()
    fake_run.summary = "the synthesis"
    async def _fake_dual(prompt, **k):
        return fake_run
    with patch("orchestra.openai_compat.run_dual_async", _fake_dual):
        out = asyncio.run(oc.run_target("orchestra-dual", "User: go\n\nAssistant:"))
    assert out == "the synthesis"


def test_stream_target_nonstreaming_yields_once():
    async def _collect():
        chunks = []
        async for c in oc.stream_target("oll-fast", "User: hi\n\nAssistant:"):
            chunks.append(c)
        return chunks
    with patch("orchestra.openai_compat.resolve", return_value=(_RunProvider(), "fast")):
        chunks = asyncio.run(_collect())
    assert "".join(chunks) == "one-shot"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_openai_compat.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`cannot import name 'run_target'` / `stream_target`).

- [ ] **Step 3: Add the dispatch**

Add to `orchestra/openai_compat.py` (add at top: `import asyncio`, `from orchestra.providers.fallback import resolve`, `from orchestra.engine.runners.common import run_dual_async, run_critical_async`):

```python
def _reply_from_run(run) -> str:
    summary = getattr(run, "summary", None)
    if summary:
        return summary
    agents = getattr(run, "agents", None) or []
    return agents[-1].stdout_log if agents else ""


async def run_target(model: str, prompt: str) -> str:
    """Full (non-streaming) reply for a model id."""
    kind, value = resolve_target(model)
    if kind == "mode":
        if value == "dual":
            return _reply_from_run(await run_dual_async(prompt))
        if value == "critical":
            return _reply_from_run(await run_critical_async(prompt))
        # auto — sync run_task wrapped in a thread; returns {"run": manifest_dict}
        from orchestra.service import run_task
        result = await asyncio.to_thread(run_task, mode="auto", task=prompt)
        run_dict = result.get("run", {}) if isinstance(result, dict) else {}
        return run_dict.get("summary") or _reply_from_manifest(run_dict)
    provider, effort = resolve(model)
    stdout, _ = await asyncio.to_thread(provider.run, prompt, effort)
    return stdout


def _reply_from_manifest(manifest: dict) -> str:
    agents = manifest.get("agents") or []
    return agents[-1].get("stdout_log", "") if agents else ""


async def stream_target(model: str, prompt: str):
    """Yield reply chunks. Live tokens for a streaming single alias; else one chunk."""
    kind, _ = resolve_target(model)
    if kind == "alias":
        provider, effort = resolve(model)
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        _SENTINEL = object()

        def _produce():
            try:
                for chunk in provider.stream(prompt, effort):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except NotImplementedError:
                stdout, _ = provider.run(prompt, effort)
                loop.call_soon_threadsafe(queue.put_nowait, stdout)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, f"[ERROR] {exc}")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        await asyncio.to_thread(_start_bg, _produce)
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                return
            yield item
    else:
        # orchestration modes don't token-stream; emit the whole reply once
        yield await run_target(model, prompt)


def _start_bg(fn) -> None:
    import threading
    threading.Thread(target=fn, daemon=True).start()
```

Note: `_start_bg` launches the producer thread; `asyncio.to_thread(_start_bg, ...)` just schedules it without blocking the loop. The streaming test's provider raises `NotImplementedError` from `stream()` (like base/CLI/remote providers), so the producer falls back to `run()` and enqueues the full reply once — matching the assertion. A real Ollama/Claude-SDK provider yields live chunks instead.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_openai_compat.py -q --no-cov -p no:cacheprovider`
Expected: PASS (13 tests). If the streaming test hangs, the sentinel/queue wiring is wrong — verify `_SENTINEL` is always enqueued in `finally`.

- [ ] **Step 5: Commit**

```bash
git add orchestra/openai_compat.py tests/test_openai_compat.py
git commit -m "feat(openai): run_target + stream_target dispatch (alias light-path, modes async)"
```

---

### Task 5: gateway routes — /v1/chat/completions + /v1/models

**Files:**
- Modify: `orchestra/gateway.py` (add routes in `make_app`)
- Test: `tests/test_openai_gateway.py`

**Interfaces:**
- Consumes: `openai_compat.{messages_to_prompt, resolve_target, run_target, stream_target, to_response, to_sse_chunk, sse_done, list_models}`; the existing `_authorized`.
- Produces: `POST /v1/chat/completions` (stream + non-stream), `GET /v1/models`. Uses `aiohttp.test_utils.TestClient(TestServer(...))` in tests (pytest-aiohttp is NOT installed — mirror tests/test_gateway.py).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_openai_gateway.py
import json
import pytest
from unittest.mock import patch
from aiohttp.test_utils import TestServer, TestClient
from orchestra import gateway


@pytest.mark.asyncio
async def test_models_lists(monkeypatch):
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.openai_compat.list_models", return_value=[{"id": "oll-fast", "object": "model"}]):
            resp = await client.get("/v1/models")
        assert resp.status == 200
        body = await resp.json()
        assert body["object"] == "list"
        assert body["data"][0]["id"] == "oll-fast"


@pytest.mark.asyncio
async def test_chat_completions_non_stream():
    async def _fake_run_target(model, prompt):
        assert model == "oll-fast"
        return "hello world"
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.openai_compat.run_target", _fake_run_target):
            resp = await client.post("/v1/chat/completions", json={
                "model": "oll-fast",
                "messages": [{"role": "user", "content": "hi"}],
            })
        assert resp.status == 200
        body = await resp.json()
        assert body["choices"][0]["message"]["content"] == "hello world"


@pytest.mark.asyncio
async def test_chat_completions_stream_sse():
    async def _fake_stream(model, prompt):
        for c in ["hel", "lo"]:
            yield c
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.openai_compat.stream_target", _fake_stream):
            resp = await client.post("/v1/chat/completions", json={
                "model": "oll-fast", "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            })
            assert resp.status == 200
            text = await resp.text()
    assert "data: " in text
    assert "[DONE]" in text
    assert "hel" in text and "lo" in text


@pytest.mark.asyncio
async def test_chat_completions_unknown_model_404():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "orchestra-bogus",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert resp.status == 404
    body = await resp.json()
    assert body["error"]["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_chat_completions_missing_messages_400():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        resp = await client.post("/v1/chat/completions", json={"model": "oll-fast"})
    assert resp.status == 400


@pytest.mark.asyncio
async def test_v1_requires_token_when_set():
    async with TestClient(TestServer(gateway.make_app(token="secret"))) as client:
        resp = await client.get("/v1/models")
    assert resp.status == 401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_openai_gateway.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (404 for /v1/models — routes not added).

- [ ] **Step 3: Add the routes to `make_app`**

In `orchestra/gateway.py`, inside `make_app`, before the `app.router.add_post("/run", _run)` block, add these handlers and register them. Add `from orchestra import openai_compat` at the top of the handler bodies (lazy import to avoid load-time cost):

```python
    async def _v1_models(request: web.Request) -> web.Response:
        if not _authorized(request, token):
            return web.json_response({"error": {"message": "Unauthorized", "type": "invalid_request_error", "code": "401"}}, status=401)
        from orchestra import openai_compat
        return web.json_response({"object": "list", "data": openai_compat.list_models()})

    async def _v1_chat(request: web.Request):
        from orchestra import openai_compat
        if not _authorized(request, token):
            return web.json_response({"error": {"message": "Unauthorized", "type": "invalid_request_error", "code": "401"}}, status=401)
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return web.json_response({"error": {"message": "Invalid JSON", "type": "invalid_request_error", "code": "400"}}, status=400)
        if not isinstance(payload, dict) or not payload.get("messages"):
            return web.json_response({"error": {"message": "messages is required", "type": "invalid_request_error", "code": "400"}}, status=400)
        model = payload.get("model", "orchestra-auto")
        try:
            openai_compat.resolve_target(model)  # validates orchestra-<mode>
        except ValueError as exc:
            return web.json_response({"error": {"message": str(exc), "type": "invalid_request_error", "code": "404"}}, status=404)
        prompt = openai_compat.messages_to_prompt(payload["messages"])
        if payload.get("stream"):
            return await _v1_chat_stream(request, model, prompt)
        try:
            content = await openai_compat.run_target(model, prompt)
        except Exception as exc:
            logger.warning("openai chat run failed: %s", exc)
            return web.json_response({"error": {"message": str(exc), "type": "server_error", "code": "500"}}, status=500)
        return web.json_response(openai_compat.to_response(model, content))

    async def _v1_chat_stream(request: web.Request, model: str, prompt: str):
        from orchestra import openai_compat
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(request)
        try:
            async for chunk in openai_compat.stream_target(model, prompt):
                await resp.write(openai_compat.to_sse_chunk(model, chunk).encode())
            await resp.write(openai_compat.to_sse_chunk(model, "", finish_reason="stop").encode())
        except Exception as exc:
            await resp.write(openai_compat.to_sse_chunk(model, f"[ERROR] {exc}", finish_reason="stop").encode())
        await resp.write(openai_compat.sse_done().encode())
        await resp.write_eof()
        return resp

    app.router.add_get("/v1/models", _v1_models)
    app.router.add_post("/v1/chat/completions", _v1_chat)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_openai_gateway.py /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_gateway.py -q --no-cov -p no:cacheprovider`
Expected: PASS (6 new + existing gateway tests green — the /run and auth paths are unchanged).

- [ ] **Step 5: Commit**

```bash
git add orchestra/gateway.py tests/test_openai_gateway.py
git commit -m "feat(gateway): OpenAI-compatible /v1/chat/completions + /v1/models routes"
```

---

### Task 6: docs + live smoke

**Files:**
- Modify: `CLAUDE.md`, `docs/architecture.md`.
- No new test; a local smoke with real Ollama.

- [ ] **Step 1: Add to `CLAUDE.md`**

In `## Setup & Commands`:

```
orchestra serve --token <secret>         # also serves an OpenAI API at http://<host>:7070/v1
# point opencode/aider/Continue at base_url=http://<host>:7070/v1, api_key=<secret>;
# models: any alias (oll-fast, cdx-deep@windows) or orchestra-dual / orchestra-critical / orchestra-auto
```

- [ ] **Step 2: Architecture note**

In `docs/architecture.md`, extend the gateway note: `gateway.py` also serves an OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`) via `openai_compat.py` — single aliases + `orchestra-<mode>` orchestration as models.

- [ ] **Step 3: Local smoke (real Ollama + curl)**

```
.venv/bin/python -m orchestra serve --host 127.0.0.1 --port 7099 --token smoke &
sleep 3
curl -s http://127.0.0.1:7099/v1/models -H "Authorization: Bearer smoke" | head -c 300
curl -s http://127.0.0.1:7099/v1/chat/completions -H "Authorization: Bearer smoke" \
  -d '{"model":"oll-fast","messages":[{"role":"user","content":"say hi in 2 words"}]}' | head -c 400
# kill the server afterward
```
Expected: `/v1/models` lists aliases + orchestra-* modes; the chat call returns a `chat.completion` JSON with an assistant message. No traceback.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/architecture.md
git commit -m "docs(openai): document the OpenAI-compatible gateway and opencode setup"
```

---

## Verification (end-to-end)

1. `orchestra serve --host 0.0.0.0 --token X` then point opencode at `base_url=http://<host>:7070/v1`, `api_key=X`.
2. opencode's model list shows the ✓ aliases + `orchestra-dual`/`orchestra-critical`/`orchestra-auto`.
3. Selecting `oll-fast` streams a live reply; selecting `orchestra-dual` returns a synthesized answer (one chunk).
4. `curl /v1/chat/completions` non-stream and `stream:true` both work; unknown model → 404, missing messages → 400, wrong token → 401.
5. Existing `/run`, `/aliases`, `/health` and `ask`/`chat` are unaffected; full fast suite from a scratch dir shows no new regressions.

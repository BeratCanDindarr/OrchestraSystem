# Anthropic-compatible gateway (`/v1/messages`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Anthropic-compatible `POST /v1/messages` (+ `count_tokens`) surface to the Orchestra gateway so Claude Code can drive Orchestra's aliases + orchestration modes.

**Architecture:** Extract the provider-agnostic dispatch (`resolve_target`/`run_target`/`stream_target`) out of `openai_compat.py` into a neutral `gateway_dispatch.py`, add an isolated `anthropic_compat.py` for Anthropic request/response/SSE translation, and add two routes to the existing aiohttp `gateway.py` reusing its bearer/x-api-key auth.

**Tech Stack:** Python 3.11, aiohttp, stdlib only. Tests use `aiohttp.test_utils.TestClient/TestServer` (no pytest-aiohttp) with `@pytest.mark.asyncio`.

## Global Constraints

- Pure Python, no new dependencies, no Rust.
- Dispatch is SHARED via `gateway_dispatch.py`; neither compat module imports the other.
- Single-alias dispatch uses the light path (`resolve()`+`provider.stream()/run()`); modes use `run_dual_async`/`run_critical_async`/`run_task(auto)`. (All already implemented in the dispatch code being moved — do not rewrite it.)
- **Lenient model handling for Anthropic:** unknown model (not a known alias, not a valid `orchestra-<mode>`) → `_DEFAULT_ALIAS = "oll-fast"`; NEVER 404. (The OpenAI endpoint's strict 404 stays as-is.)
- **v1 is text-only:** `tools`, `max_tokens`, `metadata` accepted and IGNORED; only `text` content blocks extracted; image/tool_use/tool_result blocks ignored.
- Auth accepts `x-api-key: <token>` OR `Authorization: Bearer <token>`, byte-compared (`hmac.compare_digest` on UTF-8 bytes). Same token as `/run` and `/v1/*`.
- Anthropic error shape: `{"type":"error","error":{"type":<class>,"message":<msg>}}`. 400 `invalid_request_error`, 401 `authentication_error`, 500 `api_error` (message GENERIC, full exception logged server-side only — no PII leak). No 404 model path.
- Streaming emits the full Anthropic event sequence and ALWAYS terminates (`content_block_stop`+`message_delta`+`message_stop`) even on error; client disconnect suppresses writes and `aclose()`s the generator.
- Functions <50 lines; files <800 lines. No silent `except: pass`.
- Conventional Commits, NO Co-Authored-By trailer.
- Run pytest from a scratch dir: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest <abs paths> -q --no-cov -p no:cacheprovider -o addopts=""`. Before each commit, if `tests/baselines/performance_baseline.json` is modified, `git checkout --` it.

---

### Task 1: Extract dispatch into `gateway_dispatch.py` (behavior-preserving refactor)

**Files:**
- Create: `orchestra/gateway_dispatch.py`
- Modify: `orchestra/openai_compat.py` (remove moved code + now-unused imports)
- Modify: `orchestra/gateway.py` (call `gateway_dispatch.*` for dispatch)
- Create: `tests/test_gateway_dispatch.py` (move the dispatch tests here)
- Modify: `tests/test_openai_compat.py` (drop moved tests + update imports)
- Modify: `tests/test_openai_gateway.py` (update patch targets)

**Interfaces:**
- Produces (for Tasks 2-5): `gateway_dispatch.resolve_target(model)->tuple[str,str]`, `gateway_dispatch.run_target(model,prompt)->str` (async), `gateway_dispatch.stream_target(model,prompt)` (async generator), `gateway_dispatch._STREAM_ERROR_TEXT`, module `logger`, and the `resolve`/`known_aliases`/`run_dual_async`/`run_critical_async` imports it owns.

- [ ] **Step 1: Create `orchestra/gateway_dispatch.py`** with the dispatch code moved verbatim from `openai_compat.py` (current lines 32-46 `resolve_target`, 116-206 `_reply_from_run`/`_reply_from_manifest`/`run_target`/`_start_bg`/`stream_target`) plus the imports/constants they need:

```python
"""Provider-agnostic dispatch shared by the gateway's OpenAI and Anthropic compat layers:
resolve a model id to a provider/mode and run it — whole (run_target) or streaming (stream_target)."""
from __future__ import annotations

import asyncio
import logging
import threading

from orchestra.providers.fallback import resolve, known_aliases
from orchestra.engine.runners.common import run_dual_async, run_critical_async

logger = logging.getLogger(__name__)
_STREAM_ERROR_TEXT = "[stream error]"   # generic; never leak provider exception text

_MODES = {"dual", "critical", "auto"}


def resolve_target(model: str) -> tuple[str, str]:
    """('mode', <mode>) for orchestra-<mode>, else ('alias', <model>).

    Raises ValueError for an unknown orchestration mode OR an unknown alias
    (base name not in the alias table) — the OpenAI gateway maps this to a 404.
    """
    if model.startswith("orchestra-"):
        mode = model[len("orchestra-"):]
        if mode not in _MODES:
            raise ValueError(f"unknown orchestration mode {mode!r}")
        return "mode", mode
    base = model.split("@", 1)[0]
    if base not in known_aliases():
        raise ValueError(f"unknown model {model!r}")
    return "alias", model


def _reply_from_run(run) -> str:
    """Extract the text reply from an OrchestraRun (or dict) result."""
    summary = getattr(run, "summary", None)
    if summary:
        return summary
    agents = getattr(run, "agents", None) or []
    return agents[-1].stdout_log if agents else ""


def _reply_from_manifest(manifest: dict) -> str:
    """Extract reply text from a run_task manifest dict."""
    agents = manifest.get("agents") or []
    return agents[-1].get("stdout_log", "") if agents else ""


async def run_target(model: str, prompt: str) -> str:
    """Full (non-streaming) reply for a model id."""
    kind, value = resolve_target(model)
    if kind == "mode":
        if value == "dual":
            return _reply_from_run(await run_dual_async(prompt))
        if value == "critical":
            return _reply_from_run(await run_critical_async(prompt))
        from orchestra.service import run_task
        result = await asyncio.to_thread(run_task, mode="auto", task=prompt)
        run_dict = result.get("run", {}) if isinstance(result, dict) else {}
        return run_dict.get("summary") or _reply_from_manifest(run_dict)
    provider, effort = resolve(model)
    stdout, _ = await asyncio.to_thread(provider.run, prompt, effort)
    return stdout


def _start_bg(fn) -> None:
    """Launch fn in a daemon thread."""
    threading.Thread(target=fn, daemon=True).start()


async def stream_target(model: str, prompt: str):
    """Yield reply chunks asynchronously (see openai_compat history for the design).

    Single alias with streaming provider → live tokens via a thread-bridged Queue.
    NotImplementedError (base/CLI/remote) → falls back to run(). Modes → whole reply once.
    The consumer's finally sets a stop Event so a disconnected client abandons the
    background provider stream instead of leaking the thread + a live model call.
    """
    kind, _ = resolve_target(model)
    if kind != "alias":
        yield await run_target(model, prompt)
        return

    provider, effort = resolve(model)
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stop = threading.Event()
    sentinel = object()

    def _produce() -> None:
        try:
            for chunk in provider.stream(prompt, effort):
                if stop.is_set():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except NotImplementedError:
            if not stop.is_set():
                stdout, _ = provider.run(prompt, effort)
                loop.call_soon_threadsafe(queue.put_nowait, stdout)
        except Exception as exc:
            logger.warning("gateway stream provider failed: %s", exc)
            if not stop.is_set():
                loop.call_soon_threadsafe(queue.put_nowait, _STREAM_ERROR_TEXT)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    _start_bg(_produce)
    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                return
            yield item
    finally:
        stop.set()
```

- [ ] **Step 2: Trim `orchestra/openai_compat.py`** — delete `resolve_target` (old lines 32-46), the whole `# Run dispatch` section (old lines 112-206), and now-unused imports/constants. The file's imports become exactly:

```python
"""OpenAI-compatible translation for the Orchestra gateway (/v1/chat/completions)."""
from __future__ import annotations

import json
import time
import uuid

from orchestra.providers.fallback import available_alias_ids
from orchestra.chat.history import render_history
```

Keep ONLY: `messages_to_prompt`, `_new_id`, `to_response`, `to_sse_chunk`, `sse_done`, `list_models` (unchanged bodies). Remove `_MODES`, `logger`, `_STREAM_ERROR_TEXT`, and the `asyncio`/`logging`/`threading`/`resolve`/`known_aliases`/`run_dual_async`/`run_critical_async` imports (now in gateway_dispatch).

- [ ] **Step 3: Update `orchestra/gateway.py`** so the dispatch calls target `gateway_dispatch`. In `_v1_chat` and `_v1_chat_stream`, add `from orchestra import gateway_dispatch` next to the existing `from orchestra import openai_compat`, and change these three call sites (the ONLY dispatch references):
  - `openai_compat.resolve_target(model)` → `gateway_dispatch.resolve_target(model)`
  - `openai_compat.run_target(model, prompt)` → `gateway_dispatch.run_target(model, prompt)`
  - `openai_compat.stream_target(model, prompt)` → `gateway_dispatch.stream_target(model, prompt)`
  Leave `openai_compat.messages_to_prompt/to_response/to_sse_chunk/sse_done/list_models` untouched.

- [ ] **Step 4: Move dispatch tests** to a new `tests/test_gateway_dispatch.py`. Move these functions VERBATIM from `tests/test_openai_compat.py`: `test_resolve_target_alias`, `test_resolve_target_modes`, `test_resolve_target_unknown_mode_raises`, `test_resolve_target_unknown_alias_raises`, `test_resolve_target_known_alias_ok`, `test_resolve_target_backend_ref_ok`, `_RunProvider`, `test_run_target_alias_uses_provider_run`, `test_run_target_mode_dual_uses_summary`, `test_stream_target_nonstreaming_yields_once`, `_fake_alias_provider`, `test_stream_target_aclose_stops_producer`, `_unbounded_counting_provider`, `test_stream_target_aclose_stops_producer_early`. In the new file, apply exactly two transforms: (a) the import becomes `from orchestra.gateway_dispatch import resolve_target, run_target, stream_target` plus `import orchestra.gateway_dispatch as gd` (and keep `import asyncio, json, threading` + `from unittest.mock import patch, MagicMock` + `import pytest` as needed); (b) every `patch("orchestra.openai_compat.resolve", ...)` becomes `patch("orchestra.gateway_dispatch.resolve", ...)`, and every `oc.` referencing a moved name becomes `gd.` (the aclose tests do `monkeypatch.setattr(oc, "resolve", ...)` → `monkeypatch.setattr(gd, "resolve", ...)`; and `oc.stream_target` → `gd.stream_target`).

- [ ] **Step 5: Trim `tests/test_openai_compat.py`** — remove the functions moved in Step 4, and change the import block to only the OpenAI-format names:

```python
import json
from orchestra.openai_compat import (
    messages_to_prompt, to_response, to_sse_chunk, sse_done, list_models,
)
import orchestra.openai_compat as oc
```

(Remove the now-unused `asyncio`, `threading`, `pytest`, `patch`, `MagicMock`, `resolve_target`, `run_target`, `stream_target` imports IF no remaining test in the file uses them — verify with a grep; `list_models` test may still patch `available_alias_ids`, which is fine.)

- [ ] **Step 6: Update `tests/test_openai_gateway.py` patch targets** — the gateway now calls `gateway_dispatch`, so patches must target that module. Change:
  - `patch("orchestra.openai_compat.run_target", ...)` → `patch("orchestra.gateway_dispatch.run_target", ...)` (lines ~26, ~83, ~99)
  - `patch("orchestra.openai_compat.stream_target", ...)` → `patch("orchestra.gateway_dispatch.stream_target", ...)` (line ~42)
  Leave patches of `list_models`/`to_response`/etc. (if any) on `openai_compat`.

- [ ] **Step 7: Run all affected suites — everything must still pass (behavior preserved)**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_gateway_dispatch.py /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_openai_compat.py /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_openai_gateway.py /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_gateway.py -q --no-cov -p no:cacheprovider -o addopts=""`
Expected: all pass (same total count as before the move; no import errors).

- [ ] **Step 8: Commit**

```bash
git add orchestra/gateway_dispatch.py orchestra/openai_compat.py orchestra/gateway.py tests/test_gateway_dispatch.py tests/test_openai_compat.py tests/test_openai_gateway.py
git commit -m "refactor(gateway): extract provider-agnostic dispatch into gateway_dispatch"
```

---

### Task 2: `anthropic_compat` request translation (pure functions)

**Files:**
- Create: `orchestra/anthropic_compat.py`
- Test: `tests/test_anthropic_compat.py`

**Interfaces:**
- Consumes: `orchestra.chat.history.render_history`, `orchestra.providers.fallback.known_aliases`.
- Produces (for Task 5): `request_to_prompt(system, messages)->str`, `resolve_anthropic_model(model)->str`, `approx_tokens(text)->int`, constant `_DEFAULT_ALIAS="oll-fast"`.

- [ ] **Step 1: Write the failing tests** in `tests/test_anthropic_compat.py`:

```python
from orchestra.anthropic_compat import request_to_prompt, resolve_anthropic_model, approx_tokens


def test_request_to_prompt_system_str_and_history():
    system = "be terse"
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
    ]
    assert request_to_prompt(system, messages) == (
        "System: be terse\n\nUser: hi\n\nAssistant: hello\n\nUser: bye\n\nAssistant:"
    )


def test_request_to_prompt_system_blocks_and_content_blocks():
    system = [{"type": "text", "text": "sys one"}, {"type": "text", "text": "sys two"}]
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "look "},
        {"type": "image", "source": {}},          # ignored
        {"type": "text", "text": "here"},
    ]}]
    assert request_to_prompt(system, messages) == (
        "System: sys one\n\nsys two\n\nUser: look here\n\nAssistant:"
    )


def test_request_to_prompt_no_system():
    assert request_to_prompt(None, [{"role": "user", "content": "yo"}]) == "User: yo\n\nAssistant:"


def test_resolve_anthropic_model_known_alias_and_mode():
    assert resolve_anthropic_model("cdx-deep") == "cdx-deep"
    assert resolve_anthropic_model("cdx-deep@windows") == "cdx-deep@windows"
    assert resolve_anthropic_model("orchestra-dual") == "orchestra-dual"


def test_resolve_anthropic_model_unknown_falls_back_to_default():
    assert resolve_anthropic_model("claude-3-5-haiku") == "oll-fast"
    assert resolve_anthropic_model("orchestra-bogus") == "oll-fast"


def test_approx_tokens():
    assert approx_tokens("") == 0
    assert approx_tokens("abcdefgh") == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_anthropic_compat.py -q --no-cov -p no:cacheprovider -o addopts=""`
Expected: FAIL (ModuleNotFoundError: orchestra.anthropic_compat).

- [ ] **Step 3: Create `orchestra/anthropic_compat.py`** with the request-translation functions:

```python
"""Anthropic-compatible translation for the Orchestra gateway (/v1/messages).

Request parsing + response/SSE formatting only; the provider dispatch is shared via
gateway_dispatch. Model handling is LENIENT (unknown -> _DEFAULT_ALIAS) so Claude Code's
built-in background model names route to a real alias instead of 404ing."""
from __future__ import annotations

import json
import uuid

from orchestra.chat.history import render_history
from orchestra.providers.fallback import known_aliases

_DEFAULT_ALIAS = "oll-fast"
_MODES = {"dual", "critical", "auto"}


def _text_of(content) -> str:
    """A message's content is a str or a list of blocks; return the concatenated text
    of the str and any {'type':'text'} blocks (image/tool blocks ignored)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _system_text(system) -> str:
    """Anthropic top-level `system` is a str or a list of text blocks."""
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


def request_to_prompt(system, messages) -> str:
    """Anthropic system + messages -> one concatenated prompt ending in 'Assistant:'."""
    rest = [{"role": m.get("role", "user"), "content": _text_of(m.get("content", ""))}
            for m in messages]
    body = render_history(rest)
    sys_txt = _system_text(system)
    if not sys_txt:
        return body
    return f"System: {sys_txt}\n\n{body}" if body else f"System: {sys_txt}\n\nAssistant:"


def resolve_anthropic_model(model: str) -> str:
    """Return a model string gateway_dispatch accepts: `model` itself when it is a valid
    orchestra-<mode> or a known alias (base before any @), else _DEFAULT_ALIAS. Never raises."""
    if model.startswith("orchestra-") and model[len("orchestra-"):] in _MODES:
        return model
    if model.split("@", 1)[0] in known_aliases():
        return model
    return _DEFAULT_ALIAS


def approx_tokens(text: str) -> int:
    """Best-effort token count (chars / 4). Accuracy is a non-goal."""
    return max(0, len(text) // 4)
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_anthropic_compat.py -q --no-cov -p no:cacheprovider -o addopts=""`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add orchestra/anthropic_compat.py tests/test_anthropic_compat.py
git commit -m "feat(anthropic): request_to_prompt + lenient model resolution + approx_tokens"
```

---

### Task 3: `anthropic_compat` response + SSE event formatters

**Files:**
- Modify: `orchestra/anthropic_compat.py`
- Modify: `tests/test_anthropic_compat.py`

**Interfaces:**
- Produces (for Task 5): `new_message_id()->str`, `to_message_response(model, text, input_tokens=0, output_tokens=0, _id=None)->dict`, and SSE formatters `sse_message_start(model, msg_id, input_tokens=0)`, `sse_content_block_start()`, `sse_text_delta(text)`, `sse_content_block_stop()`, `sse_message_delta(output_tokens=0)`, `sse_message_stop()` — each returns an `event: <type>\ndata: <json>\n\n` string.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_anthropic_compat.py`):

```python
import json as _json
from orchestra.anthropic_compat import (
    to_message_response, new_message_id,
    sse_message_start, sse_content_block_start, sse_text_delta,
    sse_content_block_stop, sse_message_delta, sse_message_stop,
)


def _parse_sse(line):
    assert line.endswith("\n\n")
    event_line, data_line = line.rstrip("\n").split("\n")
    assert event_line.startswith("event: ")
    assert data_line.startswith("data: ")
    return event_line[len("event: "):], _json.loads(data_line[len("data: "):])


def test_to_message_response_shape():
    r = to_message_response("oll-fast", "hi there", input_tokens=3, output_tokens=2, _id="msg_x")
    assert r["id"] == "msg_x" and r["type"] == "message" and r["role"] == "assistant"
    assert r["model"] == "oll-fast"
    assert r["content"] == [{"type": "text", "text": "hi there"}]
    assert r["stop_reason"] == "end_turn" and r["stop_sequence"] is None
    assert r["usage"] == {"input_tokens": 3, "output_tokens": 2}


def test_new_message_id_prefix():
    assert new_message_id().startswith("msg_")


def test_sse_event_sequence_shapes():
    ev, data = _parse_sse(sse_message_start("oll-fast", "msg_1", input_tokens=5))
    assert ev == "message_start"
    assert data["message"]["id"] == "msg_1" and data["message"]["role"] == "assistant"
    assert data["message"]["usage"] == {"input_tokens": 5, "output_tokens": 0}

    ev, data = _parse_sse(sse_content_block_start())
    assert ev == "content_block_start" and data["content_block"] == {"type": "text", "text": ""}

    ev, data = _parse_sse(sse_text_delta("hello"))
    assert ev == "content_block_delta"
    assert data["delta"] == {"type": "text_delta", "text": "hello"}

    ev, data = _parse_sse(sse_content_block_stop())
    assert ev == "content_block_stop" and data["index"] == 0

    ev, data = _parse_sse(sse_message_delta(output_tokens=7))
    assert ev == "message_delta"
    assert data["delta"]["stop_reason"] == "end_turn"
    assert data["usage"] == {"output_tokens": 7}

    ev, data = _parse_sse(sse_message_stop())
    assert ev == "message_stop"
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_anthropic_compat.py -q --no-cov -p no:cacheprovider -o addopts=""`
Expected: FAIL (ImportError: cannot import name 'to_message_response').

- [ ] **Step 3: Add the formatters to `orchestra/anthropic_compat.py`** (append after `approx_tokens`):

```python
def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def to_message_response(
    model: str, text: str, input_tokens: int = 0, output_tokens: int = 0, _id: str | None = None
) -> dict:
    """Build an Anthropic `message` response envelope."""
    return {
        "id": _id or new_message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_message_start(model: str, msg_id: str, input_tokens: int = 0) -> str:
    return _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    })


def sse_content_block_start() -> str:
    return _sse("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    })


def sse_text_delta(text: str) -> str:
    return _sse("content_block_delta", {
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "text_delta", "text": text},
    })


def sse_content_block_stop() -> str:
    return _sse("content_block_stop", {"type": "content_block_stop", "index": 0})


def sse_message_delta(output_tokens: int = 0) -> str:
    return _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })


def sse_message_stop() -> str:
    return _sse("message_stop", {"type": "message_stop"})
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_anthropic_compat.py -q --no-cov -p no:cacheprovider -o addopts=""`
Expected: PASS (9 tests total).

- [ ] **Step 5: Commit**

```bash
git add orchestra/anthropic_compat.py tests/test_anthropic_compat.py
git commit -m "feat(anthropic): message response envelope + SSE event formatters"
```

---

### Task 4: Extend gateway auth to accept `x-api-key`

**Files:**
- Modify: `orchestra/gateway.py` (`_authorized`)
- Modify: `tests/test_gateway.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_authorized(request, token)` now also authorizes an `x-api-key: <token>` header (byte-compared), in addition to the existing `Authorization: Bearer <token>`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_gateway.py`):

```python
def test_authorized_accepts_x_api_key():
    req = MagicMock()
    req.headers = {"x-api-key": "sekret"}
    assert gateway._authorized(req, "sekret") is True
    req.headers = {"x-api-key": "wrong"}
    assert gateway._authorized(req, "sekret") is False


def test_authorized_still_accepts_bearer():
    req = MagicMock()
    req.headers = {"Authorization": "Bearer sekret"}
    assert gateway._authorized(req, "sekret") is True
```

- [ ] **Step 2: Run to verify the x-api-key test fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_gateway.py -q --no-cov -p no:cacheprovider -o addopts="" -k authorized`
Expected: `test_authorized_accepts_x_api_key` FAILS (x-api-key not honored → returns False); `test_authorized_still_accepts_bearer` passes.

- [ ] **Step 3: Update `_authorized` in `orchestra/gateway.py`**:

```python
def _authorized(request: web.Request, token: str) -> bool:
    if not token:
        return True
    token_bytes = token.encode("utf-8")
    api_key = request.headers.get("x-api-key")
    if api_key is not None and hmac.compare_digest(api_key.encode("utf-8"), token_bytes):
        return True
    header = request.headers.get("Authorization", "")
    parts = header.split()
    return (
        len(parts) == 2
        and parts[0].lower() == "bearer"
        and hmac.compare_digest(parts[1].encode("utf-8"), token_bytes)
    )
```

- [ ] **Step 4: Run to verify all auth tests pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_gateway.py -q --no-cov -p no:cacheprovider -o addopts=""`
Expected: PASS (all, including the new two and the existing non-ASCII/Bearer tests).

- [ ] **Step 5: Commit**

```bash
git add orchestra/gateway.py tests/test_gateway.py
git commit -m "feat(gateway): accept x-api-key auth header (Anthropic native) alongside Bearer"
```

---

### Task 5: gateway `/v1/messages` routes (non-stream + stream + count_tokens)

**Files:**
- Modify: `orchestra/gateway.py`
- Create: `tests/test_anthropic_gateway.py`

**Interfaces:**
- Consumes: `gateway_dispatch.run_target`/`stream_target` (Task 1), `anthropic_compat.*` (Tasks 2-3), `_authorized` (Task 4).
- Produces: routes `POST /v1/messages`, `POST /v1/messages/count_tokens`.

- [ ] **Step 1: Write the failing tests** in `tests/test_anthropic_gateway.py`:

```python
from __future__ import annotations

import json
import pytest
from unittest.mock import patch, AsyncMock
from aiohttp.test_utils import TestClient, TestServer

from orchestra import gateway


async def _fake_run_target(model, prompt):
    return f"reply for {model}"


async def _fake_stream(model, prompt):
    for c in ["Hel", "lo"]:
        yield c


@pytest.mark.asyncio
async def test_messages_non_stream():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.run_target", _fake_run_target):
            resp = await client.post("/v1/messages", json={
                "model": "oll-fast", "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            })
            assert resp.status == 200
            body = await resp.json()
        assert body["type"] == "message" and body["role"] == "assistant"
        assert body["content"][0]["text"] == "reply for oll-fast"
        assert body["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_messages_lenient_unknown_model_uses_default():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.run_target", new=AsyncMock(return_value="ok")) as mock_run:
            resp = await client.post("/v1/messages", json={
                "model": "claude-3-5-haiku", "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            })
            assert resp.status == 200
        # unknown model routed to the default alias, not 404
        assert mock_run.call_args.args[0] == "oll-fast"


@pytest.mark.asyncio
async def test_messages_stream_event_sequence():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.stream_target", _fake_stream):
            resp = await client.post("/v1/messages", json={
                "model": "oll-fast", "max_tokens": 16, "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            })
            assert resp.status == 200
            text = await resp.text()
    events = [ln[len("event: "):] for ln in text.splitlines() if ln.startswith("event: ")]
    assert events[0] == "message_start"
    assert events[1] == "content_block_start"
    assert "content_block_delta" in events
    assert events[-3:] == ["content_block_stop", "message_delta", "message_stop"]
    assert "text_delta" in text and "Hel" in text


@pytest.mark.asyncio
async def test_messages_requires_token_when_set():
    async with TestClient(TestServer(gateway.make_app(token="secret"))) as client:
        with patch("orchestra.gateway_dispatch.run_target", new=AsyncMock(return_value="x")) as mock_run:
            resp = await client.post("/v1/messages", json={
                "model": "oll-fast", "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            })
            assert resp.status == 401
            body = await resp.json()
        assert body["type"] == "error" and body["error"]["type"] == "authentication_error"
        mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_messages_x_api_key_authorizes():
    async with TestClient(TestServer(gateway.make_app(token="secret"))) as client:
        with patch("orchestra.gateway_dispatch.run_target", _fake_run_target):
            resp = await client.post("/v1/messages",
                headers={"x-api-key": "secret"},
                json={"model": "oll-fast", "max_tokens": 16,
                      "messages": [{"role": "user", "content": "hi"}]})
            assert resp.status == 200


@pytest.mark.asyncio
async def test_messages_missing_messages_400():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        resp = await client.post("/v1/messages", json={"model": "oll-fast", "max_tokens": 16})
        assert resp.status == 400
        body = await resp.json()
        assert body["error"]["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_messages_run_failure_500_generic():
    async def _explode(model, prompt):
        raise RuntimeError("secret /path token=abc")
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.run_target", _explode):
            resp = await client.post("/v1/messages", json={
                "model": "oll-fast", "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            })
            assert resp.status == 500
            raw = await resp.text()
        assert "secret" not in raw and "token=abc" not in raw
        assert '"type": "error"' in raw


@pytest.mark.asyncio
async def test_count_tokens():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        resp = await client.post("/v1/messages/count_tokens", json={
            "model": "oll-fast",
            "messages": [{"role": "user", "content": "hello world"}],
        })
        assert resp.status == 200
        body = await resp.json()
        assert isinstance(body["input_tokens"], int) and body["input_tokens"] >= 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_anthropic_gateway.py -q --no-cov -p no:cacheprovider -o addopts=""`
Expected: FAIL (404 — routes not registered yet).

- [ ] **Step 3: Add the handlers + routes in `orchestra/gateway.py`** inside `make_app`, next to the `_v1_chat` handlers. Add this helper and three handlers, and register the two routes:

```python
    def _anthropic_error(message: str, err_type: str, status: int) -> web.Response:
        return web.json_response(
            {"type": "error", "error": {"type": err_type, "message": message}}, status=status
        )

    async def _v1_messages(request: web.Request):
        from orchestra import gateway_dispatch, anthropic_compat
        if not _authorized(request, token):
            return _anthropic_error("authentication failed", "authentication_error", 401)
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return _anthropic_error("Invalid JSON", "invalid_request_error", 400)
        if not isinstance(payload, dict) or not payload.get("messages"):
            return _anthropic_error("messages is required", "invalid_request_error", 400)
        model = anthropic_compat.resolve_anthropic_model(payload.get("model", "oll-fast"))
        prompt = anthropic_compat.request_to_prompt(payload.get("system"), payload["messages"])
        if payload.get("stream"):
            return await _v1_messages_stream(request, model, prompt)
        try:
            text = await gateway_dispatch.run_target(model, prompt)
        except Exception as exc:
            logger.warning("anthropic messages run failed: %s", exc)
            return _anthropic_error("internal error running the model", "api_error", 500)
        return web.json_response(anthropic_compat.to_message_response(
            model, text,
            input_tokens=anthropic_compat.approx_tokens(prompt),
            output_tokens=anthropic_compat.approx_tokens(text),
        ))

    async def _v1_messages_stream(request: web.Request, model: str, prompt: str):
        from orchestra import gateway_dispatch, anthropic_compat
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(request)
        msg_id = anthropic_compat.new_message_id()
        agen = gateway_dispatch.stream_target(model, prompt)
        output = 0
        with contextlib.suppress(ConnectionResetError, ClientError):
            await resp.write(anthropic_compat.sse_message_start(model, msg_id, anthropic_compat.approx_tokens(prompt)).encode())
            await resp.write(anthropic_compat.sse_content_block_start().encode())
        try:
            async for chunk in agen:
                output += anthropic_compat.approx_tokens(chunk)
                await resp.write(anthropic_compat.sse_text_delta(chunk).encode())
        except (ConnectionResetError, ClientError):
            return resp  # client disconnected
        except Exception as exc:
            logger.warning("anthropic messages stream failed: %s", exc)
            with contextlib.suppress(ConnectionResetError, ClientError):
                await resp.write(anthropic_compat.sse_text_delta("[stream error]").encode())
        finally:
            await agen.aclose()  # deterministically stops the producer thread
        with contextlib.suppress(ConnectionResetError, ClientError):
            await resp.write(anthropic_compat.sse_content_block_stop().encode())
            await resp.write(anthropic_compat.sse_message_delta(output).encode())
            await resp.write(anthropic_compat.sse_message_stop().encode())
            await resp.write_eof()
        return resp

    async def _v1_messages_count_tokens(request: web.Request):
        from orchestra import anthropic_compat
        if not _authorized(request, token):
            return _anthropic_error("authentication failed", "authentication_error", 401)
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return _anthropic_error("Invalid JSON", "invalid_request_error", 400)
        if not isinstance(payload, dict) or not payload.get("messages"):
            return _anthropic_error("messages is required", "invalid_request_error", 400)
        prompt = anthropic_compat.request_to_prompt(payload.get("system"), payload["messages"])
        return web.json_response({"input_tokens": anthropic_compat.approx_tokens(prompt)})
```

Then register the routes (next to `app.router.add_post("/v1/chat/completions", _v1_chat)`):

```python
    app.router.add_post("/v1/messages", _v1_messages)
    app.router.add_post("/v1/messages/count_tokens", _v1_messages_count_tokens)
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_anthropic_gateway.py -q --no-cov -p no:cacheprovider -o addopts=""`
Expected: PASS (8 tests).

- [ ] **Step 5: Run the full gateway/compat suite for regressions**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_anthropic_gateway.py /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_anthropic_compat.py /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_openai_gateway.py /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_gateway.py /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_gateway_dispatch.py -q --no-cov -p no:cacheprovider -o addopts=""`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add orchestra/gateway.py tests/test_anthropic_gateway.py
git commit -m "feat(gateway): Anthropic-compatible /v1/messages + count_tokens routes"
```

---

### Task 6: docs + live smoke (Claude Code)

**Files:**
- Modify: `CLAUDE.md`, `docs/architecture.md`
- No new test; a local smoke with real Ollama + Claude Code.

- [ ] **Step 1: Add to `CLAUDE.md`** in `## Setup & Commands`, right after the OpenAI `orchestra serve` line:

```
# the same server also speaks the Anthropic API at http://<host>:7070/v1/messages —
# point Claude Code at it:  ANTHROPIC_BASE_URL=http://<host>:7070 ANTHROPIC_API_KEY=<secret>
#   ANTHROPIC_MODEL=oll-fast claude   (models: any alias or orchestra-dual/critical/auto;
#   unknown model names fall back to oll-fast; tool use is not supported — text only)
```

- [ ] **Step 2: Architecture note** — in `docs/architecture.md`, extend the gateway sentence added for the OpenAI endpoint to also mention: `gateway.py` also serves an Anthropic-compatible API (`/v1/messages`, `/v1/messages/count_tokens`) via `anthropic_compat.py`, sharing the provider dispatch with the OpenAI endpoint through `gateway_dispatch.py`.

- [ ] **Step 3: Local smoke (real Ollama + Claude Code)**

```bash
.venv/bin/orchestra serve --host 127.0.0.1 --port 7099 --token smoke &
sleep 3
# non-stream curl (Anthropic shape):
curl -s http://127.0.0.1:7099/v1/messages -H "x-api-key: smoke" -H "anthropic-version: 2023-06-01" \
  -d '{"model":"oll-fast","max_tokens":32,"messages":[{"role":"user","content":"say hi in 2 words"}]}' | head -c 400
# Claude Code end-to-end (headless):
ANTHROPIC_BASE_URL=http://127.0.0.1:7099 ANTHROPIC_API_KEY=smoke ANTHROPIC_MODEL=oll-fast \
  claude -p "reply with the two words: hello there"
# kill the server afterward
```
Expected: the curl returns an Anthropic `message` JSON with a text block; `claude -p` prints a reply routed through Orchestra → Ollama. No traceback.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/architecture.md
git commit -m "docs(anthropic): document the /v1/messages gateway and Claude Code setup"
```

---

## Verification (end-to-end)

1. `orchestra serve --host 0.0.0.0 --token X`; point Claude Code at `ANTHROPIC_BASE_URL=http://<host>:7070`, `ANTHROPIC_API_KEY=X`, `ANTHROPIC_MODEL=oll-fast` (or `orchestra-dual`).
2. `claude -p "..."` returns a reply; streaming shows incremental text.
3. Unknown model (Claude Code's background `claude-3-5-haiku`) does NOT error — routes to `oll-fast`.
4. `curl /v1/messages` non-stream and `stream:true` both work; missing messages → 400, wrong token → 401 (Anthropic error shape).
5. Existing `/run`, `/aliases`, `/health`, `/v1/chat/completions`, `/v1/models` and `ask`/`chat`/opencode are unaffected; full fast suite from a scratch dir shows no new regressions.

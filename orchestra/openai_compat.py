"""OpenAI-compatible translation for the Orchestra gateway (/v1/chat/completions)."""
from __future__ import annotations

import asyncio
import json
import time
import uuid

from orchestra.providers.fallback import available_aliases, resolve, known_aliases
from orchestra.engine.runners.common import run_dual_async, run_critical_async

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
    """('mode', <mode>) for orchestra-<mode>, else ('alias', <model>).

    Raises ValueError for an unknown orchestration mode OR an unknown alias
    (base name not in the alias table) — the gateway maps this to a 404.
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


def _new_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def to_response(
    model: str,
    content: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    _id: str | None = None,
    _created: int | None = None,
) -> dict:
    """Build an OpenAI chat.completion response envelope."""
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


def to_sse_chunk(
    model: str,
    delta_content: str = "",
    finish_reason: str | None = None,
    _id: str | None = None,
    _created: int | None = None,
) -> str:
    """Return a single SSE line for a chat.completion.chunk."""
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
    """Return the SSE stream terminator."""
    return "data: [DONE]\n\n"


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


# ---------------------------------------------------------------------------
# Run dispatch
# ---------------------------------------------------------------------------

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
    """Full (non-streaming) reply for a model id.

    Single alias  → resolve() then provider.run() via thread.
    Mode (dual)   → run_dual_async → OrchestraRun.summary.
    Mode (critical) → run_critical_async → OrchestraRun.summary.
    Mode (auto)   → run_task(mode="auto") in a thread → manifest dict.
    """
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


def _start_bg(fn) -> None:
    """Launch fn in a daemon thread (no-op wrapper for asyncio.to_thread scheduling)."""
    import threading
    threading.Thread(target=fn, daemon=True).start()


async def stream_target(model: str, prompt: str):
    """Yield reply chunks asynchronously.

    Single alias with streaming provider → live tokens via a thread-bridged Queue.
    Provider raises NotImplementedError (base/CLI/remote) → falls back to run().
    Orchestration modes → yields the full run_target reply once (no token stream).
    """
    kind, _ = resolve_target(model)
    if kind == "alias":
        provider, effort = resolve(model)
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        _SENTINEL = object()

        def _produce() -> None:
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
        # Orchestration modes don't token-stream; emit the whole reply once.
        yield await run_target(model, prompt)

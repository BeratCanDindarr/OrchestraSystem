"""OpenAI-compatible translation for the Orchestra gateway (/v1/chat/completions)."""
from __future__ import annotations

import json
import time
import uuid

from orchestra.providers.fallback import available_alias_ids
from orchestra.chat.history import render_history


def messages_to_prompt(messages: list[dict]) -> str:
    """Concatenate an OpenAI messages array into one prompt, prepending any system turns."""
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    body = render_history(rest)
    if not system:
        return body
    prefix = "\n\n".join(f"System: {m.get('content', '')}" for m in system)
    return f"{prefix}\n\n{body}" if body else f"{prefix}\n\nAssistant:"


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
    """Available single aliases (✓) + the orchestra-<mode> models."""
    models = [{"id": a, "object": "model", "owned_by": "orchestra"} for a in available_alias_ids()]
    for mode in ("dual", "consensus", "auto"):
        models.append({"id": f"orchestra-{mode}", "object": "model", "owned_by": "orchestra"})
    return models

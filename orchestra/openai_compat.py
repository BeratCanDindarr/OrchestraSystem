"""OpenAI-compatible translation for the Orchestra gateway (/v1/chat/completions)."""
from __future__ import annotations

import json
import time
import uuid

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

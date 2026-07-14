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

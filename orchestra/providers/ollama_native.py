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

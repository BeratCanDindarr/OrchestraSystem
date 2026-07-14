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

"""Render conversation history into a single concatenated prompt (portable across providers)."""
from __future__ import annotations

_ROLE_LABEL = {"user": "User", "assistant": "Assistant"}


def render_history(history: list[dict]) -> str:
    """Turn [{role, content}, ...] into 'User: …\\n\\nAssistant: …\\n\\nUser: …\\n\\nAssistant:'."""
    if not history:
        return ""
    lines = [f"{_ROLE_LABEL.get(m.get('role', ''), (m.get('role') or 'user').title())}: {m.get('content', '')}" for m in history]
    return "\n\n".join(lines) + "\n\nAssistant:"

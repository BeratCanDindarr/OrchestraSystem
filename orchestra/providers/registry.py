"""Central alias registry for Orchestra agent aliases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AgentAlias:
    name: str
    provider: str
    model: str
    tier: Literal["light", "medium", "heavy"]
    cost_per_1k: float


class AliasRegistry:
    _ALIASES: dict[str, AgentAlias] = {
        "cdx-fast": AgentAlias("cdx-fast", "codex", "gpt-5.4/low", "light", 0.004),
        "cdx-deep": AgentAlias("cdx-deep", "codex", "gpt-5.4/xhigh", "heavy", 0.012),
        "gmn-fast": AgentAlias("gmn-fast", "antigravity", "antigravity/gemini-3.8-flash-high", "light", 0.001),
        "gmn-pro": AgentAlias("gmn-pro", "antigravity", "antigravity/gemini-4-argon-high", "heavy", 0.005),
        "cld-fast": AgentAlias("cld-fast", "claude", "claude/sonnet", "light", 0.003),
        "cld-deep": AgentAlias("cld-deep", "claude", "claude/opus", "heavy", 0.015),
        "cld-write": AgentAlias("cld-write", "claude", "claude/opus", "heavy", 0.015),
        "oll-coder": AgentAlias("oll-coder", "ollama", "minimax-m3:cloud", "medium", 0.002),
        "oll-analyst": AgentAlias("oll-analyst", "ollama", "glm-5.3:cloud", "heavy", 0.003),
        "oll-fast": AgentAlias("oll-fast", "ollama", "deepseek-v4.1-flash:cloud", "light", 0.001),
        "oll-deep": AgentAlias("oll-deep", "ollama", "deepseek-v4-pro:cloud", "heavy", 0.004),
        "kimi-deep": AgentAlias("kimi-deep", "kimi", "kimi/k3", "heavy", 0.005),
        "kimi-code": AgentAlias("kimi-code", "kimi", "kimi/kimi-for-coding", "medium", 0.004),
        "oll-glm": AgentAlias("oll-glm", "ollama", "glm-5.3:cloud", "heavy", 0.003),
        "oll-flash": AgentAlias("oll-flash", "ollama", "glm-5.3-flash:cloud", "light", 0.001),
        "oll-vision": AgentAlias("oll-vision", "ollama", "gemma4:31b-cloud", "light", 0.0004),
        "oll-gemma": AgentAlias("oll-gemma", "ollama", "gemma4:31b-cloud", "light", 0.0004),
        "oll-default": AgentAlias("oll-default", "ollama", "gemma4:31b-cloud", "light", 0.0004),
        "agy-argon": AgentAlias("agy-argon", "antigravity", "antigravity/gemini-4-argon-high", "heavy", 0.005),
        "agy-deep": AgentAlias("agy-deep", "antigravity", "antigravity/gemini-4-argon-high", "heavy", 0.005),
        "agy-flash": AgentAlias("agy-flash", "antigravity", "antigravity/gemini-3.8-flash-high", "light", 0.001),
        "agy-fast": AgentAlias("agy-fast", "antigravity", "antigravity/gemini-3.8-flash-high", "light", 0.001),
    }

    @classmethod
    def get(cls, name: str) -> AgentAlias | None:
        return cls._ALIASES.get(name)

    @classmethod
    def list_aliases(cls) -> list[str]:
        return list(cls._ALIASES.keys())

    @classmethod
    def by_provider(cls, provider: str) -> list[AgentAlias]:
        return [a for a in cls._ALIASES.values() if a.provider == provider]

    @classmethod
    def by_tier(cls, tier: str) -> list[AgentAlias]:
        return [a for a in cls._ALIASES.values() if a.tier == tier]

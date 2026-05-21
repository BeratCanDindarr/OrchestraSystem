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
        "gmn-fast": AgentAlias("gmn-fast", "gemini", "gemini/flash-lite", "light", 0.002),
        "gmn-pro": AgentAlias("gmn-pro", "gemini", "gemini/pro", "medium", 0.006),
        "cld-fast": AgentAlias("cld-fast", "claude", "claude/sonnet", "light", 0.003),
        "cld-deep": AgentAlias("cld-deep", "claude", "claude/opus", "heavy", 0.015),
        "oll-coder": AgentAlias("oll-coder", "ollama", "qwen2.5-coder:7b", "light", 0.0),
        "oll-analyst": AgentAlias("oll-analyst", "ollama", "deepseek-r1:8b", "medium", 0.0),
        "oll-fast": AgentAlias("oll-fast", "ollama", "phi4-mini", "light", 0.0),
        "openai-fast": AgentAlias("openai-fast", "openai_sdk", "openai_sdk/gpt-4o-mini", "light", 0.001),
        "openai-pro": AgentAlias("openai-pro", "openai_sdk", "openai_sdk/gpt-4o", "medium", 0.01),
        "openai-reasoning": AgentAlias("openai-reasoning", "openai_sdk", "openai_sdk/o3-mini", "medium", 0.003),
        "ds-chat": AgentAlias("ds-chat", "deepseek", "deepseek/deepseek-chat", "light", 0.0002),
        "ds-reasoner": AgentAlias("ds-reasoner", "deepseek", "deepseek/deepseek-reasoner", "medium", 0.001),
        "pplx-fast": AgentAlias("pplx-fast", "perplexity", "perplexity/sonar", "light", 0.001),
        "pplx-pro": AgentAlias("pplx-pro", "perplexity", "perplexity/sonar-pro", "medium", 0.009),
        "pplx-reasoning": AgentAlias("pplx-reasoning", "perplexity", "perplexity/sonar-reasoning", "medium", 0.003),
        "az-fast": AgentAlias("az-fast", "azure_openai", "azure_openai/gpt-4o-mini", "light", 0.001),
        "az-pro": AgentAlias("az-pro", "azure_openai", "azure_openai/gpt-4o", "medium", 0.01),
        "az-reasoning": AgentAlias("az-reasoning", "azure_openai", "azure_openai/o3-mini", "medium", 0.003),
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

"""Fallback chain: resolve alias → (provider, effort_or_model)."""
from __future__ import annotations

from rich.console import Console

from orchestra.engine import provider_guard
from orchestra.providers.claude import ClaudeProvider
from orchestra.providers.codex import CodexProvider
from orchestra.providers.gemini import GeminiProvider
from orchestra.providers.ollama import OllamaProvider
from orchestra.providers.openai_sdk import OpenAISDKProvider
from orchestra.providers.azure_openai import AzureOpenAIProvider
from orchestra.providers.deepseek import DeepSeekProvider
from orchestra.providers.perplexity import PerplexityProvider
from orchestra.providers.registry import AliasRegistry

_codex = CodexProvider()
_gemini = GeminiProvider()
_claude = ClaudeProvider()
_ollama = OllamaProvider()
_openai = OpenAISDKProvider()
_azure = AzureOpenAIProvider()
_deepseek = DeepSeekProvider()
_perplexity = PerplexityProvider()
_console = Console(stderr=True)

# Alias → (provider_instance, effort_or_model)
ALIAS_TABLE: dict[str, tuple] = {
    "cdx-fast":  (_codex,  "low"),
    "cdx-deep":  (_codex,  "xhigh"),
    "gmn-fast":  (_gemini, "flash"),
    "gmn-pro":   (_gemini, "pro"),
    "cld-fast":  (_claude, "sonnet"),
    "cld-deep":  (_claude, "opus"),
    "oll-coder":   (_ollama, "coder"),
    "oll-analyst": (_ollama, "analyst"),
    "oll-fast":    (_ollama, "fast"),
    "oll-mini":    (_ollama, "mini"),
    "oll-deep":    (_ollama, "llama3"),
    "oll-kimi":    (_ollama, "kimi-k2.6:cloud"),
    "oll-minimax": (_ollama, "minimax-m2.7:cloud"),
    "oll-dev":     (_ollama, "minimax-m2.7:cloud"),
    "oll-embed":   (_ollama, "embed"),
    "openai-fast":      (_openai, "gpt-4o-mini"),
    "openai-pro":       (_openai, "gpt-4o"),
    "openai-reasoning": (_openai, "o3-mini"),
    "ds-chat":          (_deepseek, "deepseek-chat"),
    "ds-reasoner":      (_deepseek, "deepseek-reasoner"),
    "pplx-fast":        (_perplexity, "sonar"),
    "pplx-pro":         (_perplexity, "sonar-pro"),
    "pplx-reasoning":   (_perplexity, "sonar-reasoning"),
    "az-fast":          (_azure, "gpt-4o-mini"),
    "az-pro":           (_azure, "gpt-4o"),
    "az-reasoning":     (_azure, "o3-mini"),
}

def _is_backend_ref(alias: str) -> bool:
    return "@" in alias


def _resolve_at_ref(alias: str) -> tuple:
    """'left@X' -> remote node (preferred) or Ollama backend provider."""
    left, _, target = alias.partition("@")
    from orchestra import remote_nodes
    if target in remote_nodes.list_nodes():
        return remote_nodes.get_node(target), left
    from orchestra.providers import ollama_backends
    return ollama_backends.get_backend(target), left


def _canonical_alias(alias: str) -> str:
    if alias in ALIAS_TABLE:
        return alias
    if alias == "oll-dev":
        return "oll-dev"
    if alias.startswith("cld"):
        return "cld-deep"
    if alias.startswith("gmn"):
        return "gmn-pro"
    if alias == "oll-coder":
        return "oll-coder"
    if alias.startswith("oll"):
        return "oll-minimax"
    return "cdx-deep"


def _fallback_alias(alias: str) -> str:
    canonical = _canonical_alias(alias)
    if canonical.startswith("cld"):
        return "oll-minimax"
    if canonical.startswith("oll"):
        return "cdx-deep"
    if canonical.startswith("gmn"):
        return "cdx-deep"
    return "cld-deep"


def fallback_chain(alias: str) -> list[str]:
    canonical = _canonical_alias(alias)
    if canonical == "oll-dev":
        return ["oll-dev", "oll-minimax", "cld-fast", "cdx-deep"]
    if canonical == "oll-coder":
        return ["oll-coder", "oll-minimax", "cld-fast"]
    if canonical == "oll-deep":
        return ["oll-deep", "oll-kimi", "cdx-deep"]
    if canonical == "oll-fast":
        return ["oll-fast", "oll-minimax", "gmn-fast", "cdx-fast"]
    if canonical == "gmn-pro":
        return ["gmn-pro", "gmn-fast", "cdx-deep"]
    if canonical == "gmn-fast":
        return ["gmn-fast", "cdx-fast", "cld-fast", "cdx-deep"]
    if canonical == "cld-deep":
        return ["cld-deep", "oll-minimax", "cdx-deep", "gmn-pro"]
    if canonical == "cld-fast":
        return ["cld-fast", "oll-minimax", "cdx-fast", "gmn-fast"]
    if canonical == "cdx-deep":
        return ["cdx-deep", "gmn-pro", "cld-deep", "cdx-fast"]
    if canonical == "cdx-fast":
        return ["cdx-fast", "gmn-fast", "cld-fast", "cdx-deep"]
    return [canonical, _fallback_alias(canonical)]


def resolve(alias: str) -> tuple:
    """
    Resolve alias to (provider, effort_or_model).
    Falls back to next available provider if primary is down.
    """
    if _is_backend_ref(alias):
        return _resolve_at_ref(alias)
    alias = _canonical_alias(alias)

    provider, effort = ALIAS_TABLE[alias]

    if provider.is_available():
        return provider, effort

    fallback_alias = _fallback_alias(alias)
    fb_provider, fb_effort = ALIAS_TABLE[fallback_alias]
    if fb_provider.is_available():
        _console.print(
            f"[yellow]Provider '{provider.name}' is unavailable for alias '{alias}'. "
            f"Falling back to '{fallback_alias}'.[/yellow]"
        )
        return fb_provider, fb_effort

    return provider, effort  # Return original even if unavailable


def resolve_with_fallback(alias: str, attempt: int = 0) -> tuple:
    """
    Resolve alias to (provider, effort_or_model), switching to the fallback
    alias on retry attempts.
    """
    if _is_backend_ref(alias):
        return _resolve_at_ref(alias)
    chain = fallback_chain(alias)
    start_index = min(attempt, len(chain) - 1)
    for candidate_alias in chain[start_index:]:
        provider, effort = ALIAS_TABLE[candidate_alias]
        if provider.is_available() and provider_guard.can_use(provider.name):
            return provider, effort
    selected_alias = chain[start_index]
    if attempt <= 0:
        return resolve(selected_alias)
    provider, effort = ALIAS_TABLE[selected_alias]
    return provider, effort


def known_aliases() -> set[str]:
    """The set of canonical single-model alias names (ALIAS_TABLE keys)."""
    return set(ALIAS_TABLE)


def available_aliases() -> list[str]:
    result = []
    for alias, (provider, _) in ALIAS_TABLE.items():
        status = "✓" if provider.is_available() else "✗"
        registry_entry = AliasRegistry.get(alias)
        cost_str = f"  ${registry_entry.cost_per_1k}/1k" if registry_entry else ""
        result.append(f"{status} {alias:<13} {cost_str}")
    result.append("  auto     (router decides mode)")
    result.append("  dual     (cdx-deep + gmn-pro)")
    result.append("  critical (dual + synthesis)")
    result.append("  expert   (dual + oll-kimi + synthesis)")
    return result

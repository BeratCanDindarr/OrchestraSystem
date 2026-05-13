"""Fallback chain: resolve alias → (provider, effort_or_model)."""
from __future__ import annotations

from rich.console import Console

from orchestra.engine import provider_guard
from orchestra.providers.claude import ClaudeProvider
from orchestra.providers.codex import CodexProvider
from orchestra.providers.gemini import GeminiProvider
from orchestra.providers.ollama import OllamaProvider

_codex = CodexProvider()
_gemini = GeminiProvider()
_claude = ClaudeProvider()
_ollama = OllamaProvider()
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
    "oll-embed":   (_ollama, "embed"),
}

def _canonical_alias(alias: str) -> str:
    if alias in ALIAS_TABLE:
        return alias
    if alias.startswith("cld"):
        return "cld-deep"
    if alias.startswith("gmn"):
        return "gmn-pro"
    if alias == "oll-coder":
        return "oll-coder"
    if alias.startswith("oll"):
        return "oll-coder"
    return "cdx-deep"


def _fallback_alias(alias: str) -> str:
    canonical = _canonical_alias(alias)
    if canonical.startswith("oll"):
        return "cdx-deep"
    if canonical.startswith("cld"):
        return "cdx-deep"
    if canonical.startswith("gmn"):
        return "cdx-deep"
    return "cld-deep"


def fallback_chain(alias: str) -> list[str]:
    canonical = _canonical_alias(alias)
    if canonical == "oll-coder":
        return ["oll-coder", "cld-fast"]
    if canonical == "oll-deep":
        return ["oll-deep", "oll-coder", "cdx-deep"]
    if canonical == "oll-fast":
        return ["oll-fast", "oll-coder", "gmn-fast", "cdx-fast", "cdx-deep"]
    if canonical == "gmn-pro":
        return ["gmn-pro", "gmn-fast", "cdx-deep"]
    if canonical == "gmn-fast":
        return ["gmn-fast", "cdx-fast", "cld-fast", "cdx-deep"]
    if canonical == "cld-deep":
        return ["cld-deep", "cdx-deep", "gmn-pro", "gmn-fast"]
    if canonical == "cld-fast":
        return ["cld-fast", "cdx-fast", "gmn-fast", "cdx-deep"]
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


def available_aliases() -> list[str]:
    result = []
    for alias, (provider, _) in ALIAS_TABLE.items():
        status = "✓" if provider.is_available() else "✗"
        result.append(f"{status} {alias}")
    result.append("  auto     (router decides mode)")
    result.append("  dual     (cdx-deep + gmn-pro)")
    result.append("  critical (dual + synthesis)")
    return result

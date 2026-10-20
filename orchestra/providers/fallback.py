"""Fallback chain: resolve alias → (provider, effort_or_model)."""
from __future__ import annotations

import difflib

from rich.console import Console

from orchestra import config
from orchestra.engine import provider_guard
from orchestra.providers.claude import ClaudeProvider
from orchestra.providers.codex import CodexProvider
from orchestra.providers.gemini import GeminiProvider
from orchestra.providers.kimi import KimiProvider
from orchestra.providers.antigravity import AntigravityProvider
from orchestra.providers.ollama import OllamaProvider
from orchestra.providers.openai_sdk import OpenAISDKProvider
from orchestra.providers.azure_openai import AzureOpenAIProvider
from orchestra.providers.deepseek import DeepSeekProvider
from orchestra.providers.perplexity import PerplexityProvider
from orchestra.providers.registry import AliasRegistry

_codex = CodexProvider()
_gemini = GeminiProvider()
_claude = ClaudeProvider()
# Same CLI with its own config section ([providers.claude_write]): the profile that may change files.
_claude_write = ClaudeProvider(config_key="claude_write")
_kimi = KimiProvider()
_antigravity = AntigravityProvider()
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
    "cld-write": (_claude_write, "opus"),
    # Kimi runs through the logged-in Kimi CLI, never through an Ollama cloud tag.
    "kimi-deep": (_kimi, "k3"),
    "kimi-code": (_kimi, "code"),
    "agy-argon": (_antigravity, "argon-high"),
    "agy-deep":  (_antigravity, "heavy"),
    "agy-flash": (_antigravity, "flash"),
    "agy-fast":  (_antigravity, "light"),
    "oll-coder":   (_ollama, "coder"),
    "oll-analyst": (_ollama, "analyst"),
    "oll-fast":    (_ollama, "fast"),
    "oll-mini":    (_ollama, "mini"),
    "oll-deep":    (_ollama, "deepseek-v4-pro:cloud"),
    "oll-glm":     (_ollama, "glm-5.3:cloud"),
    "oll-flash":   (_ollama, "glm-5.3-flash:cloud"),
    # General-purpose default (also where unknown oll-* names land): Gemma 4, the best
    # Ollama model in the benchmark (docs/model-routing.md) and one that ignored an
    # instruction injected into its input, which DeepSeek V4 Pro obeyed.
    "oll-default": (_ollama, "gemma4:31b-cloud"),
    # Older names for the default, kept so saved runs, playbooks and configs still resolve.
    "oll-minimax": (_ollama, "gemma4:31b-cloud"),
    "oll-dev":     (_ollama, "gemma4:31b-cloud"),
    # Gemma 4: the one cloud model that read a game screenshot correctly; also cheap and fast.
    "oll-vision":  (_ollama, "gemma4:31b-cloud"),
    "oll-gemma":   (_ollama, "gemma4:31b-cloud"),
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


class UnknownAliasError(ValueError):
    """The name given is not an agent alias."""


class ProviderUnavailableError(RuntimeError):
    """The alias's provider is switched off and nothing in its fallback chain can run."""


# Aliases that answer over an HTTP API and can do nothing but answer. They do not fall back to
# a CLI agent: a missing API key would turn a cheap text call into an agent that reads the
# project (or, with a writing profile, changes it).
_TEXT_ONLY_PREFIXES = ("openai-", "ds-", "pplx-", "az-")


def _usable(provider) -> bool:
    """Whether a provider may and can run: not switched off in `[providers.<name>]`, and available."""
    return config.provider_enabled(provider.name) and provider.is_available()


def _enabled_or_raise(provider, alias: str) -> None:
    if not config.provider_enabled(provider.name):
        raise ProviderUnavailableError(
            f"Provider {provider.name!r} is switched off ([providers.{provider.name}] enabled = false) "
            f"and no fallback of {alias!r} can run."
        )


def require_known(alias: str) -> None:
    """Raise UnknownAliasError unless *alias* names an agent.

    Accepted: a table alias, optionally with a `#mode` suffix, or a `model@target`
    reference (whose target is checked when it is resolved). Called where a name enters a
    run, because `_canonical_alias` maps anything else to an agent of the same family or to
    cdx-deep: a typo used to run, and bill, a different model without a word.
    """
    if _is_backend_ref(alias):
        return
    base = alias.split("#", 1)[0]
    if base in ALIAS_TABLE:
        return
    close = difflib.get_close_matches(base, ALIAS_TABLE, n=3)
    hint = f"Did you mean: {', '.join(close)}?" if close else "List them with: orchestra aliases"
    raise UnknownAliasError(f"Unknown agent alias {alias!r}. {hint}")


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
    if alias.startswith("agy"):
        return "agy-argon"
    if alias.startswith("cld"):
        return "cld-deep"
    if alias.startswith("kimi"):
        return "kimi-deep"
    if alias.startswith("gmn"):
        return "gmn-pro"
    if alias == "oll-coder":
        return "oll-coder"
    if alias.startswith("oll"):
        return "oll-default"
    return "cdx-deep"


def _fallback_alias(alias: str) -> str:
    canonical = _canonical_alias(alias)
    if canonical.startswith(_TEXT_ONLY_PREFIXES):
        return canonical
    if canonical.startswith("agy"):
        return "cld-deep"
    if canonical.startswith("cld"):
        return "kimi-deep"
    if canonical.startswith("kimi"):
        return "cdx-deep"
    if canonical.startswith("oll"):
        return "cdx-deep"
    if canonical.startswith("gmn"):
        return "cdx-deep"
    return "cld-deep"


def fallback_chain(alias: str) -> list[str]:
    canonical = _canonical_alias(alias)
    if canonical == "agy-argon":
        return ["agy-argon", "agy-flash", "cld-deep", "cdx-deep"]
    if canonical == "agy-deep":
        return ["agy-deep", "agy-flash", "cld-deep", "cdx-deep"]
    if canonical == "agy-flash":
        return ["agy-flash", "agy-argon", "cld-fast", "cdx-fast"]
    if canonical == "agy-fast":
        return ["agy-fast", "agy-argon", "cld-fast", "cdx-fast"]
    if canonical == "oll-dev":
        return ["oll-dev", "kimi-deep", "cld-fast", "cdx-deep"]
    if canonical == "oll-coder":
        return ["oll-coder", "kimi-code", "cld-fast"]
    if canonical == "oll-deep":
        return ["oll-deep", "kimi-deep", "cdx-deep"]
    if canonical == "oll-fast":
        return ["oll-fast", "oll-flash", "gmn-fast", "cdx-fast"]
    if canonical == "kimi-code":
        return ["kimi-code", "kimi-deep", "cld-fast"]
    if canonical == "kimi-deep":
        return ["kimi-deep", "cld-deep", "cdx-deep"]
    if canonical == "oll-glm":
        return ["oll-glm", "kimi-deep", "cld-deep"]
    if canonical == "oll-flash":
        return ["oll-flash", "oll-fast", "gmn-fast", "cdx-fast"]
    if canonical == "gmn-pro":
        return ["gmn-pro", "gmn-fast", "cdx-deep"]
    if canonical == "gmn-fast":
        return ["gmn-fast", "cdx-fast", "cld-fast", "cdx-deep"]
    if canonical == "cld-deep":
        return ["cld-deep", "kimi-deep", "cdx-deep", "gmn-pro"]
    if canonical == "cld-fast":
        return ["cld-fast", "kimi-deep", "cdx-fast", "gmn-fast"]
    if canonical == "cdx-deep":
        return ["cdx-deep", "gmn-pro", "cld-deep", "cdx-fast"]
    if canonical == "cdx-fast":
        return ["cdx-fast", "gmn-fast", "cld-fast", "cdx-deep"]
    if canonical.startswith(_TEXT_ONLY_PREFIXES):
        return [canonical]
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

    if _usable(provider):
        return provider, effort

    # The same chain the retry logic walks, so the model announced here is the one that runs.
    for fallback_alias in [*fallback_chain(alias)[1:], _fallback_alias(alias)]:
        fb_provider, fb_effort = ALIAS_TABLE[fallback_alias]
        if _usable(fb_provider):
            _console.print(
                f"[yellow]Provider '{provider.name}' is unavailable for alias '{alias}'. "
                f"Falling back to '{fallback_alias}' ({fb_provider.model_label(fb_effort)}).[/yellow]"
            )
            return fb_provider, fb_effort

    # Nothing in the chain can run. An unavailable provider is returned so the call fails with
    # its own error; one that was switched off is not run at all.
    _enabled_or_raise(provider, alias)
    return provider, effort


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
        if _usable(provider) and provider_guard.can_use(provider.name):
            return provider, effort
    selected_alias = chain[start_index]
    if attempt <= 0:
        return resolve(selected_alias)
    provider, effort = ALIAS_TABLE[selected_alias]
    _enabled_or_raise(provider, selected_alias)
    return provider, effort


def can_run(alias: str) -> bool:
    """Whether the alias's own provider can run right now. A `model@target` is taken on trust."""
    if _is_backend_ref(alias):
        return True
    entry = ALIAS_TABLE.get(alias.split("#", 1)[0])
    return bool(entry and _usable(entry[0]))


def answering_alias(alias: str) -> str:
    """The alias that answers for *alias* right now: itself, or the first of its fallbacks
    whose provider can run. Unchanged when nothing in the chain can."""
    if can_run(alias) or alias.split("#", 1)[0] not in ALIAS_TABLE:
        return alias
    return next((candidate for candidate in fallback_chain(alias) if can_run(candidate)), alias)


def model_of(alias: str) -> str:
    """The model an alias names, for telling two agents apart."""
    entry = ALIAS_TABLE.get(alias.split("#", 1)[0])
    return entry[0].model_label(entry[1]) if entry and not _is_backend_ref(alias) else alias


def known_aliases() -> set[str]:
    """The set of canonical single-model alias names (ALIAS_TABLE keys)."""
    return set(ALIAS_TABLE)


def available_alias_ids() -> list[str]:
    """Alias ids whose provider is currently available (✓), in table order."""
    return [alias for alias, (provider, _) in ALIAS_TABLE.items() if _usable(provider)]


def available_aliases() -> list[str]:
    result = []
    for alias, (provider, _) in ALIAS_TABLE.items():
        status = "✓" if _usable(provider) else "✗"
        registry_entry = AliasRegistry.get(alias)
        cost_str = f"  ${registry_entry.cost_per_1k}/1k" if registry_entry else ""
        result.append(f"{status} {alias:<13} {cost_str}")
    from orchestra import config as _cfg
    personas = _cfg.personas_config()
    impl = personas.get("implementer", "cdx-deep")
    res = personas.get("researcher", "gmn-pro")
    ver = personas.get("verifier", "gmn-fast")
    result.append("  auto      (router decides mode)")
    result.append(f"  dual      ({impl} + {res}, checked by {ver}, then synthesis)")
    result.append(f"  consensus ({impl} for, {res} against, a neutral third, then a decision)")
    return result

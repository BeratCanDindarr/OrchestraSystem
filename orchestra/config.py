"""Load and expose Orchestra configuration."""
from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore


def _find_config() -> Path:
    """Walk up from CWD to find .orchestra/config.toml."""
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        candidate = parent / ".orchestra" / "config.toml"
        if candidate.exists():
            return candidate
    # Fallback: same dir as this file's package root
    return Path(__file__).parent.parent / ".orchestra" / "config.toml"


_config_path = _find_config()

if _config_path.exists():
    with open(_config_path, "rb") as _f:
        _cfg = tomllib.load(_f)
else:
    _cfg = {}


# ── Public accessors ─────────────────────────────────────────────

def get(section: str, key: str, default=None):
    return _cfg.get(section, {}).get(key, default)


def timeout() -> int:
    return _cfg.get("orchestra", {}).get("timeout", 180)


def timeout_for_alias(alias: str | None = None) -> int:
    default_timeout = timeout()
    if not alias:
        return default_timeout

    explicit = _cfg.get("timeouts", {}).get(alias)
    if explicit is not None:
        return int(explicit)

    deep_defaults = {
        "cdx-deep": 900,
        "gmn-pro": 300,
        "cld-deep": 300,
    }
    fast_defaults = {
        "cdx-fast": 180,
        "gmn-fast": 180,
        "cld-fast": 180,
    }
    if alias in deep_defaults:
        return deep_defaults[alias]
    if alias in fast_defaults:
        return fast_defaults[alias]
    return default_timeout


def max_turns() -> int:
    return _cfg.get("limits", {}).get("max_turns", 10)


def max_budget_usd() -> float:
    return _cfg.get("limits", {}).get("max_budget_usd", 5.0)


def auto_synthesize() -> bool:
    return bool(_cfg.get("orchestra", {}).get("auto_synthesize", False))


def config_path() -> Path:
    return _config_path


def orchestra_root() -> Path:
    return _config_path.parent


def project_root() -> Path:
    return orchestra_root().parent


def artifact_root() -> Path:
    rel = _cfg.get("orchestra", {}).get("artifact_root", "runs")
    return (orchestra_root() / rel).resolve()


def db_path() -> Path:
    return (orchestra_root() / "orchestra.db").resolve()


def alias_map() -> dict[str, str]:
    return _cfg.get("aliases", {})


def resolve_alias(alias: str) -> str:
    """Return raw alias value, or the alias itself if not found."""
    return alias_map().get(alias, alias)


def provider_config(provider: str) -> dict:
    return _cfg.get("providers", {}).get(provider, {})


def price_per_1k_tokens(model_label: str) -> float:
    pricing = _cfg.get("pricing", {})
    if model_label in pricing:
        return float(pricing[model_label])

    defaults = {
        "gpt-5.4/low": 0.002,
        "gpt-5.4/medium": 0.004,
        "gpt-5.4/high": 0.008,
        "gpt-5.4/xhigh": 0.012,
        "gemini/flash-lite": 0.001,
        "gemini/flash": 0.002,
        "gemini/pro": 0.006,
    }
    return float(defaults.get(model_label, 0.004))


def redaction_patterns() -> list[str]:
    return _cfg.get("redaction", {}).get("patterns", [])

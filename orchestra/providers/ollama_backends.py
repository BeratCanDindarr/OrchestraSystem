"""Registry of named Ollama backends (local + remote), each its own provider."""
from __future__ import annotations

from orchestra import config
from orchestra.providers.ollama import OllamaProvider

_DEFAULT_BACKENDS = {"macbook": "http://localhost:11434"}

_instances: dict[str, OllamaProvider] = {}


def list_backends() -> dict[str, str]:
    """name -> base_url. Defaults to a single local 'macbook' backend."""
    raw = config.ollama_backends_config()
    if not raw:
        return dict(_DEFAULT_BACKENDS)
    result: dict[str, str] = {}
    for name, entry in raw.items():
        if isinstance(entry, dict) and entry.get("base_url"):
            result[name] = str(entry["base_url"])
    return result or dict(_DEFAULT_BACKENDS)


def get_backend(name: str) -> OllamaProvider:
    """Return a cached OllamaProvider bound to that backend's base_url."""
    backends = list_backends()
    if name not in backends:
        known = ", ".join(sorted(backends)) or "(none)"
        raise ValueError(f"Unknown Ollama backend {name!r}. Known backends: {known}")
    if name not in _instances:
        _instances[name] = OllamaProvider(backends[name])
    return _instances[name]


def reset_cache() -> None:
    """Clear cached provider instances (tests + config reload)."""
    _instances.clear()

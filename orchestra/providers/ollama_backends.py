"""Registry of named Ollama backends (local + remote), each its own provider."""
from __future__ import annotations

import json
import time
import urllib.request

from orchestra import config
from orchestra.providers.ollama import OllamaProvider

_DEFAULT_BACKENDS = {"macbook": "http://localhost:11434"}

_instances: dict[str, OllamaProvider] = {}

_DISCOVERY_TTL_S = 60.0
_discovery_cache: dict[str, dict] | None = None
_discovery_at: float = 0.0


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


def _fetch_tags(base_url: str) -> tuple[bool, list[dict]]:
    """Return (online, models[]) by querying {base_url}/api/tags."""
    timeout = int(config.availability_config().get("ollama_healthcheck_timeout_seconds", 2))
    req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return False, []
            payload = json.loads(resp.read())
    except Exception:
        return False, []
    models = [
        {"name": m.get("name", "?"), "size": int(m.get("size", 0) or 0)}
        for m in payload.get("models", [])
    ]
    return True, models


def discover_all(refresh: bool = False) -> dict[str, dict]:
    """Discover models per backend. Cached for _DISCOVERY_TTL_S seconds."""
    global _discovery_cache, _discovery_at
    now = time.monotonic()
    if not refresh and _discovery_cache is not None and (now - _discovery_at) < _DISCOVERY_TTL_S:
        return _discovery_cache
    result: dict[str, dict] = {}
    for name, base_url in list_backends().items():
        online, models = _fetch_tags(base_url)
        result[name] = {"base_url": base_url, "online": online, "models": models}
    _discovery_cache = result
    _discovery_at = now
    return result


def reset_discovery_cache() -> None:
    """Clear cached discovery results (tests + config reload)."""
    global _discovery_cache, _discovery_at
    _discovery_cache = None
    _discovery_at = 0.0


def reset_cache() -> None:
    """Clear cached provider instances and discovery results (tests + config reload)."""
    _instances.clear()
    reset_discovery_cache()

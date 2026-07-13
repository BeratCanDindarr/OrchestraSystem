"""Registry of remote Orchestra nodes (agent gateways addressed as alias@node)."""
from __future__ import annotations

from orchestra import config
from orchestra.providers.remote_node import RemoteNodeProvider

_instances: dict[str, RemoteNodeProvider] = {}


def list_nodes() -> dict[str, dict]:
    """name -> {'url':..., 'token':...}. Skips entries without a url."""
    raw = config.remote_nodes_config()
    result: dict[str, dict] = {}
    for name, entry in raw.items():
        if isinstance(entry, dict) and entry.get("url"):
            result[name] = {"url": str(entry["url"]), "token": str(entry.get("token", ""))}
    return result


def get_node(name: str) -> RemoteNodeProvider:
    nodes = list_nodes()
    if name not in nodes:
        known = ", ".join(sorted(nodes)) or "(none)"
        raise ValueError(f"Unknown remote node {name!r}. Known nodes: {known}")
    if name not in _instances:
        _instances[name] = RemoteNodeProvider(name, nodes[name]["url"], nodes[name]["token"])
    return _instances[name]


def reset_cache() -> None:
    _instances.clear()

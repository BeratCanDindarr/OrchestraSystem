"""Registry of remote Orchestra nodes (agent gateways addressed as alias@node)."""
from __future__ import annotations

import json
import socket
import urllib.request
from concurrent.futures import ThreadPoolExecutor

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


def _primary_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def default_subnet_prefix() -> str:
    return ".".join(_primary_ip().split(".")[:3])


def _probe(ip: str, port: int, token: str) -> dict | None:
    url = f"http://{ip}:{port}/health"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=1) as resp:
            if resp.status != 200:
                return None
            body = json.loads(resp.read())
    except Exception:
        return None
    return {"ip": ip, "host": body.get("host", ip), "port": port}


def scan_subnet(subnet_prefix: str, token: str = "", port: int = 7070) -> list[dict]:
    """Parallel-probe {prefix}.1..254 for Orchestra gateways; return responders."""
    ips = [f"{subnet_prefix}.{i}" for i in range(1, 255)]
    found: list[dict] = []
    with ThreadPoolExecutor(max_workers=64) as pool:
        for result in pool.map(lambda ip: _probe(ip, port, token), ips):
            if result:
                found.append(result)
    return found

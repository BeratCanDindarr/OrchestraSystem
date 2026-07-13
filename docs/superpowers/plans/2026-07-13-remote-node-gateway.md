# Remote Node Agent Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Orchestra on a remote machine as an agent gateway (`orchestra serve`) so this machine can run the remote's logged-in CLI agents and Ollama via `alias@node`.

**Architecture:** A Mac-side `RemoteNodeProvider` POSTs `{alias, task}` to a remote node's aiohttp gateway (`/run`), which calls `run_ask` locally and returns the result. `fallback.resolve()` routes `alias@node` to the node; a node registry + `nodes scan` (parallel LAN probe, token-filtered, auto-add) manage nodes. Same `@`-addressing pattern as multi-backend Ollama, one layer up.

**Tech Stack:** Python 3.11+, aiohttp (existing dep — see `federation_http.py` for the server+bearer-auth pattern), Typer CLI, pytest with aiohttp test client + mocked HTTP. No new deps. No Rust.

## Global Constraints

- Pure Python; NO new dependencies; NO Rust.
- Backward compatibility: aliases WITHOUT `@` resolve exactly as before; existing `@backend` (Ollama) resolution keeps working.
- Addressing: `alias@node` where `alias` is one of the node's aliases (what `run_ask` takes). `@` resolution checks `[remote_nodes.X]` FIRST, then `[providers.ollama.backends.X]`.
- Auth: bearer token on every gateway request; `--host 0.0.0.0` is REFUSED without a token (bind `127.0.0.1` only until a token is set). The token grants remote code execution on the node — treat as a secret.
- No silent exception swallowing; functions <50 lines; files <800; immutability (new dicts).
- Discovery/scan is parallel (ThreadPoolExecutor), token-filtered; only token-valid gateways are auto-added.
- Run pytest FROM A SCRATCH DIR: `cd /tmp && /abs/.venv/bin/pytest <abs paths> -q --no-cov -p no:cacheprovider`.
- Conventional Commits; NO Co-Authored-By lines.

## File Structure

- `orchestra/providers/remote_node.py` — CREATE: `RemoteNodeProvider(BaseProvider)` (HTTP client to a node's `/run`, `/health`).
- `orchestra/remote_nodes.py` — CREATE: node registry (`list_nodes`/`get_node`/`reset_cache`) + `scan_subnet`.
- `orchestra/gateway.py` — CREATE: the `orchestra serve` aiohttp server (`/run`, `/aliases`, `/health`, bearer auth).
- `orchestra/config.py` — MODIFY: `remote_nodes_config()` accessor.
- `orchestra/providers/fallback.py` — MODIFY: extend the `@` fast-path to check remote nodes first.
- `orchestra/cli.py` — MODIFY: `serve`, `nodes`, `nodes scan` commands.
- Tests: `tests/test_remote_node_provider.py`, `tests/test_remote_nodes.py`, `tests/test_gateway.py`, `tests/test_fallback_node_ref.py`, `tests/test_cli_nodes.py`.

---

### Task 1: RemoteNodeProvider (HTTP client)

**Files:**
- Create: `orchestra/providers/remote_node.py`
- Test: `tests/test_remote_node_provider.py`

**Interfaces:**
- Produces: `RemoteNodeProvider(name: str, url: str, token: str = "")`; `name = "remote"`; methods `native_run(prompt, effort_or_model, timeout=180, **kwargs) -> tuple[str,int]` (POSTs `{alias: effort_or_model, task: prompt}` to `{url}/run` with `Authorization: Bearer {token}` when token set; returns `(stdout, 0)` on success, `("[ERROR] ...", 1)` on failure/non-200); `is_available() -> bool` (GET `{url}/health`); `model_label(alias) -> "remote/{name}/{alias}"`; `build_command` raises NotImplementedError.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_node_provider.py
import json
from unittest.mock import patch
from orchestra.providers.remote_node import RemoteNodeProvider


class _Resp:
    def __init__(self, status, body):
        self.status = status
        self._body = json.dumps(body).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return self._body


def test_native_run_posts_alias_and_task_with_token():
    p = RemoteNodeProvider("windows", "http://192.168.1.42:7070", token="secret")
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["auth"] = req.headers.get("Authorization")
        captured["body"] = json.loads(req.data)
        return _Resp(200, {"stdout": "hello from windows", "status": "completed"})

    with patch("orchestra.providers.remote_node.urllib.request.urlopen", _fake_urlopen):
        out, rc = p.native_run("do it", "cdx-deep")

    assert out == "hello from windows"
    assert rc == 0
    assert captured["url"] == "http://192.168.1.42:7070/run"
    assert captured["auth"] == "Bearer secret"
    assert captured["body"] == {"alias": "cdx-deep", "task": "do it"}


def test_native_run_non_200_is_error():
    p = RemoteNodeProvider("windows", "http://192.168.1.42:7070", token="secret")

    def _fake_urlopen(req, timeout=None):
        return _Resp(401, {"error": "Unauthorized"})

    with patch("orchestra.providers.remote_node.urllib.request.urlopen", _fake_urlopen):
        out, rc = p.native_run("do it", "cdx-deep")
    assert rc == 1
    assert "Unauthorized" in out or "401" in out


def test_model_label_shows_node_and_alias():
    p = RemoteNodeProvider("windows", "http://x:7070")
    assert p.model_label("cdx-deep") == "remote/windows/cdx-deep"


def test_is_available_hits_health():
    p = RemoteNodeProvider("windows", "http://192.168.1.42:7070")
    seen = {}

    def _fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        return _Resp(200, {"status": "ok"})

    with patch("orchestra.providers.remote_node.urllib.request.urlopen", _fake_urlopen):
        assert p.is_available() is True
    assert seen["url"] == "http://192.168.1.42:7070/health"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_remote_node_provider.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`No module named 'orchestra.providers.remote_node'`).

- [ ] **Step 3: Create the provider**

```python
# orchestra/providers/remote_node.py
"""Provider that delegates a task to a remote Orchestra node's agent gateway."""
from __future__ import annotations

import json as _json
import urllib.request
from typing import Any

from orchestra.providers.base import BaseProvider


class RemoteNodeProvider(BaseProvider):
    name = "remote"

    def __init__(self, name: str, url: str, token: str = "") -> None:
        self._node = name
        self._url = url.rstrip("/")
        self._token = token

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def is_available(self) -> bool:
        req = urllib.request.Request(f"{self._url}/health", headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def model_label(self, effort_or_model: str) -> str:
        return f"remote/{self._node}/{effort_or_model}"

    def build_command(self, prompt: str, effort_or_model: str) -> list[str]:
        raise NotImplementedError("RemoteNodeProvider runs over HTTP, not a subprocess")

    def native_run(self, prompt: str, effort_or_model: str, timeout: int = 180, **_kwargs) -> tuple[str, int]:
        payload = _json.dumps({"alias": effort_or_model, "task": prompt}).encode()
        req = urllib.request.Request(
            f"{self._url}/run", data=payload, headers=self._headers(), method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body: dict[str, Any] = _json.loads(resp.read())
                if resp.status != 200:
                    return f"[ERROR] node {self._node}: {body.get('error', resp.status)}", 1
                return str(body.get("stdout", "")), 0
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            return f"[ERROR] node {self._node} HTTP {exc.code}: {detail}", 1
        except Exception as exc:
            return f"[ERROR] cannot reach node {self._node!r} at {self._url}: {exc}", 1
```

Add `import urllib.error` at the top as well (needed for the HTTPError branch).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_remote_node_provider.py -q --no-cov -p no:cacheprovider`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add orchestra/providers/remote_node.py tests/test_remote_node_provider.py
git commit -m "feat(remote): RemoteNodeProvider — HTTP client to a node's agent gateway"
```

---

### Task 2: config accessor + node registry

**Files:**
- Modify: `orchestra/config.py` (add accessor near `ollama_backends_config`)
- Create: `orchestra/remote_nodes.py`
- Test: `tests/test_remote_nodes.py`

**Interfaces:**
- Consumes: `RemoteNodeProvider` from Task 1.
- Produces: `config.remote_nodes_config() -> dict` (raw `[remote_nodes]` table, empty when absent); `remote_nodes.list_nodes() -> dict[str, dict]` (name→`{"url":..,"token":..}`; skips entries with no `url`; empty when none); `remote_nodes.get_node(name) -> RemoteNodeProvider` (cached; raises ValueError listing known nodes on unknown); `remote_nodes.reset_cache()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_nodes.py
from unittest.mock import patch
import pytest
from orchestra import remote_nodes as rn
from orchestra.providers.remote_node import RemoteNodeProvider


def setup_function(_):
    rn.reset_cache()


def test_list_nodes_empty_when_unconfigured():
    with patch("orchestra.config.remote_nodes_config", return_value={}):
        assert rn.list_nodes() == {}


def test_list_nodes_parses_url_and_token():
    cfg = {"windows": {"url": "http://192.168.1.42:7070", "token": "secret"}}
    with patch("orchestra.config.remote_nodes_config", return_value=cfg):
        assert rn.list_nodes() == {"windows": {"url": "http://192.168.1.42:7070", "token": "secret"}}


def test_list_nodes_skips_entry_without_url():
    cfg = {"bad": {"token": "x"}, "windows": {"url": "http://x:7070"}}
    with patch("orchestra.config.remote_nodes_config", return_value=cfg):
        assert list(rn.list_nodes()) == ["windows"]


def test_get_node_returns_cached_provider():
    cfg = {"windows": {"url": "http://192.168.1.42:7070", "token": "secret"}}
    with patch("orchestra.config.remote_nodes_config", return_value=cfg):
        p = rn.get_node("windows")
        assert isinstance(p, RemoteNodeProvider)
        assert p.model_label("cdx-deep") == "remote/windows/cdx-deep"
        assert rn.get_node("windows") is p


def test_get_node_unknown_raises_listing_known():
    cfg = {"windows": {"url": "http://x:7070"}}
    with patch("orchestra.config.remote_nodes_config", return_value=cfg):
        with pytest.raises(ValueError) as exc:
            rn.get_node("nope")
        assert "windows" in str(exc.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_remote_nodes.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`No module named 'orchestra.remote_nodes'`).

- [ ] **Step 3: Add the config accessor**

In `orchestra/config.py`, after `ollama_backends_config()`:

```python
def remote_nodes_config() -> dict:
    """Raw [remote_nodes] table; empty dict when not configured."""
    return dict(_cfg.get("remote_nodes", {}))
```

- [ ] **Step 4: Create the registry**

```python
# orchestra/remote_nodes.py
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_remote_nodes.py -q --no-cov -p no:cacheprovider`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add orchestra/config.py orchestra/remote_nodes.py tests/test_remote_nodes.py
git commit -m "feat(remote): node registry + [remote_nodes] config accessor"
```

---

### Task 3: `alias@node` resolution (remote-first)

**Files:**
- Modify: `orchestra/providers/fallback.py` (the `_resolve_backend_ref`/`_is_backend_ref` fast-path from the Ollama feature)
- Test: `tests/test_fallback_node_ref.py`

**Interfaces:**
- Consumes: `remote_nodes.get_node()` (Task 2), existing `ollama_backends.get_backend()`.
- Produces: `resolve()`/`resolve_with_fallback()` for `left@X` return `(remote_nodes.get_node(X), left)` when `X` is a remote node, else fall back to the existing Ollama-backend behavior. New helper `_resolve_at_ref(alias) -> tuple`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fallback_node_ref.py
from unittest.mock import patch
import pytest
from orchestra.providers import fallback
from orchestra import remote_nodes as rn
from orchestra.providers import ollama_backends as ob
from orchestra.providers.remote_node import RemoteNodeProvider


def test_alias_at_remote_node_routes_to_remote_provider():
    rn.reset_cache()
    node_cfg = {"windows": {"url": "http://192.168.1.42:7070", "token": "s"}}
    with patch("orchestra.config.remote_nodes_config", return_value=node_cfg):
        provider, alias = fallback.resolve("cdx-deep@windows")
        assert isinstance(provider, RemoteNodeProvider)
        assert alias == "cdx-deep"
        assert provider.model_label(alias) == "remote/windows/cdx-deep"


def test_remote_node_wins_over_ollama_backend_same_name():
    rn.reset_cache(); ob.reset_cache()
    node_cfg = {"windows": {"url": "http://192.168.1.42:7070"}}
    backend_cfg = {"windows": {"base_url": "http://192.168.1.42:11434"}}
    with patch("orchestra.config.remote_nodes_config", return_value=node_cfg), \
         patch("orchestra.config.ollama_backends_config", return_value=backend_cfg):
        provider, alias = fallback.resolve("cdx-deep@windows")
        assert isinstance(provider, RemoteNodeProvider)  # remote wins


def test_at_backend_still_routes_to_ollama_when_not_a_node():
    rn.reset_cache(); ob.reset_cache()
    backend_cfg = {"windows-gpu": {"base_url": "http://192.168.1.42:11434"}}
    with patch("orchestra.config.remote_nodes_config", return_value={}), \
         patch("orchestra.config.ollama_backends_config", return_value=backend_cfg):
        from orchestra.providers.ollama import OllamaProvider
        provider, model = fallback.resolve("qwen2.5:7b@windows-gpu")
        assert isinstance(provider, OllamaProvider)
        assert model == "qwen2.5:7b"


def test_no_at_is_backward_compatible():
    provider, effort = fallback.resolve("oll-fast")
    assert provider is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_fallback_node_ref.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`cdx-deep@windows` currently routes to the Ollama backend `get_backend("windows")` → ValueError, since remote nodes aren't checked yet).

- [ ] **Step 3: Extend the `@` fast-path to check remote nodes first**

In `orchestra/providers/fallback.py`, replace the existing `_resolve_backend_ref` helper with a remote-first version (keep `_is_backend_ref`):

```python
def _resolve_at_ref(alias: str) -> tuple:
    """'left@X' -> remote node (preferred) or Ollama backend provider."""
    left, _, target = alias.partition("@")
    from orchestra import remote_nodes
    if target in remote_nodes.list_nodes():
        return remote_nodes.get_node(target), left
    from orchestra.providers import ollama_backends
    return ollama_backends.get_backend(target), left
```

Then update the fast-path call sites in `resolve()` and `resolve_with_fallback()` (they currently call `_resolve_backend_ref(alias)`) to call `_resolve_at_ref(alias)`. If `_resolve_backend_ref` is now unused, remove it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_fallback_node_ref.py /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_fallback_backend_ref.py -q --no-cov -p no:cacheprovider`
Expected: PASS (both the new node tests and the existing backend tests — the Ollama path is preserved).

- [ ] **Step 5: Commit**

```bash
git add orchestra/providers/fallback.py tests/test_fallback_node_ref.py
git commit -m "feat(fallback): route alias@node to remote nodes (remote-first over Ollama backends)"
```

---

### Task 4: gateway server — `orchestra/gateway.py`

**Files:**
- Create: `orchestra/gateway.py`
- Test: `tests/test_gateway.py`

**Interfaces:**
- Produces: `gateway.make_app(token: str = "") -> aiohttp.web.Application` with routes `POST /run`, `GET /aliases`, `GET /health`; bearer-auth enforced on `/run` and `/aliases` when a token is set; `gateway.serve(host: str, port: int, token: str) -> None` (refuses `0.0.0.0` without a token).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gateway.py
import pytest
from unittest.mock import patch, MagicMock
from aiohttp.test_utils import TestClient, TestServer
from orchestra import gateway


class _FakeRun:
    def __init__(self):
        agent = MagicMock()
        agent.stdout_log = "remote output"
        agent.model = "gpt/5.4"
        agent.estimated_cost_usd = 0.01
        self.agents = [agent]
        self.status = MagicMock(value="completed")


@pytest.mark.asyncio
async def test_run_requires_token_when_configured(aiohttp_client):
    client = await aiohttp_client(gateway.make_app(token="secret"))
    resp = await client.post("/run", json={"alias": "cdx-deep", "task": "x"})
    assert resp.status == 401


@pytest.mark.asyncio
async def test_run_executes_and_returns_stdout(aiohttp_client):
    client = await aiohttp_client(gateway.make_app(token="secret"))
    with patch("orchestra.gateway.run_ask", return_value=_FakeRun()) as mock_ask:
        resp = await client.post(
            "/run", json={"alias": "cdx-deep", "task": "do it"},
            headers={"Authorization": "Bearer secret"},
        )
    assert resp.status == 200
    body = await resp.json()
    assert body["stdout"] == "remote output"
    assert body["status"] == "completed"
    mock_ask.assert_called_once_with("cdx-deep", "do it")


@pytest.mark.asyncio
async def test_health_needs_no_auth(aiohttp_client):
    client = await aiohttp_client(gateway.make_app(token="secret"))
    resp = await client.get("/health")
    assert resp.status == 200
    body = await resp.json()
    assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_aliases_lists_local(aiohttp_client):
    client = await aiohttp_client(gateway.make_app(token=""))
    with patch("orchestra.gateway.available_aliases", return_value=["✓ cdx-deep"]):
        resp = await client.get("/aliases")
    assert resp.status == 200
    body = await resp.json()
    assert "aliases" in body


def test_serve_refuses_public_bind_without_token():
    with pytest.raises(ValueError):
        gateway.serve(host="0.0.0.0", port=7070, token="")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_gateway.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`No module named 'orchestra.gateway'`).

- [ ] **Step 3: Create the gateway**

```python
# orchestra/gateway.py
"""Agent gateway: expose this node's run_ask over HTTP so a remote Orchestra can use it.

Small aiohttp server (see federation_http.py for the pattern). Wraps run_ask; adds nothing.
"""
from __future__ import annotations

import json
import logging
import socket

from aiohttp import web

from orchestra.engine.runners import run_ask
from orchestra.providers.fallback import available_aliases

logger = logging.getLogger(__name__)


def _authorized(request: web.Request, token: str) -> bool:
    if not token:
        return True
    header = request.headers.get("Authorization", "")
    parts = header.split()
    return len(parts) == 2 and parts[0].lower() == "bearer" and parts[1] == token


async def _health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok", "service": "orchestra-gateway", "host": socket.gethostname(),
    })


def make_app(token: str = "") -> web.Application:
    app = web.Application()

    async def _run(request: web.Request) -> web.Response:
        if not _authorized(request, token):
            return web.json_response({"error": "Unauthorized"}, status=401)
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        alias, task = payload.get("alias"), payload.get("task")
        if not alias or not task:
            return web.json_response({"error": "alias and task are required"}, status=400)
        try:
            run = run_ask(alias, task)
        except Exception as exc:  # surface, don't swallow
            logger.warning("gateway run_ask failed: %s", exc)
            return web.json_response({"error": str(exc)}, status=500)
        agent = run.agents[0] if run.agents else None
        return web.json_response({
            "stdout": agent.stdout_log if agent else "",
            "model": getattr(agent, "model", "") if agent else "",
            "cost_usd": getattr(agent, "estimated_cost_usd", 0.0) if agent else 0.0,
            "status": getattr(run.status, "value", str(run.status)),
        })

    async def _aliases(request: web.Request) -> web.Response:
        if not _authorized(request, token):
            return web.json_response({"error": "Unauthorized"}, status=401)
        return web.json_response({"aliases": available_aliases()})

    app.router.add_post("/run", _run)
    app.router.add_get("/aliases", _aliases)
    app.router.add_get("/health", _health)
    return app


def serve(host: str, port: int, token: str) -> None:
    if host not in ("127.0.0.1", "localhost") and not token:
        raise ValueError(
            "Refusing to bind a public host without a token — the gateway can run agents "
            "(and thus code) on this machine. Set a token: orchestra serve --token <secret>."
        )
    logger.info("orchestra gateway on %s:%s (token=%s)", host, port, "set" if token else "NONE")
    web.run_app(make_app(token), host=host, port=port)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_gateway.py -q --no-cov -p no:cacheprovider`
Expected: PASS (5 tests). (If `aiohttp_client` fixture is missing, the repo's `pytest-aiohttp`/`aiohttp` test utils provide it; the existing federation tests use the same fixture — confirm and mirror.)

- [ ] **Step 5: Commit**

```bash
git add orchestra/gateway.py tests/test_gateway.py
git commit -m "feat(gateway): orchestra serve — HTTP agent gateway (run/aliases/health, bearer auth)"
```

---

### Task 5: node discovery/scan (parallel, token-filtered, auto-add)

**Files:**
- Modify: `orchestra/remote_nodes.py`
- Test: `tests/test_remote_nodes.py` (extend)

**Interfaces:**
- Consumes: `list_nodes()` (Task 2).
- Produces: `remote_nodes.scan_subnet(subnet_prefix: str, token: str = "", port: int = 7070) -> list[dict]` — parallel `GET http://{prefix}.{i}:{port}/health` for i in 1..254 (ThreadPoolExecutor); each responder yields `{"ip":..,"host":..,"port":..}`; non-responders skipped. `remote_nodes.default_subnet_prefix() -> str` — the first three octets of this machine's primary LAN IP (e.g. `192.168.1`).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_remote_nodes.py
import json
from unittest.mock import patch


class _HealthResp:
    def __init__(self, host):
        self.status = 200
        self._b = json.dumps({"status": "ok", "host": host}).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return self._b


def test_scan_subnet_finds_only_responders():
    def _fake_urlopen(req, timeout=None):
        if "192.168.1.42" in req.full_url:
            return _HealthResp("winbox")
        raise OSError("no route")

    with patch("orchestra.remote_nodes.urllib.request.urlopen", _fake_urlopen):
        found = rn.scan_subnet("192.168.1", token="s", port=7070)

    ips = [f["ip"] for f in found]
    assert "192.168.1.42" in ips
    assert len(found) == 1
    assert found[0]["host"] == "winbox"


def test_default_subnet_prefix_shape():
    with patch("orchestra.remote_nodes._primary_ip", return_value="192.168.1.7"):
        assert rn.default_subnet_prefix() == "192.168.1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_remote_nodes.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`scan_subnet` / `default_subnet_prefix` not defined).

- [ ] **Step 3: Add scanning**

Add to `orchestra/remote_nodes.py` (top: `import json`, `import socket`, `import urllib.request`, `from concurrent.futures import ThreadPoolExecutor`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_remote_nodes.py -q --no-cov -p no:cacheprovider`
Expected: PASS (7 tests total).

- [ ] **Step 5: Commit**

```bash
git add orchestra/remote_nodes.py tests/test_remote_nodes.py
git commit -m "feat(remote): parallel token-filtered LAN scan for gateways"
```

---

### Task 6: CLI — `serve`, `nodes`, `nodes scan`

**Files:**
- Modify: `orchestra/cli.py` (add near the `backends` command from the Ollama feature)
- Test: `tests/test_cli_nodes.py`

**Interfaces:**
- Consumes: `gateway.serve` (Task 4), `remote_nodes.list_nodes`/`get_node`/`scan_subnet`/`default_subnet_prefix` (Tasks 2, 5), `service.config_set_value(path, value, persist=True)` (existing — the function `orchestra config set` calls; writes `.orchestra/config.toml` + `config.reload_config()`).
- Produces: `orchestra serve [--host] [--port 7070] [--token]`; `orchestra nodes` (list + reachability); `orchestra nodes scan [--subnet] [--token] [--port] [--yes]` (auto-adds found nodes to `[remote_nodes.*]`).

⚠️ cli.py historically had duplicated command groups — grep for existing `serve`/`nodes` and add each command ONCE. `nodes` and `nodes scan` should be a Typer sub-app (like other grouped commands) OR two flat commands `nodes` and `nodes-scan` — match whatever grouping style cli.py already uses (check how `config`/`config set` is wired) and keep it consistent.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_nodes.py
from unittest.mock import patch
from typer.testing import CliRunner
from orchestra.cli import app

runner = CliRunner()


def test_nodes_lists_configured():
    cfg = {"windows": {"url": "http://192.168.1.42:7070", "token": "s"}}
    with patch("orchestra.config.remote_nodes_config", return_value=cfg), \
         patch("orchestra.remote_nodes.get_node") as mock_get:
        mock_get.return_value.is_available.return_value = True
        result = runner.invoke(app, ["nodes"])
    assert result.exit_code == 0
    assert "windows" in result.output


def test_nodes_scan_auto_adds(tmp_path):
    found = [{"ip": "192.168.1.42", "host": "winbox", "port": 7070}]
    written = {}
    with patch("orchestra.remote_nodes.default_subnet_prefix", return_value="192.168.1"), \
         patch("orchestra.remote_nodes.scan_subnet", return_value=found), \
         patch("orchestra.cli._set_node_config", side_effect=lambda n, u, t: written.update({n: (u, t)})):
        result = runner.invoke(app, ["nodes-scan", "--token", "s"])
    assert result.exit_code == 0
    assert "winbox" in written  # auto-added by hostname
    assert written["winbox"][0] == "http://192.168.1.42:7070"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_cli_nodes.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`No such command 'nodes'`).

- [ ] **Step 3: Add the commands**

In `orchestra/cli.py`, near the `backends` command, add (adjust the config-write helper to the repo's existing `config set` mechanism — see how `config set` writes `.orchestra/config.toml`; reuse that here):

```python
def _set_node_config(name: str, url: str, token: str) -> None:
    """Persist a remote node to [remote_nodes.<name>] via the existing config-set path."""
    from orchestra.service import config_set_value
    config_set_value(f"remote_nodes.{name}.url", url)
    if token:
        config_set_value(f"remote_nodes.{name}.token", token)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host (0.0.0.0 needs --token)"),
    port: int = typer.Option(7070, "--port"),
    token: str = typer.Option("", "--token", help="Shared secret; required for non-local host"),
):
    """Run this machine as an Orchestra agent gateway for remote nodes."""
    from orchestra import gateway
    try:
        gateway.serve(host=host, port=port, token=token)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


@app.command()
def nodes():
    """List configured remote nodes and their reachability."""
    from orchestra import remote_nodes
    listing = remote_nodes.list_nodes()
    if not listing:
        console.print("[dim]No remote nodes configured. Add one with 'orchestra nodes-scan'.[/dim]")
        return
    for name, info in listing.items():
        try:
            up = remote_nodes.get_node(name).is_available()
        except Exception:
            up = False
        status = "[green]✓ reachable[/green]" if up else "[red]✗ unreachable[/red]"
        console.print(f"  {name:<14} {info['url']:<32} {status}")


@app.command("nodes-scan")
def nodes_scan(
    subnet: str = typer.Option("", "--subnet", help="e.g. 192.168.1 (default: auto-detect)"),
    token: str = typer.Option("", "--token", help="Shared secret used by the gateways"),
    port: int = typer.Option(7070, "--port"),
):
    """Scan the LAN for Orchestra gateways and auto-add them as remote nodes."""
    from orchestra import remote_nodes
    prefix = subnet or remote_nodes.default_subnet_prefix()
    console.print(f"[dim]Scanning {prefix}.1-254:{port} …[/dim]")
    found = remote_nodes.scan_subnet(prefix, token=token, port=port)
    if not found:
        console.print("[yellow]No gateways found. Is 'orchestra serve --token …' running and the token correct?[/yellow]")
        return
    for f in found:
        url = f"http://{f['ip']}:{f['port']}"
        _set_node_config(f["host"], url, token)
        console.print(f"  [green]added[/green] {f['host']:<14} {url}")
```

Note: `service.config_set_value(path, value, persist=True)` (service.py:659) is the exact function `orchestra config set` uses — it writes `.orchestra/config.toml` and calls `config.reload_config()`. Do not hand-parse the TOML file. The `nodes-scan` test monkeypatches `orchestra.cli._set_node_config`, so the real write is not exercised in the unit test.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_cli_nodes.py -q --no-cov -p no:cacheprovider`
Expected: PASS (2 tests).

- [ ] **Step 5: Verify command-registry (no duplicate commands)**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_cli_command_registry.py -q --no-cov -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add orchestra/cli.py tests/test_cli_nodes.py
git commit -m "feat(cli): serve / nodes / nodes-scan commands for remote node gateway"
```

---

### Task 7: docs + smoke

**Files:**
- Modify: `CLAUDE.md` (commands + a remote-nodes config example), `docs/architecture.md` (a Providers/entry note).
- No new test (doc only) + a local smoke.

- [ ] **Step 1: Add commands + config example to `CLAUDE.md`**

In the `## Setup & Commands` block:

```
orchestra serve --token <secret>        # run THIS machine as an agent gateway for others
orchestra nodes-scan --token <secret>   # find LAN gateways and auto-add them
orchestra ask cdx-deep@windows "task"   # run a remote node's agent (its logged-in CLI account)
```

In `## Conventions`, after the multi-backend Ollama note:

```
- Remote nodes: run `orchestra serve --token X` on the machine whose agents you want to share;
  address them as `alias@node`. Config:
  [remote_nodes.windows]
  url = "http://192.168.1.42:7070"
  token = "X"
  Security: the token grants remote code execution on the serving machine — trusted networks only;
  `--host 0.0.0.0` is refused without a token.
```

- [ ] **Step 2: Add an architecture note**

In `docs/architecture.md`, extend the Providers row (or add a line under Entry points) noting
`gateway.py` (agent gateway) and `remote_node.py`/`remote_nodes.py` (`alias@node` remote execution).

- [ ] **Step 3: Local smoke (loopback)**

Run a gateway on localhost and hit it end-to-end:
```
# terminal A
.venv/bin/python -m orchestra serve --host 127.0.0.1 --port 7070 --token smoketest
# terminal B (or after backgrounding A)
.venv/bin/python -m orchestra config set remote_nodes.local.url http://127.0.0.1:7070
.venv/bin/python -m orchestra config set remote_nodes.local.token smoketest
.venv/bin/python -m orchestra nodes            # local → ✓ reachable
.venv/bin/python -m orchestra ask oll-fast@local "say hi in 3 words"
```
Expected: `nodes` shows `local ✓ reachable`; the `ask` returns the local Ollama's output routed through the gateway. No traceback.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/architecture.md
git commit -m "docs(remote): document the agent gateway, alias@node, and nodes-scan"
```

---

## Verification (end-to-end)

1. On the remote machine: `orchestra serve --host 0.0.0.0 --token <secret>` (with `OLLAMA_HOST=0.0.0.0` if sharing Ollama too).
2. On this machine: `orchestra nodes-scan --token <secret>` → the remote is auto-added by hostname.
3. `orchestra ask cdx-deep@<host> "..."` runs the remote's logged-in Codex; `oll-fast@<host>` runs its Ollama. Output + status appear locally.
4. `orchestra ask oll-fast` (no `@`) still works (backward compatibility); existing `@backend` Ollama refs still work.
5. Full fast suite from a scratch dir shows no new regressions:
   `cd /tmp && /abs/.venv/bin/pytest /abs/tests -m "not slow" -q --no-cov -p no:cacheprovider --continue-on-collection-errors`.

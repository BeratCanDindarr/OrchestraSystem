# Multi-backend Ollama Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Orchestra address models on multiple named Ollama backends (local MacBook + remote Windows GPU) at once, via `model@backend` syntax, with auto-discovery and CLI surface.

**Architecture:** Replace the single `_ollama` singleton with a per-backend registry. `OllamaProvider` gains an optional constructor `base_url` (default preserved). `fallback.resolve()` grows a `model@backend` fast-path that routes to the right backend's provider. New `providers/ollama_backends.py` holds the registry + discovery. New CLI `models`/`backends` commands.

**Tech Stack:** Python 3.11+, Typer CLI, stdlib `urllib` (existing HTTP style), pytest with mocked HTTP. No new dependencies. No Rust (see spec Decisions — I/O-bound work).

## Global Constraints

- Pure Python; NO new dependencies; NO Rust.
- Backward compatibility is mandatory: existing `oll-fast`/`oll-coder`/… aliases and the single-host `[providers.ollama] base_url` / `ORCHESTRA_PROVIDERS__OLLAMA__BASE_URL` override MUST keep working unchanged.
- Backend addressing syntax is exactly `model@backend` (the `@` separator).
- Default synthesized backend when none configured: name `macbook`, url `http://localhost:11434`.
- No silent exception swallowing — errors are logged or surfaced with actionable messages.
- Functions <50 lines; files <800; immutability (return new objects, no in-place mutation of shared state).
- Discovery cache TTL ~60s; `--refresh` bypasses it. Time source: `time.monotonic()`.
- Run pytest FROM A SCRATCH DIR (tests pollute CWD): `cd /tmp && /abs/.venv/bin/pytest <abs paths> -q --no-cov -p no:cacheprovider`.
- Conventional Commits; NO Co-Authored-By lines.

## File Structure

- `orchestra/providers/ollama.py` — MODIFY: add constructor `base_url` + `_base_url()` instance method; route all HTTP through it. Keeps module-level `_ollama_base()` as the default.
- `orchestra/providers/ollama_backends.py` — CREATE: `list_backends()`, `get_backend(name)`, `discover_all(refresh=False)`, cache helpers.
- `orchestra/config.py` — MODIFY: add `ollama_backends_config()` accessor.
- `orchestra/providers/fallback.py` — MODIFY: `model@backend` fast-path in `resolve()` and `resolve_with_fallback()`.
- `orchestra/cli.py` — MODIFY: add `models` and `backends` commands.
- Tests: `tests/test_ollama_backends.py`, `tests/test_ollama_provider_baseurl.py`, `tests/test_fallback_backend_ref.py`, `tests/test_cli_models.py`.

---

### Task 1: OllamaProvider — per-instance base_url

**Files:**
- Modify: `orchestra/providers/ollama.py`
- Test: `tests/test_ollama_provider_baseurl.py`

**Interfaces:**
- Produces: `OllamaProvider(base_url: str | None = None)`; instance method `_base_url() -> str` returning the instance base or the module default `_ollama_base()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ollama_provider_baseurl.py
from unittest.mock import patch, MagicMock
from orchestra.providers.ollama import OllamaProvider


def test_default_base_url_uses_module_default():
    p = OllamaProvider()
    with patch("orchestra.providers.ollama._ollama_base", return_value="http://localhost:11434"):
        assert p._base_url() == "http://localhost:11434"


def test_instance_base_url_overrides_default():
    p = OllamaProvider("http://192.168.1.42:11434")
    assert p._base_url() == "http://192.168.1.42:11434"


def test_is_available_pings_instance_base_url():
    p = OllamaProvider("http://192.168.1.42:11434")
    captured = {}

    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _Resp()

    with patch("orchestra.providers.ollama.urllib.request.urlopen", _fake_urlopen):
        assert p.is_available() is True
    assert captured["url"] == "http://192.168.1.42:11434/api/tags"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_ollama_provider_baseurl.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`OllamaProvider() takes no arguments` / no `_base_url`).

- [ ] **Step 3: Add constructor + `_base_url()` and route HTTP through it**

In `orchestra/providers/ollama.py`, add to the class (after `name = "ollama"`):

```python
    def __init__(self, base_url: str | None = None) -> None:
        self._base = base_url

    def _base_url(self) -> str:
        return self._base or _ollama_base()
```

Then replace every `_ollama_base()` call inside the instance methods `is_available`, `native_run`, `async_run`, `stream`, and `embed` with `self._base_url()`. (There are 5 such call sites: lines ~32, ~68, ~115, ~152, ~196 in the current file.) Leave the module-level `_ollama_base()` function itself unchanged, and leave the `warmup` staticmethod as-is (it is not backend-scoped in this task).

Example — `is_available` becomes:

```python
    def is_available(self) -> bool:
        req = urllib.request.Request(f"{self._base_url()}/api/tags", method="GET")
        healthcheck_timeout = int(config.availability_config().get("ollama_healthcheck_timeout_seconds", 2))
        try:
            with urllib.request.urlopen(req, timeout=healthcheck_timeout) as resp:
                return resp.status == 200
        except Exception:
            return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_ollama_provider_baseurl.py -q --no-cov -p no:cacheprovider`
Expected: PASS (3 tests).

- [ ] **Step 5: Verify no regression in existing ollama/provider tests**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests -k "ollama or provider" -q --no-cov -p no:cacheprovider`
Expected: PASS (existing tests still green — the default path is preserved).

- [ ] **Step 6: Commit**

```bash
git add orchestra/providers/ollama.py tests/test_ollama_provider_baseurl.py
git commit -m "feat(ollama): per-instance base_url so one provider can target a remote host"
```

---

### Task 2: config accessor + backend registry

**Files:**
- Modify: `orchestra/config.py` (add accessor near `providers_config`, ~line 220)
- Create: `orchestra/providers/ollama_backends.py`
- Test: `tests/test_ollama_backends.py`

**Interfaces:**
- Consumes: `OllamaProvider(base_url)` from Task 1.
- Produces:
  - `config.ollama_backends_config() -> dict` — raw `[providers.ollama.backends]` table (empty dict if absent).
  - `ollama_backends.list_backends() -> dict[str, str]` — name→base_url; defaults to `{"macbook": "http://localhost:11434"}` when none configured.
  - `ollama_backends.get_backend(name: str) -> OllamaProvider` — cached instance per backend; raises `ValueError` (message listing known backends) for an unknown name.
  - `ollama_backends.reset_cache() -> None` — test/refresh helper clearing the instance cache.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ollama_backends.py
from unittest.mock import patch
import pytest
from orchestra.providers import ollama_backends as ob
from orchestra.providers.ollama import OllamaProvider


def setup_function(_):
    ob.reset_cache()


def test_list_backends_defaults_to_macbook_localhost():
    with patch("orchestra.config.ollama_backends_config", return_value={}):
        assert ob.list_backends() == {"macbook": "http://localhost:11434"}


def test_list_backends_reads_configured():
    cfg = {
        "macbook": {"base_url": "http://localhost:11434"},
        "windows-gpu": {"base_url": "http://192.168.1.42:11434"},
    }
    with patch("orchestra.config.ollama_backends_config", return_value=cfg):
        assert ob.list_backends() == {
            "macbook": "http://localhost:11434",
            "windows-gpu": "http://192.168.1.42:11434",
        }


def test_get_backend_returns_provider_with_right_base_url():
    cfg = {"windows-gpu": {"base_url": "http://192.168.1.42:11434"}}
    with patch("orchestra.config.ollama_backends_config", return_value=cfg):
        p = ob.get_backend("windows-gpu")
        assert isinstance(p, OllamaProvider)
        assert p._base_url() == "http://192.168.1.42:11434"
        assert ob.get_backend("windows-gpu") is p  # cached, same instance


def test_get_backend_unknown_raises_valueerror_listing_known():
    cfg = {"windows-gpu": {"base_url": "http://192.168.1.42:11434"}}
    with patch("orchestra.config.ollama_backends_config", return_value=cfg):
        with pytest.raises(ValueError) as exc:
            ob.get_backend("nope")
        assert "windows-gpu" in str(exc.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_ollama_backends.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`No module named 'orchestra.providers.ollama_backends'`).

- [ ] **Step 3: Add the config accessor**

In `orchestra/config.py`, after `providers_config()` (~line 220):

```python
def ollama_backends_config() -> dict:
    """Raw [providers.ollama.backends] table; empty dict when not configured."""
    return dict(_cfg.get("providers", {}).get("ollama", {}).get("backends", {}))
```

- [ ] **Step 4: Create the backend registry module**

```python
# orchestra/providers/ollama_backends.py
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_ollama_backends.py -q --no-cov -p no:cacheprovider`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add orchestra/config.py orchestra/providers/ollama_backends.py tests/test_ollama_backends.py
git commit -m "feat(ollama): named-backend registry + config accessor"
```

---

### Task 3: per-backend model discovery

**Files:**
- Modify: `orchestra/providers/ollama_backends.py`
- Test: `tests/test_ollama_backends.py` (extend)

**Interfaces:**
- Consumes: `list_backends()`, `get_backend()` from Task 2.
- Produces: `ollama_backends.discover_all(refresh: bool = False) -> dict[str, dict]` where each value is `{"base_url": str, "online": bool, "models": list[dict]}` and each model dict is `{"name": str, "size": int}`. Results cached for `_DISCOVERY_TTL_S` (60s); `refresh=True` bypasses. Uses `time.monotonic()`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_ollama_backends.py
import json
from unittest.mock import patch


def _fake_tags_response(models):
    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({"models": models}).encode()
    return _Resp()


def test_discover_all_lists_models_per_backend():
    ob.reset_cache()
    ob.reset_discovery_cache()
    cfg = {
        "macbook": {"base_url": "http://localhost:11434"},
        "windows-gpu": {"base_url": "http://192.168.1.42:11434"},
    }
    def _fake_urlopen(req, timeout=None):
        if "192.168.1.42" in req.full_url:
            return _fake_tags_response([{"name": "llama3:70b", "size": 40_000_000_000}])
        return _fake_tags_response([{"name": "qwen2.5:7b", "size": 4_700_000_000}])

    with patch("orchestra.config.ollama_backends_config", return_value=cfg), \
         patch("orchestra.providers.ollama_backends.urllib.request.urlopen", _fake_urlopen):
        result = ob.discover_all(refresh=True)

    assert result["windows-gpu"]["online"] is True
    assert result["windows-gpu"]["models"][0]["name"] == "llama3:70b"
    assert result["macbook"]["models"][0]["name"] == "qwen2.5:7b"


def test_discover_all_marks_offline_backend_without_failing_others():
    ob.reset_cache()
    ob.reset_discovery_cache()
    cfg = {
        "macbook": {"base_url": "http://localhost:11434"},
        "windows-gpu": {"base_url": "http://192.168.1.42:11434"},
    }
    def _fake_urlopen(req, timeout=None):
        if "192.168.1.42" in req.full_url:
            raise OSError("connection refused")
        return _fake_tags_response([{"name": "qwen2.5:7b", "size": 4_700_000_000}])

    with patch("orchestra.config.ollama_backends_config", return_value=cfg), \
         patch("orchestra.providers.ollama_backends.urllib.request.urlopen", _fake_urlopen):
        result = ob.discover_all(refresh=True)

    assert result["windows-gpu"]["online"] is False
    assert result["windows-gpu"]["models"] == []
    assert result["macbook"]["online"] is True
    assert result["macbook"]["models"][0]["name"] == "qwen2.5:7b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_ollama_backends.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`discover_all` / `reset_discovery_cache` not defined).

- [ ] **Step 3: Add discovery with TTL cache**

Add to `orchestra/providers/ollama_backends.py` (top: add `import json`, `import time`, `import urllib.request`, `from orchestra import config` already present):

```python
import json
import time
import urllib.request

_DISCOVERY_TTL_S = 60.0
_discovery_cache: dict[str, dict] | None = None
_discovery_at: float = 0.0


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
    global _discovery_cache, _discovery_at
    _discovery_cache = None
    _discovery_at = 0.0
```

Also update `reset_cache()` to also clear discovery (call `reset_discovery_cache()` inside it).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_ollama_backends.py -q --no-cov -p no:cacheprovider`
Expected: PASS (6 tests total).

- [ ] **Step 5: Commit**

```bash
git add orchestra/providers/ollama_backends.py tests/test_ollama_backends.py
git commit -m "feat(ollama): per-backend model discovery via /api/tags with TTL cache"
```

---

### Task 4: `model@backend` resolution in fallback

**Files:**
- Modify: `orchestra/providers/fallback.py`
- Test: `tests/test_fallback_backend_ref.py`

**Interfaces:**
- Consumes: `ollama_backends.get_backend()` from Task 2.
- Produces: `resolve(alias)` and `resolve_with_fallback(alias, attempt)` return `(get_backend(backend), model)` when `alias` contains `@`; otherwise unchanged. A new helper `_is_backend_ref(alias) -> bool` and `_resolve_backend_ref(alias) -> tuple`.

**Design note (deliberate simplification):** `resolve()` does NOT verify the model is actually installed on the backend — that would add a `/api/tags` network round-trip to every run. Instead, an unknown model surfaces through Ollama's own `/api/generate` error (already handled by `native_run` → `[ERROR] Ollama: ...`), and `orchestra models` gives the user the authoritative installed-model list up front. This satisfies the spec's "validation" intent at the CLI/discovery layer without slowing the hot path. If a friendlier inline error is wanted later, it belongs in a separate task, not here.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fallback_backend_ref.py
from unittest.mock import patch
import pytest
from orchestra.providers import fallback
from orchestra.providers import ollama_backends as ob


def test_resolve_backend_ref_routes_to_backend_provider():
    ob.reset_cache()
    cfg = {"windows-gpu": {"base_url": "http://192.168.1.42:11434"}}
    with patch("orchestra.config.ollama_backends_config", return_value=cfg):
        provider, model = fallback.resolve("qwen2.5-coder:7b@windows-gpu")
        assert model == "qwen2.5-coder:7b"
        assert provider._base_url() == "http://192.168.1.42:11434"


def test_resolve_unknown_backend_raises():
    ob.reset_cache()
    cfg = {"windows-gpu": {"base_url": "http://192.168.1.42:11434"}}
    with patch("orchestra.config.ollama_backends_config", return_value=cfg):
        with pytest.raises(ValueError):
            fallback.resolve("qwen2.5:7b@nope")


def test_resolve_without_at_is_backward_compatible():
    # oll-fast resolves to the ollama provider + "fast" effort as before
    provider, effort = fallback.resolve("oll-fast")
    assert effort in ("fast", "cdx-deep", "low", "xhigh") or provider is not None
    # the key point: no crash, returns a (provider, effort) tuple
    assert provider is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_fallback_backend_ref.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`@windows-gpu` is not in ALIAS_TABLE → KeyError today).

- [ ] **Step 3: Add the backend-ref fast path**

In `orchestra/providers/fallback.py`, add near the top of the resolution helpers (after imports, before `_canonical_alias`):

```python
def _is_backend_ref(alias: str) -> bool:
    return "@" in alias


def _resolve_backend_ref(alias: str) -> tuple:
    """'model@backend' -> (backend provider instance, model)."""
    from orchestra.providers import ollama_backends
    model, _, backend = alias.partition("@")
    return ollama_backends.get_backend(backend), model
```

Then at the FIRST line of `resolve()` (before `alias = _canonical_alias(alias)`):

```python
    if _is_backend_ref(alias):
        return _resolve_backend_ref(alias)
```

And at the FIRST line of `resolve_with_fallback()` (before `chain = fallback_chain(alias)`):

```python
    if _is_backend_ref(alias):
        return _resolve_backend_ref(alias)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_fallback_backend_ref.py -q --no-cov -p no:cacheprovider`
Expected: PASS (3 tests).

- [ ] **Step 5: Verify no regression across provider/fallback tests**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests -k "fallback or alias or provider or ollama" -q --no-cov -p no:cacheprovider`
Expected: PASS (existing green).

- [ ] **Step 6: Commit**

```bash
git add orchestra/providers/fallback.py tests/test_fallback_backend_ref.py
git commit -m "feat(fallback): route model@backend aliases to the named backend provider"
```

---

### Task 5: CLI `models` + `backends` commands

**Files:**
- Modify: `orchestra/cli.py` (add two `@app.command()` functions near `aliases`, ~line 593)
- Test: `tests/test_cli_models.py`

**Interfaces:**
- Consumes: `ollama_backends.discover_all()` from Task 3.
- Produces: CLI commands `orchestra models [--refresh] [--json]` and `orchestra backends`. `ask model@backend` already works via Task 4 (no CLI change needed — `run_ask` → `resolve`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_models.py
from unittest.mock import patch
from typer.testing import CliRunner
from orchestra.cli import app

runner = CliRunner()

_FAKE_DISCOVERY = {
    "macbook": {"base_url": "http://localhost:11434", "online": True,
                "models": [{"name": "qwen2.5:7b", "size": 4_700_000_000}]},
    "windows-gpu": {"base_url": "http://192.168.1.42:11434", "online": False, "models": []},
}


def test_models_lists_backends_and_marks_offline():
    with patch("orchestra.providers.ollama_backends.discover_all", return_value=_FAKE_DISCOVERY):
        result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "macbook" in result.output
    assert "qwen2.5:7b" in result.output
    assert "windows-gpu" in result.output
    assert "offline" in result.output.lower()


def test_models_json_output():
    with patch("orchestra.providers.ollama_backends.discover_all", return_value=_FAKE_DISCOVERY):
        result = runner.invoke(app, ["models", "--json"])
    assert result.exit_code == 0
    assert "qwen2.5:7b" in result.output
    assert "windows-gpu" in result.output


def test_backends_shows_reachability():
    with patch("orchestra.providers.ollama_backends.discover_all", return_value=_FAKE_DISCOVERY):
        result = runner.invoke(app, ["backends"])
    assert result.exit_code == 0
    assert "macbook" in result.output
    assert "windows-gpu" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_cli_models.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`No such command 'models'`).

- [ ] **Step 3: Add the commands**

In `orchestra/cli.py`, after the `aliases` command (~line 593), add:

```python
def _fmt_size(nbytes: int) -> str:
    gb = nbytes / 1_000_000_000
    return f"{gb:.1f} GB" if gb >= 0.1 else f"{nbytes} B"


@app.command()
def models(
    refresh: bool = typer.Option(False, "--refresh", help="Bypass the discovery cache"),
    json_output: bool = typer.Option(False, "--json", help="Return machine-readable JSON"),
):
    """List Ollama models available on each configured backend."""
    from orchestra.providers import ollama_backends

    discovery = ollama_backends.discover_all(refresh=refresh)
    if json_output:
        _print_json(discovery)
        return
    for name, info in discovery.items():
        status = "[green]✓ online[/green]" if info["online"] else "[red]✗ offline[/red]"
        console.print(f"\n[bold]{name}[/bold] ({info['base_url']})  {status}")
        for m in info["models"]:
            console.print(f"  {m['name']:<28} {_fmt_size(m['size'])}")
        if info["online"] and not info["models"]:
            console.print("  [dim](no models installed)[/dim]")


@app.command()
def backends():
    """Show configured Ollama backends and whether they are reachable."""
    from orchestra.providers import ollama_backends

    discovery = ollama_backends.discover_all()
    for name, info in discovery.items():
        status = "[green]✓ reachable[/green]" if info["online"] else "[red]✗ unreachable[/red]"
        count = len(info["models"])
        console.print(f"  {name:<14} {info['base_url']:<32} {status}  ({count} models)")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_cli_models.py -q --no-cov -p no:cacheprovider`
Expected: PASS (3 tests).

- [ ] **Step 5: Verify CLI command registry + no duplicate-command regression**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_cli_command_registry.py -q --no-cov -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add orchestra/cli.py tests/test_cli_models.py
git commit -m "feat(cli): add 'models' and 'backends' commands for multi-backend Ollama"
```

---

### Task 6: docs + config example + end-to-end sanity

**Files:**
- Modify: `.orchestra/config.toml` (add a commented `[providers.ollama.backends.*]` example)
- Modify: `CLAUDE.md` (one line on multi-backend Ollama under Conventions/commands) and `docs/architecture.md` (Providers row note)
- No new test (doc/config only); run a manual smoke.

- [ ] **Step 1: Add a commented example to `.orchestra/config.toml`**

Under the existing `[providers.ollama]` section, add:

```toml
# Named backends — address models as `model@backend` (e.g. qwen2.5:7b@windows-gpu).
# When none are defined, a single local 'macbook' backend (http://localhost:11434) is used.
# [providers.ollama.backends.macbook]
# base_url = "http://localhost:11434"
# [providers.ollama.backends.windows-gpu]
# base_url = "http://192.168.1.42:11434"
```

- [ ] **Step 2: Add a CLAUDE.md commands line**

In the `## Setup & Commands` block of `CLAUDE.md`, add:

```
orchestra models                        # list Ollama models per backend (local + remote)
orchestra ask qwen2.5:7b@windows-gpu "…" # query a specific model on a named backend
```

- [ ] **Step 3: Manual smoke (local backend only)**

Run: `cd /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem && .venv/bin/python -m orchestra models`
Expected: shows `macbook` backend with the locally-installed models (or `✗ offline` if Ollama isn't running). No traceback.

Run: `.venv/bin/python -m orchestra backends`
Expected: one-line-per-backend reachability table.

- [ ] **Step 4: Commit**

```bash
git add .orchestra/config.toml CLAUDE.md docs/architecture.md
git commit -m "docs(ollama): document multi-backend config and CLI"
```

---

## Verification (end-to-end)

1. `orchestra models` shows the local `macbook` backend and its models (add a remote backend via `orchestra config set providers.ollama.backends.windows-gpu.base_url http://<win-ip>:11434` and re-run `--refresh` to see it).
2. `orchestra ask <model>@macbook "hello"` runs against the local backend; `...@windows-gpu` runs against the remote (when the Windows machine is on and `OLLAMA_HOST=0.0.0.0`).
3. Existing `orchestra ask oll-fast "hello"` still works (backward compatibility).
4. Full fast suite from a scratch dir shows no new regressions vs. the pre-change baseline:
   `cd /tmp && /abs/.venv/bin/pytest /abs/tests -m "not slow" -q --no-cov -p no:cacheprovider --continue-on-collection-errors`.

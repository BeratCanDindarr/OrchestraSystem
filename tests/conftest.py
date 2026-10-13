"""Shared pytest configuration."""
import pytest

from orchestra import config

# Disable idempotency caching for tests by default to avoid cross-test pollution.
config.set("idempotency", "enabled", False)


# ---------------------------------------------------------------------------
# Path isolation: redirect all Orchestra filesystem paths to temp dirs so the
# suite never writes artefacts (DBs, runs/, chat_sessions/, live_status.json,
# checkpoints/) into the repo root or the real ~/.orchestra directory.
#
# We use a session-scoped autouse fixture.  Because pytest's built-in
# `monkeypatch` fixture is function-scoped, we instantiate MonkeyPatch()
# directly and call mp.undo() in the finaliser — this is the documented way
# to get session-scoped monkeypatching.
#
# Tests that already pass their own explicit paths (tmp_path, tempfile, …) are
# unaffected: we only replace the DEFAULT return values of the config accessor
# functions; callers that supply an explicit path bypass these accessors.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _orchestra_config_isolation():
    """Neutralize user config sections that change agent routing.

    The repo's .orchestra/config.toml is real user configuration (personas,
    tiers, aliases, local_first). Tests assert factory defaults, so these
    sections are removed for the session and restored afterwards.
    """
    from orchestra import config as _config

    keys = ("personas", "tiers", "tier_routing", "aliases", "local_first")
    backup = {k: _config._cfg.pop(k) for k in keys if k in _config._cfg}
    # Point the config loader at a nonexistent file so any reload_config()
    # during the suite cannot resurrect the user's real config.
    real_path = _config._config_path
    _config._config_path = real_path.with_name("config.toml.tests-nonexistent")
    yield
    _config._config_path = real_path
    _config._cfg.update(backup)


@pytest.fixture(scope="session", autouse=True)
def _orchestra_path_isolation(tmp_path_factory):
    """Redirect Orchestra path accessors to session-scoped temp directories.

    Accessors redirected
    --------------------
    - config.orchestra_root()     → <tmp>/orchestra_root/
    - config.artifact_root()      → <tmp>/orchestra_root/runs/
    - config.db_path()            → <tmp>/orchestra_root/orchestra.db
    - config.memory_db_path()     → <tmp>/orchestra_root/memory.db
    - config.eventlog_db_path()   → <tmp>/orchestra_root/events.db
    - config.cache_db_path()      → <tmp>/orchestra_root/fts_cache.db
    - config.checkpoint_dir()     → <tmp>/orchestra_root/checkpoints/

    chat.store.sessions_dir() and live_status._status_path() both call
    config.orchestra_root() internally, so they are covered automatically.
    """
    import pytest

    root = tmp_path_factory.mktemp("orchestra_root")
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    mp = pytest.MonkeyPatch()

    mp.setattr("orchestra.config.orchestra_root", lambda: root)
    mp.setattr("orchestra.config.artifact_root", lambda: runs_dir)
    mp.setattr("orchestra.config.db_path", lambda: root / "orchestra.db")
    mp.setattr("orchestra.config.memory_db_path", lambda: root / "memory.db")
    mp.setattr("orchestra.config.eventlog_db_path", lambda: root / "events.db")
    mp.setattr("orchestra.config.cache_db_path", lambda: root / "fts_cache.db")
    mp.setattr("orchestra.config.fts_cache_db_path", lambda: str(root / "fts_cache.db"))
    mp.setattr("orchestra.config.checkpoint_dir", lambda: checkpoints_dir)

    yield

    mp.undo()


@pytest.fixture(autouse=True)
def _no_real_agent_cli(monkeypatch):
    """A test must never start a real agent CLI: it costs money and its answer is not a fixture.

    Tests that need the process runner itself take the `run_agent_command` fixture.
    """
    from orchestra.providers import base

    def refuse(cmd, timeout, cwd=None, **hooks):
        raise AssertionError(f"a test reached a real agent CLI: {cmd[:2]}")

    monkeypatch.setattr(base, "run_command", refuse)

    # Claude, Codex and Gemini start their CLI themselves on the async path.
    import asyncio
    import os

    real_exec = asyncio.create_subprocess_exec

    async def guarded_exec(program, *args, **kwargs):
        if os.path.basename(str(program)) in _AGENT_CLIS:
            raise AssertionError(f"a test reached a real agent CLI: {program} {list(args)[:1]}")
        return await real_exec(program, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", guarded_exec)


@pytest.fixture(autouse=True)
def _no_real_ollama_model(request, monkeypatch):
    """A test must never ask a live Ollama model (some of them are paid cloud models).

    Tests of the Ollama provider itself stub its HTTP layer and are left alone.
    """
    names = (request.module.__name__ + request.node.nodeid).lower()
    if "ollama" in names or "native" in names:
        return
    from orchestra.providers.ollama import OllamaProvider

    def refuse(self, *args, **kwargs):
        raise AssertionError("a test reached a real Ollama model")

    async def refuse_async(self, *args, **kwargs):
        raise AssertionError("a test reached a real Ollama model")

    monkeypatch.setattr(OllamaProvider, "native_run", refuse)
    monkeypatch.setattr(OllamaProvider, "async_run", refuse_async)
    monkeypatch.setattr(OllamaProvider, "native_messages", refuse)


@pytest.fixture(autouse=True)
def _no_real_embedding_endpoint(monkeypatch):
    """A test must never ask a live Ollama for an embedding: the suite would depend on one.

    The store sees what it sees when Ollama is down (no vector); tests of the store patch
    `orchestra.storage.memory._ollama.embed` with their own vectors.
    """
    from orchestra.storage import memory

    monkeypatch.setattr(memory._ollama, "embed", lambda *args, **kwargs: [])


@pytest.fixture(autouse=True)
def _no_span_export(request, monkeypatch):
    """A test must never send spans to a collector: none is listening, and the exporter
    retries in the background until the suite ends.

    A test of the exporter itself takes the `real_otlp_exporter` fixture.
    """
    from orchestra.engine import tracer

    if "real_otlp_exporter" not in request.fixturenames:
        monkeypatch.setattr(tracer, "otlp_exporter", lambda *args, **kwargs: None)


@pytest.fixture
def real_otlp_exporter():
    """Opt out of `_no_span_export` (the exporter is built, nothing is sent)."""
    from orchestra.engine import tracer

    return tracer.otlp_exporter


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Deterministic vectors for tests that need the memory store to accept rows."""
    import zlib

    import numpy as np

    from orchestra.storage import memory

    def embed(text, *args, **kwargs):
        vector = np.random.default_rng(zlib.crc32(str(text).encode())).standard_normal(384)
        return (vector / np.linalg.norm(vector)).astype(np.float32).tolist()

    monkeypatch.setattr(memory._ollama, "embed", embed)


@pytest.fixture
def run_agent_command():
    """The real process runner, for tests of the runner itself (they start only Python)."""
    from orchestra.providers import base

    return _REAL_RUN_COMMAND


from orchestra.providers import base as _providers_base  # noqa: E402

_REAL_RUN_COMMAND = _providers_base.run_command
_AGENT_CLIS = frozenset({"claude", "codex", "kimi", "gemini", "ollama"})

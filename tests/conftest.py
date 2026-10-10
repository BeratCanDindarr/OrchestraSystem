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

    def refuse(cmd, timeout, cwd=None):
        raise AssertionError(f"a test reached a real agent CLI: {cmd[:2]}")

    monkeypatch.setattr(base, "run_command", refuse)


@pytest.fixture
def run_agent_command():
    """The real process runner, for tests of the runner itself (they start only Python)."""
    from orchestra.providers import base

    return _REAL_RUN_COMMAND


from orchestra.providers import base as _providers_base  # noqa: E402

_REAL_RUN_COMMAND = _providers_base.run_command

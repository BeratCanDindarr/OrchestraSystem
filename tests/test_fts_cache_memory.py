from __future__ import annotations
"""Regression tests for audit finding C-3: in-memory FtsSearchCache must not touch filesystem.

These tests are written to FAIL before the fix and PASS after.
"""

import pytest

import orchestra.engine.runners.common as common_module
from orchestra.storage.fts_cache import FtsSearchCache


class TestInMemoryCreatesNoFile:
    """FtsSearchCache(db_path=':memory:') must never create files in CWD."""

    def test_in_memory_creates_no_file(self, tmp_path, monkeypatch):
        """Constructing with ':memory:' sentinel must not create any file in CWD."""
        monkeypatch.chdir(tmp_path)

        cache = FtsSearchCache(db_path=":memory:")

        # Basic round-trip: store then lookup to confirm the cache is functional
        entry_id = cache.store("run-mem-1", "hello world test", "a response")
        assert isinstance(entry_id, str) and len(entry_id) == 36

        result = cache.lookup("hello world test", "")
        assert result is not None and result.response == "a response"

        cache.close()

        # No file whose name starts with ':memory:' should exist in the CWD
        leaked = [p for p in tmp_path.iterdir() if p.name.startswith(":memory:")]
        assert leaked == [], f"Leaked files: {leaked}"


class TestFallbackOnConfigFailureNoFile:
    """_get_fts_cache() fallback to :memory: must not create filesystem artefacts."""

    def test_fallback_on_config_failure_no_file(self, tmp_path, monkeypatch):
        """When config.fts_cache_db_path() raises OSError, fallback to :memory: creates no file."""
        monkeypatch.chdir(tmp_path)

        # Force config to fail
        monkeypatch.setattr(
            "orchestra.config.fts_cache_db_path",
            lambda: (_ for _ in ()).throw(OSError("simulated config failure")),
        )

        # Reset the cached singleton so _get_fts_cache() re-initialises
        original_cache = common_module._fts_cache
        common_module._fts_cache = None

        try:
            cache = common_module._get_fts_cache()

            # Returned cache must be usable
            assert cache is not None
            entry_id = cache.store("run-fallback-1", "fallback test prompt", "response")
            assert isinstance(entry_id, str) and len(entry_id) == 36
        finally:
            # Restore singleton so we don't pollute other tests
            common_module._fts_cache = original_cache

        # No ':memory:' files should have been created
        leaked = [p for p in tmp_path.iterdir() if p.name.startswith(":memory:")]
        assert leaked == [], f"Leaked files: {leaked}"

"""Comprehensive tests for Orchestra configuration system.

Tests all 8 configuration sections: EventLog, Checkpoint, Idempotency, PiiScrubber,
FTS5Cache, OtelTracer, WorkspaceGuard, Performance, and Retry policies.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestra import config


class TestEventlogConfiguration:
    """Tests for EventLog block configuration."""


    def test_eventlog_busy_timeout_ms_default(self):
        """Test default busy timeout is 5000ms."""
        assert config.eventlog_busy_timeout_ms() == 5000

    def test_eventlog_cache_size_kb_default(self):
        """Test default cache size is 64MB."""
        assert config.eventlog_cache_size_kb() == 64000


    @patch.dict(os.environ, {"ORCHESTRA_EVENTLOG__BUSY_TIMEOUT_MS": "10000"})
    def test_eventlog_env_override(self):
        """Test environment variable override for busy_timeout_ms."""
        config.reload_config()
        try:
            assert config.eventlog_busy_timeout_ms() == 10000
        finally:
            config.reload_config()


class TestCheckpointConfiguration:
    """Tests for state suspension checkpoint configuration."""

    def test_checkpoint_ttl_hours_default(self):
        """Test default checkpoint TTL is 48 hours."""
        assert config.checkpoint_ttl_hours() == 48


    @patch.dict(os.environ, {"ORCHESTRA_CHECKPOINT__TTL_HOURS": "72"})
    def test_checkpoint_ttl_env_override(self):
        """Test environment variable override for checkpoint TTL."""
        config.reload_config()
        try:
            assert config.checkpoint_ttl_hours() == 72
        finally:
            config.reload_config()


class TestScrubberConfiguration:
    """Tests for PII scrubber configuration."""


    def test_scrubber_entropy_threshold_default(self):
        """Test default entropy threshold is 5.0."""
        assert config.scrubber_entropy_threshold() == 5.0


    @patch.dict(os.environ, {"ORCHESTRA_SCRUBBER__ENTROPY_THRESHOLD": "6.5"})
    def test_scrubber_entropy_env_override(self):
        """Test environment variable override for entropy threshold."""
        config.reload_config()
        try:
            assert config.scrubber_entropy_threshold() == 6.5
        finally:
            config.reload_config()


class TestCacheConfiguration:
    """Tests for FTS5Cache configuration."""

    def test_cache_db_path_default(self):
        """Test default FTS5 cache database path."""
        path = config.cache_db_path()
        assert isinstance(path, Path)
        assert path.name == "fts_cache.db"

    def test_cache_ttl_hours_default(self):
        """Test default cache TTL is 168 hours (7 days)."""
        assert config.cache_ttl_hours() == 168


    def test_cache_size_kb_default(self):
        """Test default cache SQLite size is 64MB."""
        assert config.cache_size_kb() == 64000


class TestTracerConfiguration:
    """Tests for OpenTelemetry tracer configuration."""

    def test_tracer_enabled_default(self):
        """Test tracer is enabled by default."""
        assert config.tracer_enabled() is True


    def test_tracer_jaeger_enabled_default(self):
        """Test Jaeger exporter is disabled by default (requires explicit enable)."""
        assert config.tracer_jaeger_enabled() is False

    def test_tracer_batch_size_default(self):
        """Test default batch size is 512."""
        assert config.tracer_batch_size() == 512

    def test_tracer_batch_timeout_ms_default(self):
        """Test default batch timeout is 5000ms."""
        assert config.tracer_batch_timeout_ms() == 5000


class TestWorkspaceGuardConfiguration:
    """Tests for workspace guard configuration."""

    def test_workspace_guard_enabled_default(self):
        """Test workspace guard is enabled by default."""
        assert config.workspace_guard_enabled() is True

    def test_workspace_guard_critical_files_default(self):
        """Test default critical files list."""
        files = config.workspace_guard_critical_files()
        assert isinstance(files, list)
        assert "orchestra/engine/runner.py" in files
        assert "orchestra/config.py" in files


    @patch.dict(os.environ, {"ORCHESTRA_WORKSPACE_GUARD__ENABLED": "false"})
    def test_workspace_guard_disabled_env_override(self):
        """Test environment variable can disable workspace guard."""
        config.reload_config()
        try:
            assert config.workspace_guard_enabled() is False
        finally:
            config.reload_config()


class TestPerformanceConfiguration:
    """Tests for performance tuning configuration."""

    def test_thread_pool_size_default(self):
        """Test default thread pool size is 10."""
        assert config.thread_pool_size() == 10

    def test_thread_pool_timeout_s_default(self):
        """Test default thread pool timeout is 300 seconds."""
        assert config.thread_pool_timeout_s() == 300


    @patch.dict(os.environ, {"ORCHESTRA_PERFORMANCE__THREAD_POOL_SIZE": "20"})
    def test_thread_pool_size_env_override(self):
        """Test environment variable override for thread pool size."""
        config.reload_config()
        try:
            assert config.thread_pool_size() == 20
        finally:
            config.reload_config()


class TestRetryConfiguration:
    """Tests for retry policy configuration."""

    def test_retry_max_attempts_default(self):
        """Test default max retry attempts is 3."""
        assert config.retry_max_attempts() == 3

    def test_retry_backoff_ms_default(self):
        """Test default backoff is 100ms."""
        assert config.retry_backoff_ms() == 100

    def test_retry_backoff_multiplier_default(self):
        """Test default backoff multiplier is 2.0."""
        assert config.retry_backoff_multiplier() == 2.0

    def test_retry_config_returns_dict(self):
        """Test retry_config returns all fields."""
        cfg = config.retry_config()
        assert isinstance(cfg, dict)
        assert "max_attempts" in cfg
        assert "backoff_ms" in cfg
        assert "backoff_multiplier" in cfg

    @patch.dict(os.environ, {
        "ORCHESTRA_RETRY__MAX_ATTEMPTS": "5",
        "ORCHESTRA_RETRY__BACKOFF_MS": "200",
    })
    def test_retry_env_override(self):
        """Test environment variable override for retry policy."""
        config.reload_config()
        try:
            assert config.retry_max_attempts() == 5
            assert config.retry_backoff_ms() == 200
        finally:
            config.reload_config()


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_validate_config_success(self):
        """Test validation passes with default config."""
        is_valid, errors = config.validate_config()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_eventlog_busy_timeout(self):
        """Test validation catches low busy_timeout_ms."""
        config.set("eventlog", "busy_timeout_ms", 50)
        try:
            is_valid, errors = config.validate_config()
            assert is_valid is False
            assert any("busy_timeout_ms" in e for e in errors)
        finally:
            config.reload_config()

    def test_validate_checkpoint_ttl(self):
        """Test validation catches non-positive checkpoint TTL."""
        config.set("checkpoint", "ttl_hours", 0)
        try:
            is_valid, errors = config.validate_config()
            assert is_valid is False
            assert any("checkpoint.ttl_hours" in e for e in errors)
        finally:
            config.reload_config()


    def test_validate_scrubber_entropy(self):
        """Test validation catches entropy threshold out of range."""
        config.set("scrubber", "entropy_threshold", 9.0)
        try:
            is_valid, errors = config.validate_config()
            assert is_valid is False
            assert any("entropy_threshold" in e for e in errors)
        finally:
            config.reload_config()


    def test_validate_tracer_batch_size(self):
        """Test validation catches zero batch_size."""
        config.set("tracer", "batch_size", 0)
        try:
            is_valid, errors = config.validate_config()
            assert is_valid is False
            assert any("batch_size" in e for e in errors)
        finally:
            config.reload_config()

    def test_validate_thread_pool_size(self):
        """Test validation catches zero thread_pool_size."""
        config.set("performance", "thread_pool_size", 0)
        try:
            is_valid, errors = config.validate_config()
            assert is_valid is False
            assert any("thread_pool_size" in e for e in errors)
        finally:
            config.reload_config()

    def test_validate_retry_backoff_multiplier(self):
        """Test validation catches backoff_multiplier <= 1."""
        config.set("retry", "backoff_multiplier", 1.0)
        try:
            is_valid, errors = config.validate_config()
            assert is_valid is False
            assert any("backoff_multiplier" in e for e in errors)
        finally:
            config.reload_config()


class TestDynamicSetOperation:
    """Tests for runtime configuration updates."""

    def test_set_creates_section(self):
        """Test set() creates section if it doesn't exist."""
        config.set("test_section", "test_key", "test_value")
        value = config.get("test_section", "test_key")
        assert value == "test_value"

    def test_set_overwrites_existing(self):
        """Test set() overwrites existing values."""
        config.set("eventlog", "busy_timeout_ms", 10000)
        assert config.eventlog_busy_timeout_ms() == 10000

    def test_set_changes_are_in_memory(self):
        """Test set() changes are not persisted to file."""
        config.set("test_section", "ephemeral_key", "ephemeral_value")
        # reload_config should clear the ephemeral value
        config.reload_config()
        value = config.get("test_section", "ephemeral_key")
        assert value is None


class TestEnvironmentVariableOverrides:
    """Tests for environment variable override behavior."""

    def test_nested_env_override_parsing(self):
        """Test ORCHESTRA_SECTION__KEY parsing."""
        with patch.dict(os.environ, {
            "ORCHESTRA_EVENTLOG__CACHE_SIZE_KB": "128000"
        }):
            config.reload_config()
            try:
                assert config.eventlog_cache_size_kb() == 128000
            finally:
                config.reload_config()

    def test_env_override_boolean_parsing(self):
        """Test environment variable boolean parsing."""
        with patch.dict(os.environ, {
            "ORCHESTRA_ORCHESTRA__AUTO_SYNTHESIZE": "true",
            "ORCHESTRA_FTS_CACHE__ENABLED": "false",
        }):
            config.reload_config()
            try:
                assert config.auto_synthesize() is True
                assert config.fts_cache_enabled() is False
            finally:
                config.reload_config()

    def test_env_override_float_parsing(self):
        """Test environment variable float parsing."""
        with patch.dict(os.environ, {
            "ORCHESTRA_SCRUBBER__ENTROPY_THRESHOLD": "6.5"
        }):
            config.reload_config()
            try:
                assert config.scrubber_entropy_threshold() == 6.5
            finally:
                config.reload_config()


class TestConfigIntegration:
    """Integration tests across multiple blocks."""


    def test_validation_with_all_blocks(self):
        """Test validation across all blocks."""
        is_valid, errors = config.validate_config()
        assert is_valid is True
        assert isinstance(errors, list)

    def test_config_reload_preserves_functionality(self):
        """Test reload_config doesn't break accessors."""
        before = config.eventlog_busy_timeout_ms()
        config.reload_config()
        after = config.eventlog_busy_timeout_ms()
        assert before == after


def test_the_users_config_is_under_the_projects(tmp_path, monkeypatch):
    # ~/.orchestra/config.toml holds what is the user's in every project (their plugins' MCP
    # servers, user_setup); the project's config wins key by key; file_config() stays the project's.
    user = tmp_path / "user.toml"
    user.write_text('[orchestra]\ntimeout = 111\nmax_agent_seconds = 5000\n[providers.claude]\nuser_setup = true\n')
    project = tmp_path / "project" / ".orchestra" / "config.toml"
    project.parent.mkdir(parents=True)
    project.write_text('[orchestra]\ntimeout = 222\n[providers.claude]\ntools = ["Read"]\n')
    monkeypatch.setenv("ORCHESTRA_USER_CONFIG", str(user))
    monkeypatch.setattr(config, "_config_path", project)
    try:
        config.reload_config()
        assert config.timeout() == 222 and config.max_agent_seconds() == 5000
        assert config.provider_config("claude") == {"user_setup": True, "tools": ["Read"]}
        assert "max_agent_seconds" not in config.file_config().get("orchestra", {})
        assert "user_config" not in config.raw_config()   # the path variable is not a setting
    finally:
        monkeypatch.undo()
        config.reload_config()


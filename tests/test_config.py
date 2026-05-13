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

    def test_eventlog_db_path_default(self):
        """Test default EventLog database path expansion."""
        path = config.eventlog_db_path()
        assert isinstance(path, Path)
        assert not str(path).startswith("~")
        assert path.name == "events.db"

    def test_eventlog_wal_mode_default(self):
        """Test WAL mode is enabled by default."""
        assert config.eventlog_wal_mode() is True

    def test_eventlog_busy_timeout_ms_default(self):
        """Test default busy timeout is 5000ms."""
        assert config.eventlog_busy_timeout_ms() == 5000

    def test_eventlog_cache_size_kb_default(self):
        """Test default cache size is 64MB."""
        assert config.eventlog_cache_size_kb() == 64000

    def test_eventlog_config_returns_dict(self):
        """Test eventlog_config returns complete configuration dict."""
        cfg = config.eventlog_config()
        assert isinstance(cfg, dict)
        assert "db_path" in cfg
        assert "wal_mode" in cfg
        assert "busy_timeout_ms" in cfg
        assert "cache_size_kb" in cfg

    @patch.dict(os.environ, {"ORCHESTRA_EVENTLOG__BUSY_TIMEOUT_MS": "10000"})
    def test_eventlog_env_override(self):
        """Test environment variable override for busy_timeout_ms."""
        config.reload_config()
        try:
            assert config.eventlog_busy_timeout_ms() == 10000
        finally:
            config.reload_config()

    @patch.dict(os.environ, {"ORCHESTRA_EVENTLOG__WAL_MODE": "false"})
    def test_eventlog_wal_mode_env_override(self):
        """Test environment variable override for WAL mode."""
        config.reload_config()
        try:
            assert config.eventlog_wal_mode() is False
        finally:
            config.reload_config()


class TestCheckpointConfiguration:
    """Tests for state suspension checkpoint configuration."""

    def test_checkpoint_ttl_hours_default(self):
        """Test default checkpoint TTL is 48 hours."""
        assert config.checkpoint_ttl_hours() == 48

    def test_checkpoint_dir_default(self):
        """Test checkpoint directory path expansion."""
        path = config.checkpoint_dir()
        assert isinstance(path, Path)
        assert not str(path).startswith("~")
        assert path.name == "checkpoints"

    def test_checkpoint_config_returns_dict(self):
        """Test checkpoint_config returns complete configuration."""
        cfg = config.checkpoint_config()
        assert isinstance(cfg, dict)
        assert "ttl_hours" in cfg
        assert "dir" in cfg

    @patch.dict(os.environ, {"ORCHESTRA_CHECKPOINT__TTL_HOURS": "72"})
    def test_checkpoint_ttl_env_override(self):
        """Test environment variable override for checkpoint TTL."""
        config.reload_config()
        try:
            assert config.checkpoint_ttl_hours() == 72
        finally:
            config.reload_config()


class TestIdempotencyConfiguration:
    """Tests for idempotency caching configuration."""

    def test_idempotency_ttl_hours_default(self):
        """Test default idempotency TTL is 48 hours."""
        assert config.idempotency_ttl_hours() == 48

    def test_idempotency_max_cache_size_default(self):
        """Test default idempotency cache size is 10000 entries."""
        assert config.idempotency_max_cache_size() == 10000

    def test_idempotency_config_returns_dict(self):
        """Test idempotency_config returns complete configuration."""
        cfg = config.idempotency_config()
        assert isinstance(cfg, dict)
        assert "ttl_hours" in cfg
        assert "max_cache_size" in cfg

    @patch.dict(os.environ, {"ORCHESTRA_IDEMPOTENCY__MAX_CACHE_SIZE": "50000"})
    def test_idempotency_cache_size_env_override(self):
        """Test environment variable override for cache size."""
        config.reload_config()
        try:
            assert config.idempotency_max_cache_size() == 50000
        finally:
            config.reload_config()


class TestScrubberConfiguration:
    """Tests for PII scrubber configuration."""

    def test_scrubber_enabled_default(self):
        """Test scrubber is enabled by default."""
        assert config.scrubber_enabled() is True

    def test_scrubber_entropy_threshold_default(self):
        """Test default entropy threshold is 5.0."""
        assert config.scrubber_entropy_threshold() == 5.0

    def test_scrubber_whitelist_patterns_default(self):
        """Test whitelist patterns include Base64 and hex defaults."""
        patterns = config.scrubber_whitelist_patterns()
        assert isinstance(patterns, dict)
        assert "base64_image" in patterns
        assert "hash_hex" in patterns

    def test_scrubber_scrub_on_cache_default(self):
        """Test scrubbing is enabled on cache by default."""
        assert config.scrubber_scrub_on_cache() is True

    def test_scrubber_scrub_on_log_default(self):
        """Test scrubbing is enabled on logging by default."""
        assert config.scrubber_scrub_on_log() is True

    def test_scrubber_extended_config_returns_dict(self):
        """Test scrubber_extended_config returns all fields."""
        cfg = config.scrubber_extended_config()
        assert isinstance(cfg, dict)
        assert "enabled" in cfg
        assert "entropy_threshold" in cfg
        assert "whitelist_patterns" in cfg
        assert "scrub_on_cache" in cfg
        assert "scrub_on_log" in cfg

    @patch.dict(os.environ, {"ORCHESTRA_SCRUBBER__ENTROPY_THRESHOLD": "6.5"})
    def test_scrubber_entropy_env_override(self):
        """Test environment variable override for entropy threshold."""
        config.reload_config()
        try:
            assert config.scrubber_entropy_threshold() == 6.5
        finally:
            config.reload_config()

    @patch.dict(os.environ, {"ORCHESTRA_SCRUBBER__ENABLED": "false"})
    def test_scrubber_disabled_env_override(self):
        """Test environment variable can disable scrubber."""
        config.reload_config()
        try:
            assert config.scrubber_enabled() is False
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

    def test_cache_bm25_threshold_default(self):
        """Test default BM25 threshold is -0.8."""
        assert config.cache_bm25_threshold() == -0.8

    def test_cache_bm25_strict_threshold_default(self):
        """Test default strict BM25 threshold is -2.0."""
        assert config.cache_bm25_strict_threshold() == -2.0

    def test_cache_similarity_min_default(self):
        """Test default similarity min is 0.85."""
        assert config.cache_similarity_min() == 0.85

    def test_cache_topk_default(self):
        """Test default top-K is 3."""
        assert config.cache_topk() == 3

    def test_cache_size_kb_default(self):
        """Test default cache SQLite size is 64MB."""
        assert config.cache_size_kb() == 64000

    def test_cache_mmap_size_mb_default(self):
        """Test default mmap size is 30MB."""
        assert config.cache_mmap_size_mb() == 30

    def test_cache_config_returns_dict(self):
        """Test cache_config returns all fields."""
        cfg = config.cache_config()
        assert isinstance(cfg, dict)
        assert "db_path" in cfg
        assert "ttl_hours" in cfg
        assert "bm25_threshold" in cfg
        assert "bm25_strict_threshold" in cfg
        assert "similarity_min" in cfg
        assert "topk" in cfg
        assert "cache_size_kb" in cfg
        assert "mmap_size_mb" in cfg

    @patch.dict(os.environ, {"ORCHESTRA_CACHE__SIMILARITY_MIN": "0.9"})
    def test_cache_similarity_env_override(self):
        """Test environment variable override for similarity threshold."""
        config.reload_config()
        try:
            assert config.cache_similarity_min() == 0.9
        finally:
            config.reload_config()

    @patch.dict(os.environ, {"ORCHESTRA_CACHE__TOPK": "5"})
    def test_cache_topk_env_override(self):
        """Test environment variable override for topk."""
        config.reload_config()
        try:
            assert config.cache_topk() == 5
        finally:
            config.reload_config()


class TestTracerConfiguration:
    """Tests for OpenTelemetry tracer configuration."""

    def test_tracer_enabled_default(self):
        """Test tracer is enabled by default."""
        assert config.tracer_enabled() is True

    def test_tracer_jaeger_host_default(self):
        """Test default Jaeger host is localhost."""
        assert config.tracer_jaeger_host() == "localhost"

    def test_tracer_jaeger_port_default(self):
        """Test default Jaeger port is 6831."""
        assert config.tracer_jaeger_port() == 6831

    def test_tracer_jaeger_enabled_default(self):
        """Test Jaeger exporter is disabled by default (requires explicit enable)."""
        assert config.tracer_jaeger_enabled() is False

    def test_tracer_batch_size_default(self):
        """Test default batch size is 512."""
        assert config.tracer_batch_size() == 512

    def test_tracer_batch_timeout_ms_default(self):
        """Test default batch timeout is 5000ms."""
        assert config.tracer_batch_timeout_ms() == 5000

    def test_tracer_config_returns_dict(self):
        """Test tracer_config returns all fields."""
        cfg = config.tracer_config()
        assert isinstance(cfg, dict)
        assert "enabled" in cfg
        assert "jaeger_host" in cfg
        assert "jaeger_port" in cfg
        assert "jaeger_enabled" in cfg
        assert "batch_size" in cfg
        assert "batch_timeout_ms" in cfg

    @patch.dict(os.environ, {
        "ORCHESTRA_TRACER__JAEGER_HOST": "jaeger.example.com",
        "ORCHESTRA_TRACER__JAEGER_PORT": "6832",
    })
    def test_tracer_jaeger_env_override(self):
        """Test environment variable override for Jaeger host/port."""
        config.reload_config()
        try:
            assert config.tracer_jaeger_host() == "jaeger.example.com"
            assert config.tracer_jaeger_port() == 6832
        finally:
            config.reload_config()


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

    def test_workspace_guard_config_returns_dict(self):
        """Test workspace_guard_config returns all fields."""
        cfg = config.workspace_guard_config()
        assert isinstance(cfg, dict)
        assert "enabled" in cfg
        assert "critical_files" in cfg

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

    def test_performance_config_returns_dict(self):
        """Test performance_config returns all fields."""
        cfg = config.performance_config()
        assert isinstance(cfg, dict)
        assert "thread_pool_size" in cfg
        assert "thread_pool_timeout_s" in cfg

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

    def test_validate_idempotency_ttl(self):
        """Test validation catches non-positive idempotency TTL."""
        config.set("idempotency", "ttl_hours", -1)
        try:
            is_valid, errors = config.validate_config()
            assert is_valid is False
            assert any("idempotency.ttl_hours" in e for e in errors)
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

    def test_validate_cache_similarity_min(self):
        """Test validation catches invalid similarity_min."""
        config.set("cache", "similarity_min", 1.5)
        try:
            is_valid, errors = config.validate_config()
            assert is_valid is False
            assert any("similarity_min" in e for e in errors)
        finally:
            config.reload_config()

    def test_validate_cache_topk(self):
        """Test validation catches zero topk."""
        config.set("cache", "topk", 0)
        try:
            is_valid, errors = config.validate_config()
            assert is_valid is False
            assert any("topk" in e for e in errors)
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
            "ORCHESTRA_EVENTLOG__WAL_MODE": "true",
            "ORCHESTRA_SCRUBBER__ENABLED": "false",
        }):
            config.reload_config()
            try:
                assert config.eventlog_wal_mode() is True
                assert config.scrubber_enabled() is False
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

    def test_all_blocks_accessible(self):
        """Test all configuration blocks are accessible."""
        blocks = [
            config.eventlog_config(),
            config.checkpoint_config(),
            config.idempotency_config(),
            config.scrubber_extended_config(),
            config.cache_config(),
            config.tracer_config(),
            config.workspace_guard_config(),
            config.performance_config(),
            config.retry_config(),
        ]
        assert all(isinstance(b, dict) for b in blocks)
        assert sum(len(b) for b in blocks) > 0

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

"""Load and expose Orchestra configuration."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore


_cfg: dict = {}
_file_cfg: dict = {}
_config_error: str | None = None


def _find_config() -> Path:
    """Walk up from CWD to find .orchestra/config.toml."""
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        candidate = parent / ".orchestra" / "config.toml"
        if candidate.exists():
            return candidate
    # Fallback: same dir as this file's package root
    return Path(__file__).parent.parent / ".orchestra" / "config.toml"


_config_path = _find_config()

def _load_config() -> tuple[dict, str | None]:
    if not _config_path.exists():
        return {}, None
    try:
        with open(_config_path, "rb") as _f:
            return tomllib.load(_f), None
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def _coerce_env_value(raw: str):
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        pass
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_coerce_env_value(part.strip()) for part in inner.split(",")]
    return raw


def _deep_merge(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _env_overrides() -> dict:
    root: dict = {}
    prefix = "ORCHESTRA_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix):].lower()
        parts = [part for part in path.split("__") if part]
        if not parts:
            continue
        cursor = root
        for part in parts[:-1]:
            next_cursor = cursor.get(part)
            if not isinstance(next_cursor, dict):
                next_cursor = {}
                cursor[part] = next_cursor
            cursor = next_cursor
        cursor[parts[-1]] = _coerce_env_value(value)
    return root


_file_cfg, _config_error = _load_config()
_cfg = _deep_merge(_file_cfg, _env_overrides())


# ── Public accessors ─────────────────────────────────────────────

def get(section: str, key: str, default=None):
    return section_data(section).get(key, default)


def section_data(name: str) -> dict:
    value = _cfg.get(name, {})
    return value if isinstance(value, dict) else {}


def section(name: str) -> dict:
    return section_data(name)


def raw_config() -> dict:
    return _cfg


def file_config() -> dict:
    return _file_cfg


def reload_config() -> None:
    global _cfg, _file_cfg, _config_error
    _file_cfg, _config_error = _load_config()
    _cfg = _deep_merge(_file_cfg, _env_overrides())


def timeout() -> int:
    return _cfg.get("orchestra", {}).get("timeout", 180)


def timeout_for_alias(alias: str | None = None) -> int:
    default_timeout = timeout()
    if not alias:
        return default_timeout

    explicit = _cfg.get("timeouts", {}).get(alias)
    if explicit is not None:
        return int(explicit)

    deep_defaults = {
        "cdx-deep": 900,
        "gmn-pro": 300,
        "cld-deep": 300,
    }
    fast_defaults = {
        "cdx-fast": 180,
        "gmn-fast": 180,
        "cld-fast": 180,
    }
    if alias in deep_defaults:
        return deep_defaults[alias]
    if alias in fast_defaults:
        return fast_defaults[alias]
    return default_timeout


def max_turns() -> int:
    return _cfg.get("limits", {}).get("max_turns", 10)


def max_budget_usd() -> float:
    return _cfg.get("limits", {}).get("max_budget_usd", 5.0)


def auto_synthesize() -> bool:
    return bool(_cfg.get("orchestra", {}).get("auto_synthesize", False))


def config_path() -> Path:
    return _config_path


def config_error() -> str | None:
    return _config_error


def fts_cache_db_path() -> str:
    """Return FTS5 cache database path (default: :memory: for in-process)."""
    return _cfg.get("fts_cache", {}).get("db_path", ":memory:")


def orchestra_root() -> Path:
    return _config_path.parent


def project_root() -> Path:
    return orchestra_root().parent


def artifact_root() -> Path:
    rel = _cfg.get("orchestra", {}).get("artifact_root", "runs")
    return (orchestra_root() / rel).resolve()


def db_path() -> Path:
    return (orchestra_root() / "orchestra.db").resolve()


def alias_map() -> dict[str, str]:
    return _cfg.get("aliases", {})


def resolve_alias(alias: str) -> str:
    """Return raw alias value, or the alias itself if not found."""
    return alias_map().get(alias, alias)


def provider_config(provider: str) -> dict:
    return _cfg.get("providers", {}).get(provider, {})


def price_per_1k_tokens(model_label: str) -> float:
    pricing = _cfg.get("pricing", {})
    if model_label in pricing:
        return float(pricing[model_label])

    defaults = {
        "gpt-5.4/low": 0.002,
        "gpt-5.4/medium": 0.004,
        "gpt-5.4/high": 0.008,
        "gpt-5.4/xhigh": 0.012,
        "gemini/flash-lite": 0.001,
        "gemini/flash": 0.002,
        "gemini/pro": 0.006,
    }
    return float(defaults.get(model_label, 0.004))


def redaction_patterns() -> list[str]:
    return _cfg.get("redaction", {}).get("patterns", [])


def routing_config() -> dict:
    return section("routing")


def review_config() -> dict:
    return section("review")


def synthesis_config() -> dict:
    return section("synthesis")


def retry_config() -> dict:
    return section("retry")


def validator_config() -> dict:
    return section("validator")


def execution_config() -> dict:
    return section("execution")


def availability_config() -> dict:
    providers = section("providers")
    availability = providers.get("availability", {}) if isinstance(providers, dict) else {}
    return availability if isinstance(availability, dict) else {}


def token_budget_config() -> dict:
    return section("token_budget")


def rate_limit_config() -> dict:
    return section("rate_limits")


def caching_config() -> dict:
    return section("caching")


def tier_routing_config() -> dict:
    return section("tier_routing")


def tier_config() -> dict:
    return section("tiers")


def cascade_config() -> dict:
    return section("cascade")


def approval_policies_config() -> dict:
    """Return approval policy configuration.

    Approval policies control when human approval gates are triggered during run execution.
    Policies can be enabled based on:
    - Cost threshold: triggers approval if run cost exceeds cost_threshold_usd
    - Confidence threshold: triggers approval if avg confidence < confidence_threshold
    - Task complexity: triggers approval if task contains complexity_keywords
    - Execution mode: approval required for specific modes (critical, planned)

    Returns:
        dict: Approval policy settings with keys:
            - enabled (bool): Master switch for approval gates
            - require_approval_on_high_cost (bool): Cost-based gate
            - cost_threshold_usd (float): Cost threshold in USD
            - require_approval_on_low_confidence (bool): Confidence-based gate
            - confidence_threshold (float): Confidence threshold (0-1)
            - require_approval_on_complex_task (bool): Complexity-based gate
            - complexity_keywords (list[str]): Keywords that trigger complexity gate
            - max_approval_wait_hours (int): Max time to wait for approval
            - approval_required_for_modes (list[str]): Modes that require approval
    """
    return section("approval_policies")


def scrubber_config() -> dict:
    """Return PII scrubber configuration.

    Scrubber settings control in-memory redaction of sensitive patterns
    (API keys, emails, JWTs, etc) before caching or logging.

    Returns:
        dict: Scrubber settings with keys:
            - enabled (bool): Master switch for scrubbing (default True)
            - entropy_threshold (float): Entropy threshold for filtering (default 5.0)
            - scrub_on_cache (bool): Scrub before caching (default True)
            - scrub_on_log (bool): Scrub before event logging (default True)
    """
    return section("scrubber")


def scrubber_entropy_threshold() -> float:
    """Get entropy threshold for PII scrubber false positive filtering."""
    return float(get("scrubber", "entropy_threshold", 5.0))


def scrubber_enabled() -> bool:
    """Check if PII scrubber is globally enabled."""
    return bool(get("scrubber", "enabled", True))


# ── EventLog Block Configuration ─────────────────────────────────────────────

def eventlog_db_path() -> Path:
    """Get EventLog database path, creating parent directories as needed."""
    path_str = get("eventlog", "db_path", "~/.orchestra/events.db")
    return Path(path_str).expanduser().resolve()


def eventlog_wal_mode() -> bool:
    """Check if EventLog uses SQLite WAL mode (default True for concurrency)."""
    return bool(get("eventlog", "wal_mode", True))


def eventlog_busy_timeout_ms() -> int:
    """Get EventLog PRAGMA busy_timeout value in milliseconds."""
    return int(get("eventlog", "busy_timeout_ms", 5000))


def eventlog_cache_size_kb() -> int:
    """Get EventLog PRAGMA cache_size in KB (-X means use X KB)."""
    return int(get("eventlog", "cache_size_kb", 64000))


def eventlog_config() -> dict:
    """Return complete EventLog configuration."""
    return {
        "db_path": eventlog_db_path(),
        "wal_mode": eventlog_wal_mode(),
        "busy_timeout_ms": eventlog_busy_timeout_ms(),
        "cache_size_kb": eventlog_cache_size_kb(),
    }


# ── State Suspension Configuration ───────────────────────────────────────────

def checkpoint_ttl_hours() -> int:
    """Get checkpoint TTL in hours before expiration."""
    return int(get("checkpoint", "ttl_hours", 48))


def checkpoint_dir() -> Path:
    """Get checkpoint directory for state suspension."""
    path_str = get("checkpoint", "dir", "~/.orchestra/checkpoints")
    return Path(path_str).expanduser().resolve()


def checkpoint_config() -> dict:
    """Return complete checkpoint configuration."""
    return {
        "ttl_hours": checkpoint_ttl_hours(),
        "dir": checkpoint_dir(),
    }


# ── Idempotency Configuration ────────────────────────────────────────────────

def idempotency_ttl_hours() -> int:
    """Get idempotency cache TTL in hours."""
    return int(get("idempotency", "ttl_hours", 48))


def idempotency_max_cache_size() -> int:
    """Get maximum idempotency cache entries before eviction."""
    return int(get("idempotency", "max_cache_size", 10000))


def idempotency_config() -> dict:
    """Return complete idempotency configuration."""
    return {
        "ttl_hours": idempotency_ttl_hours(),
        "max_cache_size": idempotency_max_cache_size(),
    }


# ── PiiScrubber Configuration ────────────────────────────────────────────────

def scrubber_entropy_threshold() -> float:
    """Get entropy threshold for PII scrubber (prevents false positives on hashes)."""
    return float(get("scrubber", "entropy_threshold", 5.0))


def scrubber_whitelist_patterns() -> dict:
    """Get PII scrubber whitelist patterns for known safe data."""
    default_whitelist = {
        "base64_image": ["iVB", "/9j", "GIF89a", "ffd8ff"],
        "hash_hex": r"^[a-f0-9]{40,}$",  # SHA-1 and SHA-256
    }
    return get("scrubber", "whitelist_patterns", default_whitelist)


def scrubber_scrub_on_cache() -> bool:
    """Check if scrubbing is enabled before caching responses."""
    return bool(get("scrubber", "scrub_on_cache", True))


def scrubber_scrub_on_log() -> bool:
    """Check if scrubbing is enabled before event logging."""
    return bool(get("scrubber", "scrub_on_log", True))


def scrubber_extended_config() -> dict:
    """Return complete PII scrubber configuration."""
    return {
        "enabled": scrubber_enabled(),
        "entropy_threshold": scrubber_entropy_threshold(),
        "whitelist_patterns": scrubber_whitelist_patterns(),
        "scrub_on_cache": scrubber_scrub_on_cache(),
        "scrub_on_log": scrubber_scrub_on_log(),
    }


# ── FTS5Cache Configuration ──────────────────────────────────────────────────

def cache_db_path() -> Path:
    """Get FTS5 cache database path."""
    path_str = get("cache", "db_path", "~/.orchestra/fts_cache.db")
    return Path(path_str).expanduser().resolve()


def cache_ttl_hours() -> int:
    """Get FTS5 cache entry TTL in hours (default 7 days)."""
    return int(get("cache", "ttl_hours", 168))


def cache_bm25_threshold() -> float:
    """Get BM25 similarity threshold for cache hits (more negative = stricter)."""
    return float(get("cache", "bm25_threshold", -0.8))


def cache_bm25_strict_threshold() -> float:
    """Get strict BM25 threshold for high-precision matching."""
    return float(get("cache", "bm25_strict_threshold", -2.0))


def cache_similarity_min() -> float:
    """Get minimum similarity (0-1 range) for semantic deduplication."""
    return float(get("cache", "similarity_min", 0.85))


def cache_topk() -> int:
    """Get top-K results to return from FTS5 semantic search."""
    return int(get("cache", "topk", 3))


def cache_size_kb() -> int:
    """Get FTS5 cache SQLite cache size in KB."""
    return int(get("cache", "cache_size_kb", 64000))


def cache_mmap_size_mb() -> int:
    """Get FTS5 memory-mapped I/O size in MB for performance."""
    return int(get("cache", "mmap_size_mb", 30))


def cache_config() -> dict:
    """Return complete FTS5 cache configuration."""
    return {
        "db_path": cache_db_path(),
        "ttl_hours": cache_ttl_hours(),
        "bm25_threshold": cache_bm25_threshold(),
        "bm25_strict_threshold": cache_bm25_strict_threshold(),
        "similarity_min": cache_similarity_min(),
        "topk": cache_topk(),
        "cache_size_kb": cache_size_kb(),
        "mmap_size_mb": cache_mmap_size_mb(),
    }


# ── OtelTracer Configuration ─────────────────────────────────────────────────

def tracer_enabled() -> bool:
    """Check if OpenTelemetry tracing is enabled."""
    return bool(get("tracer", "enabled", True))


def tracer_jaeger_host() -> str:
    """Get Jaeger agent hostname for trace export."""
    return str(get("tracer", "jaeger_host", "localhost"))


def tracer_jaeger_port() -> int:
    """Get Jaeger agent port (default 6831 for thrift)."""
    return int(get("tracer", "jaeger_port", 6831))


def tracer_jaeger_enabled() -> bool:
    """Check if Jaeger exporter is explicitly enabled (requires explicit config)."""
    return bool(get("tracer", "jaeger_enabled", False))


def tracer_batch_size() -> int:
    """Get span batch size before exporting to Jaeger."""
    return int(get("tracer", "batch_size", 512))


def tracer_batch_timeout_ms() -> int:
    """Get timeout in ms before exporting partial span batch."""
    return int(get("tracer", "batch_timeout_ms", 5000))


def tracer_config() -> dict:
    """Return complete OpenTelemetry tracer configuration."""
    return {
        "enabled": tracer_enabled(),
        "jaeger_host": tracer_jaeger_host(),
        "jaeger_port": tracer_jaeger_port(),
        "jaeger_enabled": tracer_jaeger_enabled(),
        "batch_size": tracer_batch_size(),
        "batch_timeout_ms": tracer_batch_timeout_ms(),
    }


# ── WorkspaceGuard Configuration ─────────────────────────────────────────────

def workspace_guard_enabled() -> bool:
    """Check if workspace guard is enabled (prevents modification of critical files)."""
    return bool(get("workspace_guard", "enabled", True))


def workspace_guard_critical_files() -> list[str]:
    """Get list of critical files protected by workspace guard."""
    default_critical = [
        "orchestra/engine/runner.py",
        "orchestra/config.py",
        "orchestra/storage/event_log.py",
    ]
    return get("workspace_guard", "critical_files", default_critical)


def workspace_guard_config() -> dict:
    """Return complete workspace guard configuration."""
    return {
        "enabled": workspace_guard_enabled(),
        "critical_files": workspace_guard_critical_files(),
    }


# ── Performance Tuning Configuration ─────────────────────────────────────────

def thread_pool_size() -> int:
    """Get thread pool size for executor."""
    return int(get("performance", "thread_pool_size", 10))


def thread_pool_timeout_s() -> int:
    """Get thread pool shutdown timeout in seconds."""
    return int(get("performance", "thread_pool_timeout_s", 300))


def performance_config() -> dict:
    """Return complete performance tuning configuration."""
    return {
        "thread_pool_size": thread_pool_size(),
        "thread_pool_timeout_s": thread_pool_timeout_s(),
    }


# ── Retry Policy Configuration ───────────────────────────────────────────────

def retry_max_attempts() -> int:
    """Get maximum retry attempts for transient failures."""
    return int(get("retry", "max_attempts", 3))


def retry_backoff_ms() -> int:
    """Get initial backoff delay in milliseconds."""
    return int(get("retry", "backoff_ms", 100))


def retry_backoff_multiplier() -> float:
    """Get exponential backoff multiplier."""
    return float(get("retry", "backoff_multiplier", 2.0))


def retry_config() -> dict:
    """Return complete retry policy configuration."""
    return {
        "max_attempts": retry_max_attempts(),
        "backoff_ms": retry_backoff_ms(),
        "backoff_multiplier": retry_backoff_multiplier(),
    }


# ── Configuration Validation ─────────────────────────────────────────────────

def validate_config() -> tuple[bool, list[str]]:
    """Validate all configuration values for correctness.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    # EventLog validation
    if eventlog_busy_timeout_ms() < 100:
        errors.append("eventlog.busy_timeout_ms must be >= 100ms")
    if eventlog_cache_size_kb() < 1000:
        errors.append("eventlog.cache_size_kb should be >= 1000KB for good perf")

    # Checkpoint validation
    if checkpoint_ttl_hours() <= 0:
        errors.append("checkpoint.ttl_hours must be > 0")

    # Idempotency validation
    if idempotency_ttl_hours() <= 0:
        errors.append("idempotency.ttl_hours must be > 0")
    if idempotency_max_cache_size() < 100:
        errors.append("idempotency.max_cache_size should be >= 100")

    # Scrubber validation
    scrub_entropy = scrubber_entropy_threshold()
    if scrub_entropy < 3.0 or scrub_entropy > 8.0:
        errors.append(f"scrubber.entropy_threshold should be 3.0-8.0, got {scrub_entropy}")

    # Cache validation
    if cache_ttl_hours() <= 0:
        errors.append("cache.ttl_hours must be > 0")
    if cache_similarity_min() <= 0 or cache_similarity_min() > 1:
        errors.append("cache.similarity_min must be in range (0, 1]")
    if cache_topk() < 1:
        errors.append("cache.topk must be >= 1")

    # Tracer validation
    if tracer_batch_size() < 1:
        errors.append("tracer.batch_size must be >= 1")
    if tracer_batch_timeout_ms() < 100:
        errors.append("tracer.batch_timeout_ms must be >= 100ms")

    # Performance validation
    if thread_pool_size() < 1:
        errors.append("performance.thread_pool_size must be >= 1")
    if thread_pool_timeout_s() < 10:
        errors.append("performance.thread_pool_timeout_s should be >= 10s")

    # Retry validation
    if retry_max_attempts() < 1:
        errors.append("retry.max_attempts must be >= 1")
    if retry_backoff_ms() < 1:
        errors.append("retry.backoff_ms must be >= 1")
    if retry_backoff_multiplier() <= 1.0:
        errors.append("retry.backoff_multiplier must be > 1.0")

    return len(errors) == 0, errors


# ── Dynamic Configuration Updates ────────────────────────────────────────────

def set(section: str, key: str, value: Any) -> None:
    """Programmatically set a configuration value (in-memory only).

    Changes are NOT persisted to file. Use for runtime tuning.
    """
    global _cfg
    if section not in _cfg:
        _cfg[section] = {}
    _cfg[section][key] = value

"""Configuration accessors for the infrastructure blocks (event log, checkpoints, PII scrubber,
answer cache, tracer, workspace guard, thread pool) and `validate_config`.

Moved out of `config.py` to keep it under 800 lines; re-exported there, so callers use
`config.<name>`. Every read goes through `config.get` / `config.<accessor>` so env overrides,
`reload_config` and test patches apply.
"""
from __future__ import annotations

from pathlib import Path

from orchestra import config


# ── EventLog Block Configuration ─────────────────────────────────────────────

def eventlog_busy_timeout_ms() -> int:
    """Get EventLog PRAGMA busy_timeout value in milliseconds."""
    return int(config.get("eventlog", "busy_timeout_ms", 5000))


def eventlog_cache_size_kb() -> int:
    """Get EventLog PRAGMA cache_size in KB (-X means use X KB)."""
    return int(config.get("eventlog", "cache_size_kb", 64000))


# ── State Suspension Configuration ───────────────────────────────────────────

def checkpoint_ttl_hours() -> int:
    """Get checkpoint TTL in hours before expiration."""
    return int(config.get("checkpoint", "ttl_hours", 48))


# ── PiiScrubber Configuration ────────────────────────────────────────────────

def scrubber_entropy_threshold() -> float:
    """Get entropy threshold for PII scrubber (prevents false positives on hashes)."""
    return float(config.get("scrubber", "entropy_threshold", 5.0))


# ── FTS5Cache Configuration ──────────────────────────────────────────────────

def cache_db_path() -> Path:
    """Get FTS5 cache database path."""
    path_str = config.get("fts_cache", "db_path", "~/.orchestra/fts_cache.db")
    return Path(path_str).expanduser().resolve()


def fts_cache_enabled() -> bool:
    """Whether a run may be answered from the cross-run answer cache."""
    return bool(config.get("fts_cache", "enabled", True))


def cache_ttl_hours() -> int:
    """Get FTS5 cache entry TTL in hours (default 7 days)."""
    return int(config.get("fts_cache", "ttl_hours", 168))


def cache_size_kb() -> int:
    """Get FTS5 cache SQLite cache size in KB."""
    return int(config.get("fts_cache", "cache_size_kb", 64000))


# ── OtelTracer Configuration ─────────────────────────────────────────────────

def tracer_enabled() -> bool:
    """Check if OpenTelemetry tracing is enabled."""
    return bool(config.get("tracer", "enabled", True))


def tracing_enabled() -> bool:
    """Alias for config.tracer_enabled(). Allows disabling tracing without uninstalling packages."""
    return config.tracer_enabled()


def tracer_otlp_endpoint() -> str:
    """Base URL of the OTLP collector spans are sent to (`[tracer] otlp_endpoint`); empty: none."""
    return str(config.get("tracer", "otlp_endpoint", "") or "")


def tracer_otlp_headers() -> dict:
    """Headers for the collector, e.g. its credentials (`[tracer] otlp_headers`)."""
    headers = config.get("tracer", "otlp_headers", {})
    return {str(key): str(value) for key, value in headers.items()} if isinstance(headers, dict) else {}


def tracer_jaeger_enabled() -> bool:
    """Check if Jaeger exporter is explicitly enabled (requires explicit config)."""
    return bool(config.get("tracer", "jaeger_enabled", False))


def tracer_batch_size() -> int:
    """Get span batch size before exporting to Jaeger."""
    return int(config.get("tracer", "batch_size", 512))


def tracer_batch_timeout_ms() -> int:
    """Get timeout in ms before exporting partial span batch."""
    return int(config.get("tracer", "batch_timeout_ms", 5000))


# ── WorkspaceGuard Configuration ─────────────────────────────────────────────

def workspace_guard_enabled() -> bool:
    """Check if workspace guard is enabled (prevents modification of critical files)."""
    return bool(config.get("workspace_guard", "enabled", True))


def workspace_guard_critical_files() -> list[str]:
    """Get list of critical files protected by workspace guard."""
    default_critical: list[str] = [
        "orchestra/engine/runner.py",
        "orchestra/config.py",
        "orchestra/storage/event_log.py",
    ]
    return list(config.get("workspace_guard", "critical_files", default_critical))


# ── Performance Tuning Configuration ─────────────────────────────────────────

def thread_pool_size() -> int:
    """Get thread pool size for executor."""
    return int(config.get("performance", "thread_pool_size", 10))


def thread_pool_timeout_s() -> int:
    """Get thread pool shutdown timeout in seconds."""
    return int(config.get("performance", "thread_pool_timeout_s", 300))


# ── Configuration Validation ─────────────────────────────────────────────────

def validate_config() -> tuple[bool, list[str]]:
    """Validate all configuration values for correctness.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    # EventLog validation
    if config.eventlog_busy_timeout_ms() < 100:
        errors.append("eventlog.busy_timeout_ms must be >= 100ms")
    if config.eventlog_cache_size_kb() < 1000:
        errors.append("eventlog.cache_size_kb should be >= 1000KB for good perf")

    # Checkpoint validation
    if config.checkpoint_ttl_hours() <= 0:
        errors.append("checkpoint.ttl_hours must be > 0")

    # Idempotency validation

    # Scrubber validation
    scrub_entropy = config.scrubber_entropy_threshold()
    if scrub_entropy < 3.0 or scrub_entropy > 8.0:
        errors.append(f"scrubber.entropy_threshold should be 3.0-8.0, got {scrub_entropy}")

    # FTS cache validation
    if config.cache_ttl_hours() <= 0:
        errors.append("fts_cache.ttl_hours must be > 0")

    # Tracer validation
    if config.tracer_batch_size() < 1:
        errors.append("tracer.batch_size must be >= 1")
    if config.tracer_batch_timeout_ms() < 100:
        errors.append("tracer.batch_timeout_ms must be >= 100ms")

    # Performance validation
    if config.thread_pool_size() < 1:
        errors.append("performance.thread_pool_size must be >= 1")
    if config.thread_pool_timeout_s() < 10:
        errors.append("performance.thread_pool_timeout_s should be >= 10s")

    # Retry validation
    if config.retry_max_attempts() < 1:
        errors.append("retry.max_attempts must be >= 1")
    if config.retry_backoff_ms() < 1:
        errors.append("retry.backoff_ms must be >= 1")
    if config.retry_backoff_multiplier() <= 1.0:
        errors.append("retry.backoff_multiplier must be > 1.0")

    # Persona validation
    personas = config.personas_config()
    known_personas = {"implementer", "reviewer", "researcher", "synthesizer", "verifier"}
    for key in personas:
        if key not in known_personas and key != "fluid_override":
            errors.append(f"personas.{key} is not a recognized persona")

    # Autonomy policy validation
    policy = config.autonomy_policy_config()
    if policy.get("default_action") not in (None, "", "escalate", "block"):
        errors.append("autonomy_policy.default_action must be 'escalate' or 'block'")
    if policy.get("cost_threshold_usd", 0.5) < 0:
        errors.append("autonomy_policy.cost_threshold_usd must be >= 0")
    if policy.get("disagreement_entropy_threshold", 0.6) < 0 or policy.get("disagreement_entropy_threshold", 0.6) > 1:
        errors.append("autonomy_policy.disagreement_entropy_threshold must be in [0, 1]")

    return len(errors) == 0, errors

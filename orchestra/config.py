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

def _read_toml(path: Path) -> tuple[dict, str | None]:
    if not path.exists():
        return {}, None
    try:
        with open(path, "rb") as _f:
            return tomllib.load(_f), None
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return {}, f"{path}: {type(exc).__name__}: {exc}"


def _load_config() -> tuple[dict, str | None]:
    return _read_toml(_config_path)


def user_config_path() -> Path:
    """`~/.orchestra/config.toml`: what is the user's in every project (their plugins' MCP servers,
    user_setup). A project's config wins key by key."""
    return Path(os.environ.get("ORCHESTRA_USER_CONFIG") or Path.home() / ".orchestra" / "config.toml")


def _load_user_config() -> tuple[dict, str | None]:
    path = user_config_path()
    if path.resolve() == _config_path.resolve():   # run from the home folder: one file, not two
        return {}, None
    return _read_toml(path)


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
        if not key.startswith(prefix) or key == "ORCHESTRA_USER_CONFIG":   # a path, not a setting
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
_user_cfg, _user_config_error = _load_user_config()
_cfg = _deep_merge(_deep_merge(_user_cfg, _file_cfg), _env_overrides())


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
    global _cfg, _file_cfg, _config_error, _user_cfg, _user_config_error
    _file_cfg, _config_error = _load_config()
    _user_cfg, _user_config_error = _load_user_config()
    _cfg = _deep_merge(_deep_merge(_user_cfg, _file_cfg), _env_overrides())


def timeout() -> int:
    return int(_cfg.get("orchestra", {}).get("timeout", 180))


def max_agent_seconds() -> int:
    """`[orchestra] max_agent_seconds`: the longest one agent call may run (default 2 h).

    An agent's timeout (`[timeouts]`) is the longest it may stay silent; one that keeps
    answering or thinking runs on until it finishes or reaches this cap.
    """
    return int(_cfg.get("orchestra", {}).get("max_agent_seconds", 7200))


def timeout_for_alias(alias: str | None = None) -> int:
    default_timeout = timeout()
    if not alias:
        return default_timeout

    # `alias#mode` and `alias@node` are the same agent: the exact name is looked up first,
    # then the name without the mode, then the bare alias.
    without_mode = alias.split("#", 1)[0] + ("@" + alias.split("@", 1)[1] if "@" in alias else "")
    bare = alias.split("#", 1)[0].split("@", 1)[0]
    configured = _cfg.get("timeouts", {})
    for name in (alias, without_mode, bare):
        if configured.get(name) is not None:
            return int(configured[name])
    alias = bare

    deep_defaults = {
        "cdx-deep": 900,
        # Antigravity (gmn-*/agy-*): a task that searches the web outlasted 180 s on every attempt.
        "gmn-pro": 600,
        "agy-argon": 600,
        "agy-deep": 600,
        "cld-deep": 300,
        "kimi-deep": 600,
    }
    fast_defaults = {
        "cdx-fast": 180,
        "gmn-fast": 300,
        "agy-flash": 300,
        "agy-fast": 300,
        "cld-fast": 180,
        "kimi-code": 300,
    }
    if alias in deep_defaults:
        return deep_defaults[alias]
    if alias in fast_defaults:
        return fast_defaults[alias]
    return default_timeout


def max_turns() -> int:
    return int(_cfg.get("limits", {}).get("max_turns", 10))


def isolation_config() -> dict:
    """`[isolation]`: run writing aliases in a git worktree of their own (default off)."""
    settings = section("isolation")
    aliases = settings.get("aliases")
    return {
        "enabled": bool(settings.get("enabled", False)),
        "aliases": tuple(aliases) if isinstance(aliases, (list, tuple)) else ("cld-write",),
    }


def outcome_routing_config() -> dict:
    """`[outcome_routing]`: let `auto` pick the mode that worked on similar tasks (default off)."""
    settings = section("outcome_routing")
    return {
        "enabled": bool(settings.get("enabled", False)),
        "min_history_threshold": int(settings.get("min_history_threshold", 50)),
        "top_k_similar": int(settings.get("top_k_similar", 10)),
        "time_decay_days": float(settings.get("time_decay_days", 30)),
    }


def approval_timeout_hours() -> float:
    """How long a run may wait for approval before it is cancelled (`[orchestra]
    approval_timeout_hours`, default 24): a forgotten gate held a run for three months."""
    return float(_cfg.get("orchestra", {}).get("approval_timeout_hours", 24))


def stale_run_seconds() -> int:
    """How long a running run may show no activity, with no live agent process, before it is
    recorded as failed (`[orchestra] stale_run_seconds`, default two hours)."""
    return int(_cfg.get("orchestra", {}).get("stale_run_seconds", 7200))


def max_budget_usd() -> float:
    """Spending limit of one run in USD; 0 means no limit.

    `[limits] max_budget_usd` is the documented key. `[execution] max_budget_usd` is the
    name the check used to read, and still counts when the documented key is absent.
    """
    for settings in (section("limits"), execution_config()):
        value = settings.get("max_budget_usd")
        if value is not None:
            return float(value)
    return 0.0


def auto_synthesize() -> bool:
    return bool(_cfg.get("orchestra", {}).get("auto_synthesize", False))


def config_path() -> Path:
    return _config_path


def config_error() -> str | None:
    return _config_error or _user_config_error


def fts_cache_db_path() -> str:
    """Return FTS5 cache database path."""
    raw_path = str(_cfg.get("fts_cache", {}).get("db_path", "~/.orchestra/fts_cache.db"))
    import os
    return os.path.expanduser(raw_path)


def orchestra_root() -> Path:
    return _config_path.parent


def project_root() -> Path:
    return orchestra_root().parent


def artifact_root() -> Path:
    rel = str(_cfg.get("orchestra", {}).get("artifact_root", "runs"))
    return (orchestra_root() / rel).resolve()


def memory_db_path() -> Path:
    return (orchestra_root() / "memory.db").resolve()


def db_path() -> Path:
    env_path = os.environ.get("ORCHESTRA_DB_PATH")
    if env_path:
        return Path(env_path).resolve()
    return (orchestra_root() / "orchestra.db").resolve()


def provider_config(provider: str) -> dict:
    return dict(_cfg.get("providers", {}).get(provider, {}))


def providers_config() -> dict:
    return section("providers")


def ollama_backends_config() -> dict:
    """Raw [providers.ollama.backends] table; empty dict when not configured."""
    return dict(_cfg.get("providers", {}).get("ollama", {}).get("backends", {}))


_LOCAL_PROVIDERS = ("ollama/",)
_CLOUD_MODEL_MARKS = (":cloud", "-cloud")


def price_of(model_label: str) -> dict:
    """Prices of a model per 1k tokens: {"input", "output", "cached"}.

    `[pricing]` holds either one number (input and output alike) or a table with `input`,
    `output` and optionally `cached` (input read from a cache; default a tenth of `input`).
    A local model, an Ollama model that is not a cloud one, costs nothing unless it is priced.
    """
    entry = _cfg.get("pricing", {}).get(model_label)
    if isinstance(entry, dict):
        price_in = float(entry.get("input", 0.0))
        return {"input": price_in, "output": float(entry.get("output", price_in)),
                "cached": float(entry.get("cached", price_in * 0.1))}
    if entry is None and model_label.startswith(_LOCAL_PROVIDERS) \
            and not model_label.endswith(_CLOUD_MODEL_MARKS):
        return {"input": 0.0, "output": 0.0, "cached": 0.0}
    flat = price_per_1k_tokens(model_label)
    return {"input": flat, "output": flat, "cached": flat * 0.1}


def price_per_1k_tokens(model_label: str) -> float:
    pricing = _cfg.get("pricing", {})
    if model_label in pricing:
        entry = pricing[model_label]
        return float(entry.get("output", entry.get("input", 0.0)) or 0.0 if isinstance(entry, dict) else entry)

    defaults = {
        "gpt-5.4/low": 0.002,
        "gpt-5.4/medium": 0.004,
        "gpt-5.4/high": 0.008,
        "gpt-5.4/xhigh": 0.012,
        "claude/sonnet": 0.006,
        "claude/opus": 0.015,
        "antigravity/gemini-4-argon-high": 0.005,
        "antigravity/gemini-3.8-flash-high": 0.001,
    }
    return float(defaults.get(model_label, 0.004))


def routing_config() -> dict:
    return section("routing")


def review_config() -> dict:
    return section("review")


def synthesis_config() -> dict:
    return section("synthesis")


def validator_config() -> dict:
    return section("validator")


def context_strategy() -> str:
    """Return the context injection strategy: 'full', 'surgical', or 'minimal'."""
    return str(_cfg.get("orchestra", {}).get("context_strategy", "full"))

def execution_config() -> dict:
    return section("execution")


def provider_enabled(name: str) -> bool:
    """False when `[providers.<name>] enabled = false`: the provider is never used,
    and aliases that point at it fall straight through to their fallback."""
    providers = section("providers")
    entry = providers.get(name, {}) if isinstance(providers, dict) else {}
    return bool(entry.get("enabled", True)) if isinstance(entry, dict) else True


def availability_config() -> dict:
    providers = section("providers")
    availability = providers.get("availability", {}) if isinstance(providers, dict) else {}
    return availability if isinstance(availability, dict) else {}


def token_budget_config() -> dict:
    return section("token_budget")


def rate_limit_config() -> dict:
    return section("rate_limits")


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


# ── Eval Harness Configuration ───────────────────────────────────────────────

def eval_harness_enabled() -> bool:
    """Auto-judge run output with the LLM eval harness after completion (default off)."""
    return bool(get("eval", "auto_enabled", False))


def eval_judge_alias() -> str:
    """Alias of the judge model used by the auto eval harness."""
    return str(get("eval", "judge_alias", "cld-fast"))


def calibration_judge_alias() -> str:
    """Alias of the judge model used for calibration / anti-sycophancy assessment."""
    return str(get("calibration", "judge_alias", "cld-fast"))


def eval_calibration_enabled() -> bool:
    """Also assess calibration / anti-sycophancy during auto eval (default off)."""
    return bool(get("eval", "calibration_enabled", False))


def eval_sample_rate() -> float:
    """Fraction of completed runs to auto-judge (production trace sampling; 1.0 = all)."""
    return float(get("eval", "sample_rate", 1.0))


def mcp_allow_approve() -> bool:
    """Whether the agent calling the MCP tools may approve a run paused at an approval gate
    (`[mcp] allow_approve`, default off: the gate is for the owner)."""
    return bool(get("mcp", "allow_approve", False))


# ── Durable Execution ──────────────────────────────────────────────────────────

def durable_execution_enabled() -> bool:
    """Persist run state on SIGINT/SIGTERM and enable ask/dual crash-recovery
    resume from the last checkpoint (default off)."""
    return bool(get("durable_execution", "enabled", False))


# ── Reflection Loop Configuration ────────────────────────────────────────────

def reflection_enabled() -> bool:
    """Run a self-critique + retry loop on ask-mode output (default off)."""
    return bool(get("reflection", "enabled", False))


def reflection_max_rounds() -> int:
    """Max critique/retry rounds in the reflection loop."""
    return int(get("reflection", "max_rounds", 2))


def reflection_critique_alias() -> str:
    """Alias of the critic model used by the reflection loop."""
    return str(get("reflection", "critique_alias", "cld-fast"))


def reflection_record_lessons() -> bool:
    """Record a lesson to episodic memory when reflection revises an answer (default off).

    With RAG enabled, these lessons are retrieved for future similar tasks (Reflexion loop).
    """
    return bool(get("reflection", "record_lessons", False))


# ── Supervisor (Hierarchical Subagents) Configuration ────────────────────────

def supervisor_max_depth() -> int:
    """Max recursion depth for hierarchical subagent spawning (guards infinite spawn)."""
    return int(get("supervisor", "max_depth", 2))


def planned_max_seconds() -> int:
    """Time budget of a planned run: past it no further node is started ([planned] max_seconds)."""
    return int(get("planned", "max_seconds", 900))


def web_research_alias() -> str:
    """The agent for a task that needs the web (`[routing] web_research_alias`): it must have web
    tools; Gemini 4 Argon scored highest in the 2026-10-21 bench and searches the web."""
    return str(_cfg.get("routing", {}).get("web_research_alias", "agy-argon"))


def default_light_alias() -> str:
    """The alias for small single-agent work: the first agent of the light tier."""
    agents = (tier_config().get("light") or {}).get("agents") or []
    return str(agents[0]) if agents else "cdx-fast"


def supervisor_worker_alias() -> str:
    """Alias of the leaf worker model that solves atomic subtasks."""
    return str(get("supervisor", "worker_alias", None) or default_light_alias())


def supervisor_planner_alias() -> str:
    """Alias of the planner model that decomposes tasks into subtasks."""
    return str(get("supervisor", "planner_alias", None) or personas_config().get("researcher") or "gmn-pro")


def supervisor_synthesizer_alias() -> str:
    """Alias of the synthesizer model that combines subagent results."""
    return str(get("supervisor", "synthesizer_alias", None) or personas_config().get("synthesizer") or "gmn-pro")


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


def retry_enabled() -> bool:
    """Retry a failed run with bounded attempts before giving up (default off)."""
    return bool(get("retry", "enabled", False))


def retry_degrade_to_cheaper() -> bool:
    """On retry, step down to a cheaper same-provider alias (graceful degradation)."""
    return bool(get("retry", "degrade_to_cheaper", False))


# ── Automatic RAG Configuration ──────────────────────────────────────────────

def rag_enabled() -> bool:
    """Retrieve relevant semantic-memory context and inject it before each agent run."""
    return bool(get("rag", "enabled", False))


def rag_retrieval_limit() -> int:
    """Top-k entries pulled from semantic memory for RAG injection."""
    return int(get("rag", "limit", 3))


def rag_max_chars() -> int:
    """Character cap on the injected retrieved-context block."""
    return int(get("rag", "max_chars", 2000))


def context_memory_config() -> dict:
    """Return context memory configuration for tiered prompt assembly."""
    return section("context_memory")


# ── Typed Memory & Recency Configuration ─────────────────────────────────────

def memory_typed_enabled() -> bool:
    """Split semantic memory into episodic/semantic/procedural stores (default off)."""
    return bool(get("memory", "typed_enabled", False))


def memory_recency_enabled() -> bool:
    """Add an exponential recency decay boost to memory retrieval scoring (default off)."""
    return bool(get("memory", "recency_enabled", False))


def memory_recency_half_life_hours() -> float:
    """Age (hours) at which the recency boost halves."""
    return float(get("memory", "recency_half_life_hours", 168.0))


def memory_recency_weight() -> float:
    """Maximum recency boost (added to similarity for a brand-new entry)."""
    return float(get("memory", "recency_weight", 0.1))


def retry_config() -> dict:
    """Return complete retry policy configuration."""
    return {
        "max_attempts": retry_max_attempts(),
        "backoff_ms": retry_backoff_ms(),
        "backoff_multiplier": retry_backoff_multiplier(),
    }


# ── Persona Configuration ────────────────────────────────────────────────────

def personas_config() -> dict:
    """Return persona-to-alias mapping for fixed-role UX.

    Keys:
        implementer, reviewer, researcher, synthesizer, verifier
    """
    return section("personas")


# ── Local-First Configuration ────────────────────────────────────────────────

def local_first_enabled() -> bool:
    """Check if local-first (air-gapped) mode is active.

    When enabled, all routing defaults to Ollama/local providers only.
    """
    return bool(get("local_first", "enabled", False))


def local_first_config() -> dict:
    """Return complete local-first configuration.

    Keys:
        enabled: bool
        synthesis_alias: str — local model for synthesis
        persona_implementer: str — local model for implementation
        persona_reviewer: str — local model for review
        persona_researcher: str — local model for research
        persona_verifier: str — local model for verification
        cascade_chain: list[str] — local-only cascade escalation chain
    """
    return section("local_first")


# ── Autonomy Policy Configuration ────────────────────────────────────────────

def autonomy_policy_config() -> dict:
    """Return autonomy policy configuration for human-in-the-loop escalation.

    Keys:
        cost_threshold_usd: float — max estimated cost before escalation
        disagreement_entropy_threshold: float — max divergence before escalation
        semantic_novelty_std_threshold: float — max std-dev from historical tasks
        privileged_actions: list[str] — keywords that always trigger escalation
        mode_gates: list[str] — modes that always require approval
        default_action: str — "escalate" or "block"
    """
    return section("autonomy_policy")


# ── Adversarial Configuration ────────────────────────────────────────────────

def adversarial_enabled() -> bool:
    return bool(_cfg.get("adversarial", {}).get("enabled", False))

def adversarial_diversity_enforced() -> bool:
    return bool(_cfg.get("adversarial", {}).get("diversity_enforced", True))

def adversarial_red_team_alias() -> str:
    return str(_cfg.get("adversarial", {}).get("red_team_alias", "cld-fast"))

def adversarial_counter_example_alias() -> str:
    return str(_cfg.get("adversarial", {}).get("counter_example_alias", "gmn-pro"))

def adversarial_counter_example_threshold() -> float:
    return float(_cfg.get("adversarial", {}).get("counter_example_threshold", 0.15))


# ── Compression Configuration ────────────────────────────────────────────────

def compression_config() -> dict:
    """Return context compression layer configuration.

    Keys:
        semantic_retrieval_enabled: bool — query SemanticMemory for relevant context
        code_context_enabled: bool — read file headers mentioned in the prompt
    """
    return dict(_cfg.get("compression", {}))


# ── Replay Configuration ─────────────────────────────────────────────────────

def replay_enabled() -> bool:
    return bool(_cfg.get("replay", {}).get("enabled", False))


# ── Federation Configuration ─────────────────────────────────────────────────

def federation_config() -> dict:
    """Return federated semantic memory configuration.

    Keys:
        enabled (bool): Master switch for federation
        peer_urls (list[str]): Static list of peer endpoints
        auth_token (str): Shared secret for peer authentication
        sync_interval_seconds (int): Background sync interval
        max_outbox_size (int): Cap on pending outbound summaries
    """
    return section("federation")


# ── Dynamic Configuration Updates ────────────────────────────────────────────

def set(section: str, key: str, value: Any) -> None:
    """Programmatically set a configuration value (in-memory only).

    Changes are NOT persisted to file. Use for runtime tuning.
    """
    global _cfg
    if section not in _cfg:
        _cfg[section] = {}
    _cfg[section][key] = value


# ── Policy YAML Configuration ────────────────────────────────────────────────

def policy_yaml_path() -> Path | None:
    """Return the path to .orchestra/policy.yaml if it exists, else None.

    Looks in the same directory that contains config.toml.
    """
    cfg = _find_config()
    candidate = cfg.with_name("policy.yaml")
    if candidate.exists():
        return candidate
    return None


# Accessors kept in config_blocks.py (imported last: it reads through this module).
from orchestra.config_blocks import (  # noqa: E402,F401
    eventlog_busy_timeout_ms,
    eventlog_cache_size_kb,
    checkpoint_ttl_hours,
    scrubber_entropy_threshold,
    cache_db_path,
    fts_cache_enabled,
    cache_ttl_hours,
    cache_size_kb,
    tracer_enabled,
    tracing_enabled,
    tracer_otlp_endpoint,
    tracer_otlp_headers,
    tracer_jaeger_enabled,
    tracer_batch_size,
    tracer_batch_timeout_ms,
    workspace_guard_enabled,
    workspace_guard_critical_files,
    thread_pool_size,
    thread_pool_timeout_s,
    validate_config,
)

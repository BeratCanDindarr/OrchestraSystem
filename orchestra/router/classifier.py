"""Heuristic and semantic task router with optional local-model dispatch."""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from orchestra import config
from orchestra.providers.ollama import OllamaProvider
from orchestra.storage.db import get_db
from orchestra.storage.reputation import choose_alias_by_reputation
from orchestra.router.semantic import suggest_best_alias

def select_alias_for_candidates(task: str, aliases: list[str]) -> str:
    if not aliases: return "cdx-fast"
    if len(aliases) == 1: return aliases[0]
    try:
        semantic_best = suggest_best_alias(task, aliases)
        if semantic_best: return semantic_best
    except: pass
    try:
        connection = get_db()
        selected = choose_alias_by_reputation(connection, aliases, task=task)
        return selected or aliases[0]
    except: return aliases[0]

if sys.version_info >= (3, 11): import tomllib
else:
    try: import tomllib
    except ImportError: import tomli as tomllib

DEFAULT_POLICY_PATH = Path(__file__).with_name("policy.toml")
_DEFAULT_KEYWORDS = {
    "cheap": ["hello", "selam"],
    "deep": ["research", "araştır", "plan", "analyze"],
}


def _load_router_policy() -> dict:
    if not DEFAULT_POLICY_PATH.exists():
        return {"keywords": dict(_DEFAULT_KEYWORDS)}
    try:
        with open(DEFAULT_POLICY_PATH, "rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {"keywords": dict(_DEFAULT_KEYWORDS)}

    keywords = data.get("keywords", {}) if isinstance(data, dict) else {}
    cheap = keywords.get("cheap") if isinstance(keywords, dict) else None
    deep = keywords.get("deep") if isinstance(keywords, dict) else None
    return {
        "keywords": {
            "cheap": cheap if isinstance(cheap, list) and cheap else list(_DEFAULT_KEYWORDS["cheap"]),
            "deep": deep if isinstance(deep, list) and deep else list(_DEFAULT_KEYWORDS["deep"]),
        }
    }


_ROUTER_POLICY = _load_router_policy()

_ollama = OllamaProvider()


@dataclass
class RouteDecision:
    mode: str
    alias: str | None = None
    require_approval: bool = False
    reason: str = ""
    classifier: str = "heuristic"
    confidence: float = 0.5

def classify(task: str) -> str:
    """Return cheap | balanced | deep | extreme (auto-plan)."""
    routing = config.routing_config()
    large_task_chars = int(routing.get("large_task_chars", 1500))
    large_task_lines = int(routing.get("large_task_lines", 20))
    # 🛡 NEW: Auto-Decomposition Trigger
    if len(task) > large_task_chars or task.count("\n") > large_task_lines:
        return "extreme"

    task_low = task.lower()
    if any(k in task_low for k in _ROUTER_POLICY["keywords"]["cheap"]): return "cheap"
    if any(k in task_low for k in _ROUTER_POLICY["keywords"]["deep"]): return "deep"
    return "balanced"


def _heuristic_route(task: str, *, preferred_alias: str | None = None, intent_hint: str = "") -> RouteDecision:
    routing = config.routing_config()
    large_task_chars = int(routing.get("large_task_chars", 1500))
    large_task_lines = int(routing.get("large_task_lines", 20))
    text = (task or "").lower()
    hint = (intent_hint or "").lower()

    if len(task) > large_task_chars or task.count("\n") > large_task_lines:
        return RouteDecision(mode="planned", reason="large_or_multistep_request", classifier="heuristic", confidence=0.95)

    review_terms = [
        "review", "incele", "karşılaştır", "compare", "trade-off", "artı", "eksi",
        "pros", "cons", "hangi daha", "which is better",
    ]
    critical_terms = [
        "edit", "change", "modify", "refactor", "fix", "implement", "apply", "patch",
        "rename", "move", "create", "delete", "ekle", "düzelt", "değiştir", "taşı",
        "installer", "prefab", "scene", "scriptableobject", "scriptable object",
        "addressable", "projectsettings", "yaml",
    ]
    ask_terms = [
        "what", "neden", "niye", "nasıl", "show", "find", "bul", "ver", "where",
        "explain", "açıkla", "hangi dosya", "hangi asset", "hangi script", "lookup",
    ]

    if hint == "plan":
        return RouteDecision(mode="planned", reason="intent_hint_plan", classifier="heuristic", confidence=0.9)
    if hint == "review" and any(term in text for term in critical_terms):
        return RouteDecision(mode="critical", require_approval=True, reason="review_hint_with_mutation", classifier="heuristic", confidence=0.85)
    if hint == "review":
        return RouteDecision(mode="dual", reason="intent_hint_review", classifier="heuristic", confidence=0.85)

    if any(term in text for term in review_terms):
        return RouteDecision(mode="dual", reason="comparison_or_review_request", classifier="heuristic", confidence=0.8)
    if any(term in text for term in critical_terms):
        return RouteDecision(mode="critical", require_approval=True, reason="workspace_mutation_request", classifier="heuristic", confidence=0.86)
    if any(term in text for term in ask_terms):
        return RouteDecision(
            mode="ask",
            alias=preferred_alias or select_alias_for_candidates(task, ["cdx-fast", "gmn-fast", "cld-fast"]),
            reason="lookup_or_explanation_request",
            classifier="heuristic",
            confidence=0.78,
        )

    return RouteDecision(
        mode="ask",
        alias=preferred_alias or select_alias_for_candidates(task, ["cdx-fast", "gmn-fast", "cld-fast"]),
        reason="default_balanced_lookup",
        classifier="heuristic",
        confidence=0.55,
    )


def _model_route(task: str, *, preferred_alias: str | None = None, intent_hint: str = "") -> RouteDecision | None:
    if not _ollama.is_available():
        return None
    routing = config.routing_config()
    timeout = int(routing.get("model_router_timeout", 20))

    prompt = f"""Classify this software-engineering request for an orchestration router.
Return JSON only with keys:
mode: ask | dual | critical | planned
require_approval: true | false
reason: short_snake_case_reason

Rules:
- ask = explanation, lookup, find, inspect, answer
- dual = compare, review, architecture trade-off
- critical = likely workspace mutation, file edits, refactor, apply patch, risky config/scene/prefab changes
- planned = large decomposition, multi-step scaffold/plan
- require_approval should be true for critical edits that could mutate files/assets/config

intent_hint={intent_hint or "none"}
preferred_alias={preferred_alias or "none"}

TASK:
{task}
"""
    raw, rc = _ollama.native_run(prompt, "mini", timeout=timeout)
    if rc != 0 or not raw:
        return None

    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except Exception:
        return None

    mode = str(data.get("mode", "")).strip().lower()
    if mode not in {"ask", "dual", "critical", "planned"}:
        return None
    return RouteDecision(
        mode=mode,
        alias=(preferred_alias if mode == "ask" else None),
        require_approval=bool(data.get("require_approval", mode == "critical")),
        reason=str(data.get("reason", "model_router")).strip() or "model_router",
        classifier="oll-mini",
        confidence=0.72,
    )


def route_task(task: str, *, preferred_alias: str | None = None, intent_hint: str = "") -> RouteDecision:
    heuristic = _heuristic_route(task, preferred_alias=preferred_alias, intent_hint=intent_hint)
    threshold = float(config.routing_config().get("heuristic_confidence_threshold", 0.84))
    if heuristic.confidence >= threshold:
        return heuristic

    model_decision = _model_route(task, preferred_alias=preferred_alias, intent_hint=intent_hint)
    if model_decision is not None:
        return model_decision
    return heuristic

def task_to_mode(task: str) -> str:
    decision = route_task(task)
    if decision.mode == "ask":
        return f"ask {decision.alias or select_alias_for_candidates(task, ['cdx-fast', 'gmn-fast'])}"
    return decision.mode


_DEFAULT_TIER_ROUTING: dict[str, str] = {
    "cheap":    "light",
    "balanced": "medium",
    "deep":     "heavy",
    "extreme":  "heavy",
}

_DEFAULT_TIER_AGENTS: dict[str, list[str]] = {
    "light":  ["cdx-fast", "gmn-fast"],
    "medium": ["cdx-fast", "gmn-pro"],
    "heavy":  ["cdx-deep", "gmn-pro"],
}


def classify_to_tier(task: str) -> str:
    """Map task → light | medium | heavy tier for Protégé-style model routing."""
    task_class = classify(task)
    routing = config.tier_routing_config()
    return routing.get(task_class) or _DEFAULT_TIER_ROUTING.get(task_class, "medium")


def agents_for_tier(tier: str) -> list[str]:
    """Return the ordered agent alias list for a given tier."""
    tiers = config.tier_config()
    tier_cfg = tiers.get(tier)
    if isinstance(tier_cfg, dict):
        agents = tier_cfg.get("agents", [])
        if agents:
            return list(agents)
    return list(_DEFAULT_TIER_AGENTS.get(tier, ["cdx-fast", "gmn-pro"]))

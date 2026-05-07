"""Heuristic and Semantic task classifier with Auto-Decomposition support."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from orchestra import config
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

def classify(task: str) -> str:
    """Return cheap | balanced | deep | extreme (auto-plan)."""
    # 🛡 NEW: Auto-Decomposition Trigger
    if len(task) > 1500 or task.count("\n") > 20:
        return "extreme"

    policy = {"keywords": {"cheap": ["hello", "selam"], "deep": ["research", "araştır", "plan", "analyze"]}}
    task_low = task.lower()
    if any(k in task_low for k in policy["keywords"]["cheap"]): return "cheap"
    if any(k in task_low for k in policy["keywords"]["deep"]): return "deep"
    return "balanced"

def task_to_mode(task: str) -> str:
    task_class = classify(task)
    if task_class == "extreme":
        return "planned"
    if task_class == "cheap":
        return f"ask {select_alias_for_candidates(task, ['cdx-fast', 'gmn-fast'])}"
    if task_class == "deep":
        return "critical"
    return "dual"

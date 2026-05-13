"""Semantic Router: route tasks based on past performance on similar tasks."""
from __future__ import annotations

from typing import List, Dict, Optional
from orchestra.storage.memory import get_memory
from orchestra.storage.db import get_db

def suggest_best_alias(task: str, candidates: List[str]) -> Optional[str]:
    """
    Search past successful tasks in Vector DB and return the alias that
    performed best on those similar tasks.

    Returns the alias with highest aggregated confidence score across
    similar completed runs, filtered to candidates only.
    """
    mem = get_memory()
    similar_tasks = mem.search(task, limit=3)

    if not similar_tasks:
        return None

    conn = get_db()
    alias_performance: Dict[str, float] = {}

    for res in similar_tasks:
        meta = res.get("metadata", {})
        run_id = meta.get("run_id")
        if not run_id:
            continue

        # Use agent_stats from metadata (populated during finalization)
        agent_stats = meta.get("agent_stats", [])
        if agent_stats:
            for stat in agent_stats:
                alias = stat.get("alias")
                if alias not in candidates:
                    continue
                conf = stat.get("confidence", 0.0)
                status = stat.get("status", "unknown").lower()
                # Prefer completed agents; discount failures
                score = conf * (1.2 if status == "completed" else 0.3)
                alias_performance[alias] = alias_performance.get(alias, 0.0) + score
        else:
            # Fallback: query database directly
            rows = conn.execute(
                "SELECT alias, confidence, status FROM agents WHERE run_id = ?", (run_id,)
            ).fetchall()

            for row in rows:
                alias, conf, status = row[0], row[1], row[2]
                if alias not in candidates:
                    continue
                if status != "COMPLETED":
                    continue

                alias_performance[alias] = alias_performance.get(alias, 0.0) + float(conf or 0.0)

    if not alias_performance:
        return None

    # Return alias with highest performance score
    sorted_aliases = sorted(alias_performance.items(), key=lambda x: x[1], reverse=True)
    return sorted_aliases[0][0]


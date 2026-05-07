"""Semantic Router: route tasks based on past performance on similar tasks."""
from __future__ import annotations

from typing import List, Dict, Optional
from orchestra.storage.memory import get_memory
from orchestra.storage.db import get_db

def suggest_best_alias(task: str, candidates: List[str]) -> Optional[str]:
    """
    Search past successful tasks in Vector DB and return the alias that 
    performed best on those similar tasks.
    """
    mem = get_memory()
    # Find similar past task contents in our semantic memory
    # Note: We index tasks during finalize in runner.py (to be added)
    similar_tasks = mem.search(task, limit=3)
    
    if not similar_tasks:
        return None
        
    # We look for the run_id in metadata to check reputation of specific runs
    conn = get_db()
    alias_performance: Dict[str, float] = {}
    
    for res in similar_tasks:
        meta = res.get("metadata", {})
        run_id = meta.get("run_id")
        if not run_id: continue
        
        # Check how agents performed in this specific similar run
        rows = conn.execute(
            "SELECT alias, confidence, status FROM agents WHERE run_id = ?", (run_id,)
        ).fetchall()
        
        for row in rows:
            alias, conf, status = row[0], row[1], row[2]
            if alias not in candidates: continue
            
            score = conf * (1.2 if status == "completed" else 0.5)
            alias_performance[alias] = alias_performance.get(alias, 0) + score

    if not alias_performance:
        return None
        
    # Sort by performance and return the best
    sorted_aliases = sorted(alias_performance.items(), key=lambda x: x[1], reverse=True)
    return sorted_aliases[0][0]


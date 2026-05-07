"""Pattern Catalog: define reusable multi-agent execution topologies."""
from __future__ import annotations

from typing import List, Dict, Any, Callable
from orchestra.engine.runner import run_ask, run_dual, run_critical
from orchestra.models import OrchestraRun

def execute_pattern(name: str, task: str, **kwargs) -> OrchestraRun:
    """Execute a specific orchestration pattern by name."""
    patterns = {
        "ask": run_ask_pattern,
        "dual": run_dual_pattern,
        "critical": run_critical_pattern,
        "consensus": run_consensus_pattern,
    }
    
    handler = patterns.get(name, run_ask_pattern)
    return handler(task, **kwargs)

def run_ask_pattern(task: str, **kwargs) -> OrchestraRun:
    from orchestra.engine.runner import run_ask
    return run_ask(kwargs.get("alias", "cdx-deep"), task, **kwargs)

def run_dual_pattern(task: str, **kwargs) -> OrchestraRun:
    from orchestra.engine.runner import run_dual
    return run_dual(task, **kwargs)

def run_critical_pattern(task: str, **kwargs) -> OrchestraRun:
    from orchestra.engine.runner import run_critical
    return run_critical(task, **kwargs)

def run_consensus_pattern(task: str, **kwargs) -> OrchestraRun:
    """Run 3 agents and pick the best via majority/confidence."""
    # Implementation will involve running 3 agents in parallel and comparing
    # For now, we map to dual for stability during transition
    return run_dual_pattern(task, **kwargs)


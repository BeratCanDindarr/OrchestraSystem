"""Pattern Catalog: define reusable multi-agent execution topologies."""
from __future__ import annotations

from orchestra import config
from orchestra.models import OrchestraRun


def execute_pattern(name: str, task: str, **kwargs) -> OrchestraRun:
    """Execute a specific orchestration pattern by name."""
    patterns = {
        "ask": run_ask_pattern,
        "dual": run_dual_pattern,
        "consensus": run_consensus_pattern,
    }

    handler = patterns.get(name, run_ask_pattern)
    return handler(task, **kwargs)


def run_ask_pattern(task: str, **kwargs) -> OrchestraRun:
    from orchestra.engine.runners import run_ask
    alias = kwargs.pop("alias", None) or config.default_light_alias()
    return run_ask(alias, task, **kwargs)


def run_dual_pattern(task: str, **kwargs) -> OrchestraRun:
    from orchestra.engine.runners import run_dual
    return run_dual(task, **kwargs)


def run_consensus_pattern(task: str, **kwargs) -> OrchestraRun:
    """Two agents argue for and against, a third assesses, the synthesis decides."""
    from orchestra.engine.runners import run_dual
    return run_dual(task, stances=True, **kwargs)

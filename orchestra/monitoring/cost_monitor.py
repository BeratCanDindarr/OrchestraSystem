"""Cost monitoring and reporting for orchestration runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from orchestra.models import OrchestraRun, AgentRun


@dataclass
class AgentCost:
    """Cost breakdown for a single agent."""
    alias: str
    model: str
    estimated_tokens: int
    estimated_cost_usd: float
    status: str


@dataclass
class CostReport:
    """Complete cost report for a run."""
    run_id: str
    total_cost_usd: float
    agent_costs: List[AgentCost]
    num_agents: int
    mode: str


def extract_cost_report(run: OrchestraRun) -> CostReport:
    """Extract cost data from an orchestration run.

    Args:
        run: Completed OrchestraRun instance

    Returns:
        CostReport with aggregated cost information
    """
    agent_costs = []

    for agent in run.agents:
        agent_costs.append(AgentCost(
            alias=agent.alias,
            model=agent.model,
            estimated_tokens=agent.estimated_completion_tokens,
            estimated_cost_usd=agent.estimated_cost_usd,
            status=agent.status.value if agent.status else "unknown",
        ))

    return CostReport(
        run_id=run.run_id,
        total_cost_usd=run.total_cost_usd,
        agent_costs=agent_costs,
        num_agents=len(run.agents),
        mode=run.mode,
    )


def format_cost_report_for_broadcast(report: CostReport) -> dict:
    """Format cost report for WebSocket broadcast.

    Args:
        report: CostReport instance

    Returns:
        Dictionary ready for JSON serialization and broadcast
    """
    agent_breakdown = [
        {
            "alias": cost.alias,
            "model": cost.model,
            "tokens": cost.estimated_tokens,
            "cost_usd": round(cost.estimated_cost_usd, 6),
            "status": cost.status,
        }
        for cost in report.agent_costs
    ]

    return {
        "type": "cost_report",
        "run_id": report.run_id,
        "mode": report.mode,
        "total_cost_usd": round(report.total_cost_usd, 6),
        "num_agents": report.num_agents,
        "agent_breakdown": agent_breakdown,
    }


def broadcast_cost_report(run: OrchestraRun, broadcast_fn) -> None:
    """Extract cost data from run and broadcast it.

    Args:
        run: Completed OrchestraRun instance
        broadcast_fn: Function to call with cost report dict (accepts dict payload)
    """
    if not run or run.total_cost_usd == 0.0:
        return

    report = extract_cost_report(run)
    payload = format_cost_report_for_broadcast(report)
    broadcast_fn(payload)

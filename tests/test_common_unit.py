import pytest
from unittest.mock import MagicMock, patch
from orchestra.engine.runners.common import _estimate_run_cost
from orchestra.models import OrchestraRun, AgentRun




def test_the_run_cost_is_what_costing_charged_each_agent():
    run = OrchestraRun(run_id="r1", mode="dual", task="t")
    run.agents = [AgentRun(alias="kimi-deep", provider="kimi", model="kimi/k3", estimated_cost_usd=0.004),
                  AgentRun(alias="oll-glm", provider="ollama", model="ollama/glm-5.3:cloud", estimated_cost_usd=0.0085),
                  AgentRun(alias="cld-fast", provider="claude", model="claude/sonnet")]   # not charged yet

    assert _estimate_run_cost(run) == pytest.approx(0.0125)

import pytest
from unittest.mock import MagicMock, patch
from orchestra.engine.runners.common import _estimate_run_cost, _alias_to_model_label
from orchestra.models import OrchestraRun, AgentRun

def test_estimate_run_cost():
    run = OrchestraRun(run_id="r1", mode="ask", task="t")
    a1 = AgentRun(alias="cdx-deep", provider="p1", model="m1")
    a1.estimated_completion_tokens = 1000
    run.agents = [a1]
    
    with patch("orchestra.config.price_per_1k_tokens", return_value=0.03):
        cost = _estimate_run_cost(run)
        assert cost == 0.03

def test_alias_to_model_label():
    assert _alias_to_model_label("cdx-deep") == "gpt-5.4/xhigh"
    assert _alias_to_model_label("gmn-pro") == "gemini/pro"
    assert _alias_to_model_label("unknown") == "gpt-5.4/medium"

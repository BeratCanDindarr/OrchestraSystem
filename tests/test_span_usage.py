"""Tests for OTel token/cost span emission (_emit_span_usage)."""
from unittest.mock import MagicMock, patch

from orchestra.models import OrchestraRun, AgentRun
from orchestra.engine.runners import common


def _run_with_agent(alias="x", tokens=42, cost=0.5):
    run = OrchestraRun(mode="ask", task="t")
    agent = AgentRun(alias=alias, provider="openai", model="gpt")
    agent.estimated_completion_tokens = tokens
    agent.estimated_cost_usd = cost
    run.agents = [agent]
    return run


def test_emit_span_usage_sets_tokens_and_cost():
    tracer = MagicMock()
    span = MagicMock()
    run = _run_with_agent("x", 42, 0.5)
    with patch.object(common, "_get_tracer", return_value=tracer):
        common._emit_span_usage(span, run, "x")
    tracer.set_tokens.assert_called_once_with(span, 0, 42)
    tracer.set_cost.assert_called_once_with(span, 0.5)


def test_emit_span_usage_noop_on_none_span():
    tracer = MagicMock()
    with patch.object(common, "_get_tracer", return_value=tracer):
        common._emit_span_usage(None, _run_with_agent(), "x")
    tracer.set_tokens.assert_not_called()
    tracer.set_cost.assert_not_called()


def test_emit_span_usage_noop_on_missing_alias():
    tracer = MagicMock()
    span = MagicMock()
    with patch.object(common, "_get_tracer", return_value=tracer):
        common._emit_span_usage(span, _run_with_agent("x"), "not-present")
    tracer.set_tokens.assert_not_called()
    tracer.set_cost.assert_not_called()

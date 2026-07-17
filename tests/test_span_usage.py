"""Tests for OTel token/cost span emission (_emit_span_usage).

Regression: emission must aggregate across ALL run agents, so it works for the
multi-agent dual/critical modes (whose spans carry a label, not a real alias) —
not just single-agent ask.
"""
from unittest.mock import MagicMock, patch

from orchestra.models import OrchestraRun, AgentRun
from orchestra.engine.runners import common


def _agent(alias, tokens, cost):
    a = AgentRun(alias=alias, provider="openai", model="gpt")
    a.estimated_completion_tokens = tokens
    a.estimated_cost_usd = cost
    return a


def _run(*agents):
    run = OrchestraRun(mode="ask", task="t")
    run.agents = list(agents)
    return run


def test_emit_span_usage_single_agent():
    tracer, span = MagicMock(), MagicMock()
    with patch.object(common, "_get_tracer", return_value=tracer):
        common._emit_span_usage(span, _run(_agent("x", 42, 0.5)))
    tracer.set_tokens.assert_called_once_with(span, 0, 42)
    tracer.set_cost.assert_called_once_with(span, 0.5)


def test_emit_span_usage_aggregates_multiple_agents():
    """dual/critical: two agents → span carries the SUM (previously a silent no-op)."""
    tracer, span = MagicMock(), MagicMock()
    run = _run(_agent("a", 100, 0.30), _agent("b", 50, 0.20))
    with patch.object(common, "_get_tracer", return_value=tracer):
        common._emit_span_usage(span, run)
    tracer.set_tokens.assert_called_once_with(span, 0, 150)
    args = tracer.set_cost.call_args[0]
    assert args[0] is span and abs(args[1] - 0.5) < 1e-9


def test_emit_span_usage_noop_on_none_span():
    tracer = MagicMock()
    with patch.object(common, "_get_tracer", return_value=tracer):
        common._emit_span_usage(None, _run(_agent("x", 1, 0.1)))
    tracer.set_tokens.assert_not_called()
    tracer.set_cost.assert_not_called()


def test_emit_span_usage_noop_on_no_agents():
    tracer, span = MagicMock(), MagicMock()
    with patch.object(common, "_get_tracer", return_value=tracer):
        common._emit_span_usage(span, _run())
    tracer.set_tokens.assert_not_called()
    tracer.set_cost.assert_not_called()

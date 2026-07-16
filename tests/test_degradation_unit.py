"""Unit tests for orchestra.engine.degradation (bounded retry + cheaper-alias fallback)."""
from __future__ import annotations

from types import SimpleNamespace

from orchestra import config
from orchestra.engine import degradation
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus


def _agent(output="", status=AgentStatus.COMPLETED, soft_failed=False):
    return AgentRun(
        alias="a", provider="p", model="m",
        status=status, stdout_log=output, soft_failed=soft_failed,
    )


def _ok(output="ok"):
    r = OrchestraRun(mode="ask", task="t")
    r.status = RunStatus.COMPLETED
    r.agents = [_agent(output)]
    return r


def _soft_fail(output="soft"):
    r = OrchestraRun(mode="ask", task="t")
    r.status = RunStatus.COMPLETED
    r.agents = [_agent(output, soft_failed=True)]
    return r


def _terminal_fail():
    r = OrchestraRun(mode="ask", task="t")
    r.status = RunStatus.FAILED
    r.agents = [_agent(status=AgentStatus.FAILED)]
    r.failure = SimpleNamespace(retryable=False)   # e.g. budget/policy
    return r


def test_is_degrading_default_false():
    assert degradation.is_degrading() is False


def test_cheaper_alias_steps_down_within_provider():
    assert degradation.cheaper_alias("cdx-deep") == "cdx-fast"
    assert degradation.cheaper_alias("gmn-pro") == "gmn-fast"
    assert degradation.cheaper_alias("cdx-fast") is None      # already cheapest in provider
    assert degradation.cheaper_alias("nonexistent") is None


def test_degrade_noop_on_success():
    base = _ok()
    calls = []
    out = degradation.degrade(
        "cdx-deep", "t", base,
        run_ask_fn=lambda a, p, **k: calls.append(a) or _ok(),
        max_attempts=3, sleep_fn=lambda s: None,
    )
    assert out is base and calls == []


def test_degrade_noop_on_terminal_failure():
    base = _terminal_fail()
    calls = []
    out = degradation.degrade(
        "cdx-deep", "t", base,
        run_ask_fn=lambda a, p, **k: calls.append(a) or _ok(),
        max_attempts=3, sleep_fn=lambda s: None,
    )
    assert out is base and calls == []               # budget/policy is not retried


def test_degrade_retries_soft_fail_then_recovers():
    base = _soft_fail()
    seq = iter([_ok("recovered")])
    out = degradation.degrade(
        "cdx-deep", "t", base,
        run_ask_fn=lambda a, p, **k: next(seq),
        max_attempts=3, sleep_fn=lambda s: None,
    )
    assert out.agents[0].stdout_log == "recovered"
    assert degradation.is_degrading() is False        # guard cleared


def test_degrade_exhausts_attempts_when_never_recovers():
    base = _soft_fail()
    calls = []
    out = degradation.degrade(
        "cdx-deep", "t", base,
        run_ask_fn=lambda a, p, **k: calls.append(a) or _soft_fail(),
        max_attempts=2, sleep_fn=lambda s: None,
    )
    assert len(calls) == 2 and out.agents[0].soft_failed


def test_degrade_to_cheaper_steps_down_alias():
    base = _soft_fail()
    calls = []
    degradation.degrade(
        "cdx-deep", "t", base,
        run_ask_fn=lambda a, p, **k: calls.append(a) or _soft_fail(),
        max_attempts=2, degrade_to_cheaper=True, sleep_fn=lambda s: None,
    )
    assert calls == ["cdx-fast", "cdx-fast"]           # cdx-deep -> cheaper sibling


def test_degrade_returns_base_when_already_degrading():
    base = _soft_fail()
    token = degradation._in_degrade.set(True)
    try:
        out = degradation.degrade(
            "cdx-deep", "t", base,
            run_ask_fn=lambda a, p, **k: _ok(), max_attempts=2, sleep_fn=lambda s: None,
        )
        assert out is base
    finally:
        degradation._in_degrade.reset(token)


def test_retry_config_defaults():
    assert config.retry_enabled() is False
    assert config.retry_degrade_to_cheaper() is False
    assert config.retry_max_attempts() == 3

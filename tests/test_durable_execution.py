# tests/test_durable_execution.py
import signal
import sqlite3

import pytest

from orchestra.engine.durable_execution import resume_ask, resume_dual, suspend_on_signal
from orchestra.models import AgentRun, OrchestraRun, RunStatus
from orchestra.state import ApprovalState


def _mem_db():
    return sqlite3.connect(":memory:")


def _run(mode):
    r = OrchestraRun(mode=mode, task="t")
    r.status = RunStatus.RUNNING
    return r


def test_resume_dual_continues_from_running_checkpoint():
    run = _run("dual")
    calls = {"synth": 0, "final": 0}
    resume_dual(
        run,
        {"label": "verification_complete", "status": "running"},
        synthesize_fn=lambda r, task, **kw: calls.__setitem__("synth", calls["synth"] + 1),
        finalize_fn=lambda r: calls.__setitem__("final", calls["final"] + 1),
        load_paused_fn=lambda c, rid: None,
        delete_paused_fn=lambda c, rid: None,
        get_db_fn=_mem_db,
    )
    assert calls == {"synth": 1, "final": 1}
    assert run.approval_state == ApprovalState.RESUMED


def test_resume_ask_finalizes_but_never_synthesizes():
    run = _run("ask")
    called = []
    resume_ask(
        run,
        {"label": "ask_completion", "status": "running"},
        finalize_fn=lambda r: called.append("final"),
        load_paused_fn=lambda c, rid: None,
        delete_paused_fn=lambda c, rid: None,
        get_db_fn=_mem_db,
    )
    assert called == ["final"]


def test_resume_waits_at_pending_approval_gate():
    run = _run("ask")
    run.approval_state = ApprovalState.PENDING
    called = []
    out = resume_ask(
        run,
        {"label": "approval_gate", "status": "waiting_approval"},
        finalize_fn=lambda r: called.append("final"),
        load_paused_fn=lambda c, rid: None,
        delete_paused_fn=lambda c, rid: None,
        get_db_fn=_mem_db,
    )
    assert out.status == RunStatus.WAITING_APPROVAL
    assert called == []


def test_resume_dual_restores_paused_state():
    run = _run("dual")
    paused = _run("dual")
    agent = AgentRun(alias="x", provider="p", model="m")
    agent.stdout_log = "hi"
    paused.agents = [agent]
    paused.reviews = [{"r": 1}]
    paused.summary = "SUMMARY"
    deleted = []
    out = resume_dual(
        run,
        {"label": "round1_complete", "status": "running"},
        synthesize_fn=lambda r, task, **kw: None,
        finalize_fn=lambda r: None,
        load_paused_fn=lambda c, rid: paused,
        delete_paused_fn=lambda c, rid: deleted.append(rid),
        get_db_fn=_mem_db,
    )
    assert out.summary == "SUMMARY"
    assert len(out.agents) == 1 and out.agents[0].alias == "x"
    assert out.reviews == [{"r": 1}]
    assert deleted == [run.run_id]


def test_suspend_on_signal_noop_when_disabled():
    run = OrchestraRun(mode="dual", task="t")
    before = signal.getsignal(signal.SIGINT)
    with suspend_on_signal(
        run,
        suspend_fn=lambda c, r, **k: None,
        get_db_fn=_mem_db,
        enabled_fn=lambda: False,
    ):
        assert signal.getsignal(signal.SIGINT) is before
    assert signal.getsignal(signal.SIGINT) is before


def test_suspend_on_signal_persists_and_restores_when_enabled():
    run = OrchestraRun(mode="dual", task="t")
    run.status = RunStatus.RUNNING
    seen = {}

    def suspend_fn(conn, r, paused_by="system"):
        seen["by"] = paused_by
        seen["status"] = r.status

    before = signal.getsignal(signal.SIGINT)
    with suspend_on_signal(
        run, suspend_fn=suspend_fn, get_db_fn=_mem_db, enabled_fn=lambda: True
    ):
        handler = signal.getsignal(signal.SIGINT)
        assert handler is not before
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)
    assert seen["by"] == "signal"
    assert seen["status"] == RunStatus.WAITING_APPROVAL
    assert run.approval_state == ApprovalState.RESUMED
    assert signal.getsignal(signal.SIGINT) is before

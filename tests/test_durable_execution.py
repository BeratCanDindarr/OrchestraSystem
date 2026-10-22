# tests/test_durable_execution.py
import signal
import sqlite3

import pytest

from orchestra.engine.durable_execution import resume_ask, resume_dual, suspend_on_signal
from orchestra.engine import artifacts
from orchestra.engine.runners import resume_runner
from orchestra.engine.state_suspension import suspend_run as _suspend_run, delete_paused_run as _delete_paused_run
from orchestra.storage.db import get_db
import orchestra.engine.runners.common as _common
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus
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


def test_resume_runner_routes_dual_hard_crash(monkeypatch):
    run = OrchestraRun(mode="dual", task="hard crash task")
    run.status = RunStatus.RUNNING
    agent = AgentRun(alias="a", provider="p", model="m")
    agent.status = AgentStatus.COMPLETED
    agent.stdout_log = "partial"
    run.agents = [agent]
    artifacts.write_manifest(run)
    artifacts.write_checkpoint(run, "verification_complete")

    calls = []
    monkeypatch.setattr(
        resume_runner, "_synthesize_if_possible", lambda r, task, **kw: calls.append("synth")
    )
    monkeypatch.setattr(resume_runner, "_finalize", lambda r: calls.append("final"))

    out = resume_runner.resume_run(run.run_id)
    assert calls == ["synth", "final"]
    assert out.approval_state == ApprovalState.RESUMED


def test_run_ask_gate_off_leaves_signals_untouched(monkeypatch):
    before = signal.getsignal(signal.SIGINT)

    def fake_parallel(run, agents, **kwargs):
        alias, _prompt = agents[0]
        a = AgentRun(alias=alias, provider="p", model="m")
        a.status = AgentStatus.COMPLETED
        a.stdout_log = "ok"
        a.estimated_cost_usd = 0.0
        run.agents.append(a)
        return [a]

    monkeypatch.setattr(_common, "run_parallel", fake_parallel)
    monkeypatch.setattr("orchestra.config.durable_execution_enabled", lambda: False)

    run = _common._run_ask_impl("cdx-deep", "hello world")
    assert signal.getsignal(signal.SIGINT) is before
    assert run.status in (RunStatus.COMPLETED, RunStatus.WAITING_APPROVAL)


def test_signal_suspend_then_db_resume_dual(monkeypatch):
    """End-to-end: signal-suspend → paused_runs persist → DB-branch resume for dual mode.

    Simulates the state that the signal handler in suspend_on_signal leaves behind:
    status=WAITING_APPROVAL, approval_state=RESUMED, one COMPLETED agent. Then
    calls the real suspend_run to persist to paused_runs and resume_run to confirm
    the DB branch is taken (approval_approved checkpoint, synthesize+finalize,
    approval_state=APPROVED, paused_runs row deleted).
    """
    # Build a dual run as the signal handler would leave it.
    run = OrchestraRun(mode="dual", task="signal suspend test task")
    run.status = RunStatus.WAITING_APPROVAL
    run.approval_state = ApprovalState.RESUMED
    agent = AgentRun(alias="a", provider="p", model="m")
    agent.status = AgentStatus.COMPLETED
    agent.stdout_log = "agent output after signal"
    run.agents = [agent]
    run.summary = "partial summary"
    artifacts.write_manifest(run)

    # Persist to paused_runs using the real DB.
    conn = get_db()
    _suspend_run(conn, run, paused_by="signal")
    conn.close()

    # Monkeypatch synthesize and finalize to record calls without side-effects.
    calls = []
    monkeypatch.setattr(resume_runner, "_synthesize_if_possible", lambda r, task, **kw: calls.append("synth"))
    monkeypatch.setattr(resume_runner, "_finalize", lambda r: calls.append("final"))

    # Resume via the public entry point (default approve).
    out = resume_runner.resume_run(run.run_id)

    # Assertions: DB branch taken, not checkpoint fallback.
    assert calls == ["synth", "final"], f"expected DB-branch synth+final, got {calls}"
    assert out.approval_state == ApprovalState.APPROVED, f"expected APPROVED, got {out.approval_state}"

    # paused_runs row must be deleted after successful resume.
    verify_conn = get_db()
    row = verify_conn.execute(
        "SELECT run_id FROM paused_runs WHERE run_id = ?", (run.run_id,)
    ).fetchone()
    verify_conn.close()
    assert row is None, "paused_runs row should be deleted after DB-branch resume"

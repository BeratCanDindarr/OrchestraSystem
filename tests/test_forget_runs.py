"""Erasing a run takes everything that belongs to it and nothing else."""
from __future__ import annotations

from orchestra import config, service
from orchestra.engine import artifacts
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus
from orchestra.storage import db as db_module


def _stored(task: str, status: RunStatus = RunStatus.COMPLETED, parent: str | None = None) -> OrchestraRun:
    run = OrchestraRun(mode="ask", task=task, parent_run_id=parent)
    run.agents.append(AgentRun(alias="cld-deep", provider="test", model="test/model", status=AgentStatus.COMPLETED))
    run.status = status
    artifacts.write_manifest(run)
    artifacts.write_agent_log(run.run_id, "cld-deep", "an answer")
    db_module.upsert_run(run.to_manifest())
    return run


def _on_disk(run_id: str) -> bool:
    # not artifacts.run_dir: asking for a run's folder creates it
    return (config.artifact_root() / run_id).exists()


def _in_db(run_id: str) -> bool:
    connection = db_module.get_db()
    try:
        return connection.execute("SELECT COUNT(*) FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0] > 0
    finally:
        connection.close()


def test_a_run_goes_with_its_steps_and_its_rows():
    run = _stored("forget: the run")
    step = _stored("forget: its verifier", parent=run.run_id)
    other = _stored("forget: another run")

    result = service.forget_runs([run.run_id])

    assert sorted(result["forgotten"]) == sorted([run.run_id, step.run_id])
    assert not _on_disk(run.run_id) and not _on_disk(step.run_id)
    assert not _in_db(run.run_id) and not _in_db(step.run_id)
    assert _on_disk(other.run_id) and _in_db(other.run_id)


def test_a_running_run_is_left_alone():
    run = _stored("forget: still going", status=RunStatus.RUNNING)

    result = service.forget_runs([run.run_id, "nope0000"])

    assert result["forgotten"] == []
    assert {item["reason"] for item in result["skipped"]} == {"still running: cancel it first", "not found"}
    assert _on_disk(run.run_id)

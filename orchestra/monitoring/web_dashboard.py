import logging
from typing import Any, Callable, Optional

from orchestra import config
from orchestra.storage.db import get_db

logger = logging.getLogger(__name__)


def list_paused_runs(conn: Any = None) -> list[dict]:
    """Return runs awaiting human approval (the HITL queue). Empty on error / no table."""
    close = conn is None
    if conn is None:
        conn = get_db()
    try:
        rows = conn.execute(
            "SELECT run_id, paused_at, paused_by, status FROM paused_runs "
            "WHERE status = 'pending' ORDER BY paused_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        logger.warning("list_paused_runs failed", exc_info=True)
        return []
    finally:
        if close:
            conn.close()


# Event types that indicate an incident worth surfacing on the dashboard.
_INCIDENT_EVENTS = ("run_failed", "agent_failed", "budget_exceeded", "validation_failed", "error_logged")


def _events_conn():
    """A connection to the database the runs append their events to.

    That is the project database (see storage/db.get_event_log), not the file
    `[eventlog] db_path` names: nothing writes there, and the panels showed nothing.
    """
    return get_db()


def count_incidents(conn: Any) -> dict:
    """Count incident-type events grouped by type from an events-table connection.

    Uses events already emitted by the run path (no new instrumentation). Empty on error.
    """
    try:
        placeholders = ",".join("?" for _ in _INCIDENT_EVENTS)
        rows = conn.execute(
            f"SELECT event_type, COUNT(*) AS c FROM events "
            f"WHERE event_type IN ({placeholders}) GROUP BY event_type",
            _INCIDENT_EVENTS,
        ).fetchall()
    except Exception as exc:
        # A missing events table is benign on a fresh system (no runs yet) — the
        # dashboard polls this endpoint constantly, so keep it quiet (debug, no traceback).
        logger.debug("count_incidents skipped: %s", exc)
        return {}
    out: dict[str, int] = {}
    for row in rows:
        etype = row["event_type"] if not isinstance(row, tuple) else row[0]
        count = row["c"] if not isinstance(row, tuple) else row[1]
        out[etype] = count
    return out


def compute_budget(conn: Any = None) -> dict:
    """Total spend across runs vs the configured dashboard budget.

    ``remaining`` is None when no budget is set (0 = unlimited). Empty/zero on error.
    """
    close = conn is None
    if conn is None:
        conn = get_db()
    try:
        row = conn.execute("SELECT COALESCE(SUM(total_cost_usd), 0.0) AS s FROM runs").fetchone()
        spent = float(row["s"] if not isinstance(row, tuple) else row[0])
    except Exception:
        logger.warning("compute_budget failed", exc_info=True)
        spent = 0.0
    finally:
        if close:
            conn.close()
    budget = config.dashboard_budget_usd()
    remaining = (budget - spent) if budget > 0 else None
    return {"spent": spent, "budget": budget if budget > 0 else None, "remaining": remaining}


def compute_metrics(conn: Any = None) -> dict:
    """Aggregate run metrics over the runs table for the observability dashboard.

    Returns counts by status/mode, total & average cost, and failure rate. Never raises;
    returns a zeroed summary on error. conn is injectable for testing.
    """
    empty = {
        "total_runs": 0, "by_status": {}, "by_mode": {},
        "total_cost_usd": 0.0, "avg_cost_usd": 0.0, "failure_rate": 0.0,
    }
    close = conn is None
    if conn is None:
        conn = get_db()
    try:
        rows = conn.execute("SELECT status, mode, total_cost_usd FROM runs").fetchall()
    except Exception:
        logger.warning("compute_metrics failed", exc_info=True)
        return empty
    finally:
        if close:
            conn.close()

    total = len(rows)
    by_status: dict[str, int] = {}
    by_mode: dict[str, int] = {}
    cost = 0.0
    for row in rows:
        status = row["status"] if not isinstance(row, tuple) else row[0]
        mode = row["mode"] if not isinstance(row, tuple) else row[1]
        row_cost = row["total_cost_usd"] if not isinstance(row, tuple) else row[2]
        by_status[status] = by_status.get(status, 0) + 1
        by_mode[mode] = by_mode.get(mode, 0) + 1
        cost += float(row_cost or 0.0)
    failed = by_status.get("failed", 0)
    return {
        "total_runs": total,
        "by_status": by_status,
        "by_mode": by_mode,
        "total_cost_usd": round(cost, 4),
        "avg_cost_usd": round(cost / total, 4) if total else 0.0,
        "failure_rate": round(failed / total, 4) if total else 0.0,
    }


def compute_eval_summary(conn: Any = None) -> dict:
    """Summarize eval outcomes from the eval_runs table (PASS/FAIL), overall and by mode.

    Populated when the eval harness is enabled (eval.auto_enabled). Empty summary on error.
    """
    empty = {"total": 0, "passed": 0, "pass_rate": 0.0, "by_mode": {}}
    close = conn is None
    if conn is None:
        conn = get_db()
    try:
        rows = conn.execute("SELECT mode, status FROM eval_runs").fetchall()
    except Exception:
        logger.warning("compute_eval_summary failed", exc_info=True)
        return empty
    finally:
        if close:
            conn.close()

    total = len(rows)
    passed = 0
    by_mode: dict[str, dict] = {}
    for row in rows:
        mode = row["mode"] if not isinstance(row, tuple) else row[0]
        status = row["status"] if not isinstance(row, tuple) else row[1]
        bucket = by_mode.setdefault(mode, {"passed": 0, "total": 0})
        bucket["total"] += 1
        if status == "PASS":
            passed += 1
            bucket["passed"] += 1
    for bucket in by_mode.values():
        bucket["pass_rate"] = round(bucket["passed"] / bucket["total"], 4) if bucket["total"] else 0.0
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "by_mode": by_mode,
    }


def apply_approval(run_id: str, decision: str, *, resume_fn: Optional[Callable[[str, str], Any]] = None) -> dict:
    """Approve or reject a paused run by resuming it with the decision.

    ``decision`` is "approve" or "reject". resume_fn is injectable for testing; by default it
    resumes the run via the resume_runner. Returns a status dict (never raises).
    """
    if decision not in ("approve", "reject"):
        return {"ok": False, "error": "decision must be 'approve' or 'reject'"}
    if not run_id:
        return {"ok": False, "error": "run_id is required"}
    if resume_fn is None:
        def resume_fn(rid: str, dec: str) -> Any:
            from orchestra.engine.runners.resume_runner import resume_run
            return resume_run(rid, approval_decision=dec)
    try:
        resume_fn(run_id, decision)
        return {"ok": True, "run_id": run_id, "decision": decision}
    except Exception as exc:
        logger.warning("approval resume failed for %s: %s", run_id, exc, exc_info=True)
        return {"ok": False, "run_id": run_id, "error": str(exc)}

import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
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
    """Open a read connection to the event log DB (separate SQLite file from runs)."""
    import sqlite3
    from orchestra import config as _config
    conn = sqlite3.connect(str(_config.eventlog_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


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
    except Exception:
        logger.warning("count_incidents failed", exc_info=True)
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


_DASHBOARD_HTML = """
<html>
    <head>
        <title>Orchestra Fleet Dashboard</title>
        <style>
            body { font-family: sans-serif; padding: 20px; background: #f0f0f0; }
            table { width: 100%; border-collapse: collapse; background: white; }
            th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
            th { background: #eee; }
            .status-completed { color: green; }
            .status-failed { color: red; }
            .status-running { color: blue; }
            .metrics { margin: 12px 0 20px; display: flex; flex-wrap: wrap; gap: 10px; }
            .metric { background: white; border: 1px solid #ddd; border-radius: 6px;
                      padding: 8px 12px; font-size: 14px; }
            .metric b { color: #222; }
            .metric.incident b { color: #c00; }
        </style>
    </head>
    <body>
        <h1>Orchestra Fleet Dashboard</h1>
        <div id="metrics" class="metrics">Loading metrics…</div>
        <div id="incidents" class="metrics"></div>
        <div id="eval" class="metrics"></div>
        <table id="runs">
            <tr><th>Run ID</th><th>Mode</th><th>Status</th><th>Task</th><th>Cost</th></tr>
        </table>
        <script>
            fetch('/api/metrics').then(r => r.json()).then(m => {
                const el = document.getElementById('metrics');
                const kv = obj => Object.entries(obj || {}).map(([k, v]) => `${k}: ${v}`).join(', ') || '—';
                el.innerHTML = `
                    <span class="metric"><b>${m.total_runs || 0}</b> runs</span>
                    <span class="metric">failure rate <b>${((m.failure_rate || 0) * 100).toFixed(1)}%</b></span>
                    <span class="metric">total cost <b>$${(m.total_cost_usd || 0).toFixed(3)}</b></span>
                    <span class="metric">avg cost <b>$${(m.avg_cost_usd || 0).toFixed(4)}</b></span>
                    <span class="metric">status — ${kv(m.by_status)}</span>
                    <span class="metric">mode — ${kv(m.by_mode)}</span>
                `;
            }).catch(() => { document.getElementById('metrics').textContent = 'metrics unavailable'; });
            fetch('/api/incidents').then(r => r.json()).then(inc => {
                const el = document.getElementById('incidents');
                const entries = Object.entries(inc || {});
                el.innerHTML = entries.length
                    ? entries.map(([k, v]) => `<span class="metric incident">${k}: <b>${v}</b></span>`).join('')
                    : '<span class="metric">no incidents</span>';
            }).catch(() => {});
            fetch('/api/eval').then(r => r.json()).then(e => {
                const el = document.getElementById('eval');
                if (!e.total) { el.innerHTML = '<span class="metric">no evals yet</span>'; return; }
                const modes = Object.entries(e.by_mode || {})
                    .map(([k, v]) => `${k}: ${((v.pass_rate || 0) * 100).toFixed(0)}%`).join(', ');
                el.innerHTML = `
                    <span class="metric">eval pass rate <b>${((e.pass_rate || 0) * 100).toFixed(1)}%</b></span>
                    <span class="metric">${e.passed}/${e.total} passed</span>
                    <span class="metric">by mode — ${modes || '—'}</span>
                `;
            }).catch(() => {});
            fetch('/api/runs').then(r => r.json()).then(data => {
                const table = document.getElementById('runs');
                data.forEach(run => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${run.run_id}</td>
                        <td>${run.mode}</td>
                        <td class="status-${run.status}">${run.status}</td>
                        <td>${(run.task || '').substring(0, 100)}</td>
                        <td>$${(run.total_cost_usd || 0).toFixed(3)}</td>
                    `;
                    table.appendChild(tr);
                });
            });
        </script>
    </body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, obj: Any) -> None:
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode('utf-8'))

    def do_GET(self):
        if self.path == '/api/runs':
            conn = get_db()
            rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 100").fetchall()
            conn.close()
            self._send_json(200, [dict(r) for r in rows])
        elif self.path == '/api/paused':
            self._send_json(200, list_paused_runs())
        elif self.path == '/api/metrics':
            self._send_json(200, compute_metrics())
        elif self.path == '/api/incidents':
            conn = _events_conn()
            try:
                self._send_json(200, count_incidents(conn))
            finally:
                conn.close()
        elif self.path == '/api/eval':
            self._send_json(200, compute_eval_summary())
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(_DASHBOARD_HTML.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != '/api/approve':
            self.send_response(404)
            self.end_headers()
            return
        # Guarded off by default: the dashboard has no auth, so approvals must be opted in.
        if not config.dashboard_approvals_enabled():
            self._send_json(403, {"ok": False, "error": "approvals disabled (set dashboard.approvals_enabled)"})
            return
        length = int(self.headers.get('Content-Length', 0) or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b'{}') if length else {}
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "invalid JSON body"})
            return
        run_id = str(payload.get('run_id', ''))
        decision = str(payload.get('decision', 'approve'))
        if not run_id or decision not in ("approve", "reject"):
            self._send_json(400, {"ok": False, "error": "run_id and a valid decision are required"})
            return
        # Resume can take a while; dispatch it in the background and acknowledge immediately.
        threading.Thread(target=lambda: apply_approval(run_id, decision), daemon=True).start()
        self._send_json(202, {"ok": True, "run_id": run_id, "decision": decision, "status": "dispatched"})


def start_web_dashboard(port=8080):
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    logger.info(f"Starting web dashboard on port {port}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server

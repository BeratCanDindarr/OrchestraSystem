"""Safe speculative branching/worktree planning for Orchestra."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _root(orchestra_root: Path) -> Path:
    path = orchestra_root / "speculation"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plans_root(orchestra_root: Path) -> Path:
    path = _root(orchestra_root) / "plans"
    path.mkdir(parents=True, exist_ok=True)
    return path


def worktrees_root(orchestra_root: Path) -> Path:
    path = _root(orchestra_root) / "worktrees"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manifest_path(orchestra_root: Path, plan_id: str) -> Path:
    return plans_root(orchestra_root) / f"{plan_id}.json"


def _build_hypotheses(orchestra_root: Path, plan_id: str, names: list[str], repo_root: str, base_ref: str) -> list[dict]:
    results: list[dict] = []
    for index, name in enumerate(names):
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-") or f"hypothesis-{index+1}"
        branch = f"orch-{plan_id[:6]}-{slug}"
        worktree = worktrees_root(orchestra_root) / plan_id / slug
        results.append(
            {
                "name": name,
                "slug": slug,
                "branch": branch,
                "worktree_path": str(worktree),
                "repo_root": repo_root,
                "base_ref": base_ref,
                "status": "planned",
            }
        )
    return results


def create_speculative_plan(
    connection: sqlite3.Connection,
    orchestra_root: Path,
    *,
    repo_root: str,
    task: str,
    base_ref: str,
    hypothesis_names: list[str],
) -> dict:
    plan_id = uuid.uuid4().hex[:10]
    created_at = _now()
    manifest = {
        "plan_id": plan_id,
        "repo_root": repo_root,
        "base_ref": base_ref,
        "task": task,
        "status": "planned",
        "created_at": created_at,
        "updated_at": created_at,
        "hypotheses": _build_hypotheses(orchestra_root, plan_id, hypothesis_names, repo_root, base_ref),
        "evaluation": {
            "compile_check": "pending",
            "tests": "pending",
            "selection": "pending",
            "selected_branch": "",
            "notes": "",
        },
    }
    manifest_path = _manifest_path(orchestra_root, plan_id)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    connection.execute(
        """
        INSERT INTO speculative_plans (
            plan_id, repo_root, base_ref, task, status, manifest_path, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, 'planned', ?, ?, ?)
        """,
        (plan_id, repo_root, base_ref, task, str(manifest_path), created_at, created_at),
    )
    return manifest


def execute_speculative_plan(
    connection: sqlite3.Connection,
    orchestra_root: Path,
    plan_id: str,
    mode: str = "ask",
) -> dict:
    """Execute a speculative plan by running agents in each hypothesis worktree.

    For each hypothesis, changes into the worktree directory and runs an agent
    with the plan's task. After all hypotheses complete, compares results and
    selects a winner (highest confidence; tie goes to shortest stdout_log).

    If worktrees are missing, calls prepare_speculative_worktrees first.
    """
    from orchestra.engine.runners import run_ask
    from orchestra.models import RunStatus

    if mode not in {"ask", "fast"}:
        raise ValueError(f"Invalid mode: {mode}. Must be 'ask' or 'fast'.")

    manifest = get_speculative_plan(connection, plan_id)
    manifest_path = Path(
        connection.execute(
            "SELECT manifest_path FROM speculative_plans WHERE plan_id = ?",
            (plan_id,),
        ).fetchone()[0]
    )

    # Ensure worktrees are prepared
    for hypothesis in manifest["hypotheses"]:
        worktree_path = Path(hypothesis["worktree_path"])
        if not worktree_path.exists():
            manifest = prepare_speculative_worktrees(connection, plan_id)
            break

    task = manifest["task"]
    from orchestra import config as _config
    alias = str(_config.personas_config().get("implementer") or "cdx-deep") if mode == "ask" else _config.default_light_alias()

    for hypothesis in manifest["hypotheses"]:
        worktree_path = Path(hypothesis["worktree_path"])
        try:
            original_cwd = os.getcwd()
            os.chdir(str(worktree_path))
            try:
                run = run_ask(
                    alias,
                    task,
                    show_live=False,
                    emit_console=False,
                    install_signal_handlers=False,
                )
            finally:
                os.chdir(original_cwd)
        except Exception as exc:
            hypothesis["status"] = "failed"
            hypothesis["error"] = str(exc)
            hypothesis["stdout_log"] = ""
            hypothesis["confidence"] = 0.0
            continue

        if run.agents:
            agent = run.agents[0]
            hypothesis["stdout_log"] = agent.stdout_log or ""
            hypothesis["confidence"] = agent.confidence
        else:
            hypothesis["stdout_log"] = ""
            hypothesis["confidence"] = 0.0

        if run.status == RunStatus.COMPLETED:
            hypothesis["status"] = "completed"
        else:
            hypothesis["status"] = "failed"

    # Compare results and select winner
    eligible = [h for h in manifest["hypotheses"] if h.get("status") == "completed"]
    winner = None
    if eligible:
        # Highest confidence wins; on tie, shortest stdout_log wins
        eligible.sort(key=lambda h: (-h.get("confidence", 0.0), len(h.get("stdout_log", ""))))
        winner = eligible[0]

    if winner:
        manifest["evaluation"]["selection"] = "winner_selected"
        manifest["evaluation"]["selected_branch"] = winner["branch"]
        manifest["evaluation"]["compile_check"] = "passed"
        manifest["evaluation"]["tests"] = "passed"
        notes: list[str] = []
        for h in manifest["hypotheses"]:
            notes.append(
                f"{h['name']}: confidence={h.get('confidence', 0.0):.2f}, status={h.get('status', 'unknown')}"
            )
        notes.append(f"Winner: {winner['name']} ({winner['branch']})")
        manifest["evaluation"]["notes"] = "\n".join(notes)
    else:
        manifest["evaluation"]["selection"] = "failed"
        manifest["evaluation"]["selected_branch"] = ""
        manifest["evaluation"]["compile_check"] = "failed"
        manifest["evaluation"]["tests"] = "failed"
        notes = []
        for h in manifest["hypotheses"]:
            notes.append(
                f"{h['name']}: confidence={h.get('confidence', 0.0):.2f}, status={h.get('status', 'unknown')}"
            )
        manifest["evaluation"]["notes"] = "\n".join(notes)

    manifest["status"] = "completed"
    manifest["updated_at"] = _now()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    connection.execute(
        "UPDATE speculative_plans SET status = 'completed', updated_at = ? WHERE plan_id = ?",
        (manifest["updated_at"], plan_id),
    )
    return manifest


def list_speculative_plans(connection: sqlite3.Connection, status: str | None = None) -> list[dict]:
    query = [
        """
        SELECT plan_id, repo_root, base_ref, task, status, manifest_path, created_at, updated_at
        FROM speculative_plans
        WHERE 1 = 1
        """
    ]
    params: list[object] = []
    if status:
        query.append("AND status = ?")
        params.append(status)
    query.append("ORDER BY created_at DESC")
    rows = connection.execute(" ".join(query), params).fetchall()
    return [dict(row) for row in rows]


def get_speculative_plan(connection: sqlite3.Connection, plan_id: str) -> dict:
    row = connection.execute(
        """
        SELECT plan_id, repo_root, base_ref, task, status, manifest_path, created_at, updated_at
        FROM speculative_plans
        WHERE plan_id = ?
        """,
        (plan_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"Speculative plan not found: {plan_id}")
    manifest_path = Path(row["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest


def prepare_speculative_worktrees(connection: sqlite3.Connection, plan_id: str) -> dict:
    manifest = get_speculative_plan(connection, plan_id)
    repo_root = manifest["repo_root"]
    manifest_path = Path(
        connection.execute(
            "SELECT manifest_path FROM speculative_plans WHERE plan_id = ?",
            (plan_id,),
        ).fetchone()[0]
    )
    for hypothesis in manifest["hypotheses"]:
        worktree_path = Path(hypothesis["worktree_path"])
        if worktree_path.exists():
            shutil.rmtree(worktree_path)
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "-C",
                repo_root,
                "worktree",
                "add",
                "-b",
                hypothesis["branch"],
                str(worktree_path),
                manifest["base_ref"],
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        hypothesis["status"] = "prepared"

    manifest["status"] = "prepared"
    manifest["updated_at"] = _now()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    connection.execute(
        "UPDATE speculative_plans SET status = 'prepared', updated_at = ? WHERE plan_id = ?",
        (manifest["updated_at"], plan_id),
    )
    return manifest



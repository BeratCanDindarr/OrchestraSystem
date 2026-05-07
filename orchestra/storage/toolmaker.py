"""Safe proposal-based Tool-Maker workflow for Orchestra."""
from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from orchestra import config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _toolmaker_root() -> Path:
    root = config.orchestra_root() / "toolmaker"
    root.mkdir(parents=True, exist_ok=True)
    return root


def proposals_root() -> Path:
    path = _toolmaker_root() / "proposals"
    path.mkdir(parents=True, exist_ok=True)
    return path


def library_root() -> Path:
    path = _toolmaker_root() / "library"
    path.mkdir(parents=True, exist_ok=True)
    return path


def live_root() -> Path:
    path = _toolmaker_root() / "live"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _proposal_dir(proposal_id: str) -> Path:
    path = proposals_root() / proposal_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _live_tool_dir(tool_name: str) -> Path:
    path = live_root() / tool_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_relpath(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe proposal file path: {path}")
    return candidate


def _write_proposal_bundle(proposal_id: str, payload: dict, files: list[dict]) -> str:
    proposal_dir = _proposal_dir(proposal_id)
    (proposal_dir / "proposal.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    files_dir = proposal_dir / "files"
    files_dir.mkdir(exist_ok=True)
    for file_spec in files:
        relpath = _normalize_relpath(file_spec["path"])
        target = files_dir / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(file_spec.get("content", ""), encoding="utf-8")
    return str(proposal_dir)


def _row_to_payload(connection: sqlite3.Connection, proposal_id: str) -> dict:
    row = connection.execute(
        """
        SELECT
            proposal_id, run_id, source_alias, name, description, status,
            test_status, test_command, test_summary, approval_note,
            proposal_dir, library_path, created_at, updated_at, approved_at, promoted_at
        FROM tool_proposals
        WHERE proposal_id = ?
        """,
        (proposal_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"Tool proposal not found: {proposal_id}")

    payload = dict(row)
    proposal_json = Path(payload["proposal_dir"]) / "proposal.json"
    if proposal_json.exists():
        try:
            payload["bundle"] = json.loads(proposal_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload["bundle"] = None
    else:
        payload["bundle"] = None
    return payload


def _install_row_to_payload(connection: sqlite3.Connection, install_id: str) -> dict:
    row = connection.execute(
        """
        SELECT
            install_id, proposal_id, tool_name, live_path, library_path, status, installed_at, updated_at
        FROM tool_installs
        WHERE install_id = ?
        """,
        (install_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"Tool install not found: {install_id}")
    return dict(row)


def create_tool_proposal(
    connection: sqlite3.Connection,
    *,
    name: str,
    description: str,
    files: list[dict] | None = None,
    run_id: str | None = None,
    source_alias: str | None = None,
    test_command: str = "",
) -> dict:
    proposal_id = uuid.uuid4().hex[:10]
    created_at = _now()
    normalized_files = files or []
    payload = {
        "proposal_id": proposal_id,
        "name": name,
        "description": description,
        "run_id": run_id,
        "source_alias": source_alias,
        "status": "pending_approval",
        "test_status": "not_run",
        "test_command": test_command,
        "files": [{"path": item["path"]} for item in normalized_files],
        "created_at": created_at,
    }
    proposal_dir = _write_proposal_bundle(proposal_id, payload, normalized_files)
    connection.execute(
        """
        INSERT INTO tool_proposals (
            proposal_id, run_id, source_alias, name, description, status,
            test_status, test_command, test_summary, approval_note,
            proposal_dir, library_path, created_at, updated_at, approved_at, promoted_at
        )
        VALUES (?, ?, ?, ?, ?, 'pending_approval', 'not_run', ?, '', '', ?, '', ?, ?, NULL, NULL)
        """,
        (
            proposal_id,
            run_id,
            source_alias,
            name,
            description,
            test_command,
            proposal_dir,
            created_at,
            created_at,
        ),
    )
    return _row_to_payload(connection, proposal_id)


def list_tool_proposals(connection: sqlite3.Connection, status: str | None = None) -> list[dict]:
    query = [
        """
        SELECT
            proposal_id, run_id, source_alias, name, description, status,
            test_status, test_command, test_summary, approval_note,
            proposal_dir, library_path, created_at, updated_at, approved_at, promoted_at
        FROM tool_proposals
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


def get_tool_proposal(connection: sqlite3.Connection, proposal_id: str) -> dict:
    return _row_to_payload(connection, proposal_id)


def review_tool_proposal(
    connection: sqlite3.Connection,
    *,
    proposal_id: str,
    approve: bool,
    note: str = "",
) -> dict:
    status = "approved" if approve else "rejected"
    approved_at = _now() if approve else None
    connection.execute(
        """
        UPDATE tool_proposals
        SET status = ?, approval_note = ?, approved_at = ?, updated_at = ?
        WHERE proposal_id = ?
        """,
        (status, note, approved_at, _now(), proposal_id),
    )
    return _row_to_payload(connection, proposal_id)


def record_tool_test_result(
    connection: sqlite3.Connection,
    *,
    proposal_id: str,
    status: str,
    summary: str,
    command: str = "",
) -> dict:
    if status not in {"passed", "failed"}:
        raise ValueError("status must be 'passed' or 'failed'")
    connection.execute(
        """
        UPDATE tool_proposals
        SET test_status = ?, test_summary = ?, test_command = CASE WHEN ? != '' THEN ? ELSE test_command END, updated_at = ?
        WHERE proposal_id = ?
        """,
        (status, summary, command, command, _now(), proposal_id),
    )
    return _row_to_payload(connection, proposal_id)


def promote_tool_proposal(connection: sqlite3.Connection, proposal_id: str) -> dict:
    proposal = _row_to_payload(connection, proposal_id)
    if proposal["status"] != "approved":
        raise ValueError("Proposal must be approved before promotion")
    if proposal["test_status"] != "passed":
        raise ValueError("Proposal must have passed tests before promotion")

    source_dir = Path(proposal["proposal_dir"]) / "files"
    destination = library_root() / proposal_id
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if source_dir.exists():
        shutil.copytree(source_dir, destination / "files", dirs_exist_ok=True)
    proposal_json = Path(proposal["proposal_dir"]) / "proposal.json"
    if proposal_json.exists():
        shutil.copy2(proposal_json, destination / "proposal.json")

    connection.execute(
        """
        UPDATE tool_proposals
        SET status = 'promoted', library_path = ?, promoted_at = ?, updated_at = ?
        WHERE proposal_id = ?
        """,
        (str(destination), _now(), _now(), proposal_id),
    )
    return _row_to_payload(connection, proposal_id)


def install_tool_proposal(connection: sqlite3.Connection, proposal_id: str) -> dict:
    proposal = _row_to_payload(connection, proposal_id)
    if proposal["status"] != "promoted":
        raise ValueError("Proposal must be promoted before live install")
    if not proposal["library_path"]:
        raise ValueError("Proposal has no library path to install")

    tool_name = proposal["name"].strip().replace(" ", "-").lower() or proposal["proposal_id"]
    source_dir = Path(proposal["library_path"])
    if not source_dir.exists():
        raise ValueError("Promoted library path does not exist")

    destination = _live_tool_dir(tool_name)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, destination / "bundle", dirs_exist_ok=True)

    install_id = uuid.uuid4().hex[:10]
    now = _now()
    manifest = {
        "install_id": install_id,
        "proposal_id": proposal["proposal_id"],
        "tool_name": tool_name,
        "live_path": str(destination),
        "library_path": proposal["library_path"],
        "installed_at": now,
        "status": "installed",
    }
    (destination / "install.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    connection.execute(
        """
        INSERT INTO tool_installs (
            install_id, proposal_id, tool_name, live_path, library_path, status, installed_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, 'installed', ?, ?)
        """,
        (
            install_id,
            proposal["proposal_id"],
            tool_name,
            str(destination),
            proposal["library_path"],
            now,
            now,
        ),
    )
    return _install_row_to_payload(connection, install_id)


def list_tool_installs(connection: sqlite3.Connection, status: str | None = None) -> list[dict]:
    query = [
        """
        SELECT
            install_id, proposal_id, tool_name, live_path, library_path, status, installed_at, updated_at
        FROM tool_installs
        WHERE 1 = 1
        """
    ]
    params: list[object] = []
    if status:
        query.append("AND status = ?")
        params.append(status)
    query.append("ORDER BY installed_at DESC")
    rows = connection.execute(" ".join(query), params).fetchall()
    return [dict(row) for row in rows]


def uninstall_tool(connection: sqlite3.Connection, install_id: str) -> dict:
    install = _install_row_to_payload(connection, install_id)
    live_path = Path(install["live_path"])
    if live_path.exists():
        shutil.rmtree(live_path)
    connection.execute(
        """
        UPDATE tool_installs
        SET status = 'uninstalled', updated_at = ?
        WHERE install_id = ?
        """,
        (_now(), install_id),
    )
    return _install_row_to_payload(connection, install_id)

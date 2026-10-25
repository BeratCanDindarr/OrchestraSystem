"""Manage run artifacts under .orchestra/runs/<run_id>/ (JSON version)."""
from __future__ import annotations

import gzip
import hashlib
import json
import threading
import os
import logging
from orchestra import config
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_digest(text: str) -> str:
    """Return SHA-256 hex digest of normalized text.

    Normalization strips leading/trailing whitespace and converts all
    newline variants (\\r\\n, \\r) to \\n before hashing.
    """
    if text is None:
        text = ""
    normalized = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

from orchestra.models import OrchestraRun, RunStatus
from orchestra.redaction import redact_file
from orchestra.state import InterruptState, RunStateSnapshot

def run_dir(run_id: str) -> Path:
    root = config.artifact_root()
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "agents").mkdir(exist_ok=True)
    return d

def cancelled_on_disk(run_id: str) -> bool:
    """True when the run was cancelled from outside (`orchestra cancel`, the MCP tool)."""
    path = config.artifact_root() / run_id / "manifest.json"
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("status") == RunStatus.CANCELLED.value)
    except (OSError, json.JSONDecodeError):
        return False


def adopt_cancel(run: OrchestraRun) -> bool:
    """Take over a cancel that was written to disk; True when the run is cancelled.

    The process that runs a run holds it in memory and writes it back after every step. Without
    this, the next of those writes puts "running" over the cancel.
    """
    if run.status == RunStatus.CANCELLED:
        return True
    if run.status in (RunStatus.RUNNING, RunStatus.QUEUED) and cancelled_on_disk(run.run_id):
        run.status = RunStatus.CANCELLED
        run.interrupt_state = InterruptState.CANCELLED
        return True
    return False


def write_manifest(run: OrchestraRun) -> None:
    adopt_cancel(run)
    manifest = run.to_manifest()
    for agent_dict, agent_run in zip(manifest.get("agents", []), run.agents):
        agent_dict["stdout_digest"] = compute_digest(agent_run.stdout_log)
    write_manifest_data(run.run_id, manifest)

def manifest_path(run_id: str) -> Path:
    return run_dir(run_id) / "manifest.json"

def load_manifest(run_id: str) -> dict | None:
    run_path = config.artifact_root() / run_id
    for path in (run_path / "manifest.json", run_path / "manifest.json.gz"):
        if not path.exists():
            continue
        try:
            if path.suffix == ".gz":
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    data: dict = json.load(f)
                return data
            data = json.loads(path.read_text(encoding="utf-8"))
            return data
        except (json.JSONDecodeError, OSError):
            continue
    return None

def load_run(run_id: str) -> OrchestraRun | None:
    """The run as it is on disk, its agents' answers included; None when there is no such run.

    A manifest holds a digest of each answer, not the answer; the text is in the agent's log.
    Whoever goes on with a stored run (resume, fleet finalize) needs the text: a verifier or
    a synthesis given empty answers has nothing to work on.
    """
    manifest = load_manifest(run_id)
    if manifest is None:
        return None
    run = OrchestraRun.from_manifest(manifest)
    agents_dir = config.artifact_root() / run_id / "agents"
    for agent in run.agents:
        try:
            agent.stdout_log = (agents_dir / f"{agent.alias}.stdout.log").read_text(encoding="utf-8")
        except OSError:
            continue
    return run


def _write_atomically(path: Path, text: str) -> None:
    """Replace `path` in one step: a reader sees the old file or the new one, never half of it."""
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def write_manifest_data(run_id: str, manifest: dict) -> None:
    path = run_dir(run_id) / "manifest.json"
    _write_atomically(path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    try:
        from orchestra.storage.db import upsert_run
        upsert_run(manifest)
    except Exception as exc:
        logger.warning("Failed to upsert run manifest: %s", exc)

def append_event(run_id: str, event: dict) -> None:
    d = run_dir(run_id)
    event["ts"] = datetime.now(timezone.utc).isoformat()
    event["run_id"] = run_id
    with open(d / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    try:
        from orchestra.storage.db import get_event_log
        get_event_log().append_raw(event.get("event", "ext_1"), run_id, event)
    except Exception as exc:
        logger.warning("Failed to insert event into DB: %s", exc)
    try:
        from orchestra.events.bus import emit as _bus_emit
        _bus_emit(run_id, event)
    except Exception as exc:
        logger.warning("Failed to emit bus event: %s", exc)

def write_agent_log(run_id: str, alias: str, content: str) -> Path:
    d = run_dir(run_id)
    log_path = d / "agents" / f"{alias}.stdout.log"
    log_path.write_text(content, encoding="utf-8")
    redact_file(log_path)
    return log_path

def write_normalized(run_id: str, alias: str, content: str) -> Path:
    d = run_dir(run_id)
    log_path = d / "agents" / f"{alias}.normalized.txt"
    log_path.write_text(content, encoding="utf-8")
    redact_file(log_path)
    return log_path

def read_agent_log(run_id: str, alias: str, normalized: bool = False) -> str | None:
    suffix = ".normalized.txt" if normalized else ".stdout.log"
    path = config.artifact_root() / run_id / "agents" / f"{alias}{suffix}"
    if not path.exists(): return None
    try: return path.read_text(encoding="utf-8")
    except OSError: return None

def read_events(run_id: str) -> list[dict]:
    run_path = config.artifact_root() / run_id
    for path in (run_path / "events.jsonl.gz", run_path / "events.jsonl"):
        if not path.exists():
            continue
        try:
            if path.suffix == ".gz":
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    lines = f.readlines()
            else:
                lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        events: list[dict] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events
    return []

def checkpoints_dir(run_id: str) -> Path:
    d = run_dir(run_id) / "checkpoints"; d.mkdir(exist_ok=True); return d


def latest_checkpoint(run_id: str) -> dict | None:
    checkpoints = list_checkpoints(run_id)
    return checkpoints[-1] if checkpoints else None


def load_checkpoint(run_id: str, *, checkpoint_version: int | None = None, label: str | None = None) -> dict | None:
    checkpoints = list_checkpoints(run_id)
    if not checkpoints:
        return None

    if checkpoint_version is not None:
        for checkpoint in checkpoints:
            if int(checkpoint.get("checkpoint_version", -1)) == checkpoint_version:
                return checkpoint
        return None

    if label is not None:
        for checkpoint in checkpoints[::-1]:   # newest first; reversed() loses the type in mypy 2.4
            if checkpoint.get("label") == label:
                return checkpoint
        return None

    return checkpoints[-1]


def write_checkpoint(run: OrchestraRun, label: str) -> Path:
    run.checkpoint_version += 1
    run.last_checkpoint_at = datetime.now(timezone.utc).isoformat()
    snapshot = RunStateSnapshot(
        schema_version=run.schema_version, checkpoint_version=run.checkpoint_version,
        label=label, run_id=run.run_id, mode=run.mode, status=run.status.value,
        turns=run.turns, total_cost_usd=run.total_cost_usd, avg_confidence=run.avg_confidence,
        approval_state=run.approval_state, interrupt_state=run.interrupt_state,
        failure=run.failure.to_dict() if run.failure else None,
        agents=[a.to_dict() for a in run.agents],
    )
    path = checkpoints_dir(run.run_id) / f"{run.checkpoint_version:04d}-{label}.json"
    path.write_text(json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_manifest(run)
    try:
        from orchestra.engine.runners.common import _get_event_log
        event_log = _get_event_log()
        if event_log:
            event_log.append("checkpoint_written", run.run_id, {"label": label})
    except Exception:
        logger.exception("Ignored exception")

    return path

def list_checkpoints(run_id: str) -> list[dict]:
    d = checkpoints_dir(run_id); results: list[dict] = []
    for path in sorted(d.glob("*.json")):
        try: results.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError): continue
    return results

def child_run_ids(parent_run_id: str) -> list[str]:
    """Runs that were started as internal steps of the given run."""
    root = config.artifact_root()
    if not root.exists():
        return []
    children = []
    for manifest_path in root.glob("*/manifest.json"):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("parent_run_id") == parent_run_id:
            children.append(str(data.get("run_id") or manifest_path.parent.name))
    return children


def list_runs(include_steps: bool = False) -> list[dict]:
    try:
        from orchestra.storage.db import list_runs_db
        return list_runs_db(150, include_steps=include_steps)
    except Exception:
        # Fallback to slow filesystem scan if DB fails
        root = config.artifact_root()
        if not root.exists(): return []
        runs = []
        for d in sorted(root.iterdir(), reverse=True):
            if not d.is_dir(): continue
            manifest = load_manifest(d.name)
            if manifest and (include_steps or not manifest.get("parent_run_id")):
                runs.append(manifest)
        return runs

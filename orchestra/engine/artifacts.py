"""Manage run artifacts under .orchestra/runs/<run_id>/ (JSON version)."""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

from orchestra import config
from orchestra.models import OrchestraRun
from orchestra.redaction import redact_file
from orchestra.state import RunStateSnapshot

def run_dir(run_id: str) -> Path:
    root = config.artifact_root()
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "agents").mkdir(exist_ok=True)
    return d

def write_manifest(run: OrchestraRun) -> None:
    write_manifest_data(run.run_id, run.to_manifest())

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
                    return json.load(f)
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return None

def write_manifest_data(run_id: str, manifest: dict) -> None:
    path = run_dir(run_id) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        from orchestra.storage.db import upsert_run
        upsert_run(manifest)
    except Exception: pass

def append_event(run_id: str, event: dict) -> None:
    d = run_dir(run_id)
    event["ts"] = datetime.now(timezone.utc).isoformat()
    event["run_id"] = run_id
    with open(d / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    try:
        from orchestra.storage.db import insert_event
        insert_event(run_id, event)
    except Exception: pass
    try:
        from orchestra.server import _event_hook
        _event_hook(run_id, event)
    except Exception: pass

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

def write_checkpoint(run: OrchestraRun, label: str) -> Path:
    run.checkpoint_version += 1
    run.last_checkpoint_at = datetime.now(timezone.utc).isoformat()
    snapshot = RunStateSnapshot(
        schema_version=run.schema_version, checkpoint_version=run.checkpoint_version,
        label=label, run_id=run.run_id, mode=run.mode, status=run.status.value,
        turns=run.turns, total_cost_usd=run.total_cost_usd, avg_confidence=run.avg_confidence,
        approval_state=run.approval_state, interrupt_state=run.interrupt_state,
        failure=run.failure.to_dict() if run.failure else None,
    )
    path = checkpoints_dir(run.run_id) / f"{run.checkpoint_version:04d}-{label}.json"
    path.write_text(json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_manifest(run)
    return path

def list_checkpoints(run_id: str) -> list[dict]:
    d = checkpoints_dir(run_id); results: list[dict] = []
    for path in sorted(d.glob("*.json")):
        try: results.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError): continue
    return results

def list_runs() -> list[dict]:
    root = config.artifact_root()
    if not root.exists(): return []
    runs = []
    for d in sorted(root.iterdir(), reverse=True):
        if not d.is_dir(): continue
        manifest = load_manifest(d.name)
        if manifest: runs.append(manifest)
    return runs

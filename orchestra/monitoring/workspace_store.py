"""Persistent store for workspaces — named collections of task rooms.

A workspace groups rooms for a distinct purpose (e.g. "Research", "Shipping"). Rooms
carry a ``workspace_id``; the board shows one active workspace at a time. Workspaces and
the active selection persist to ``.orchestra/workspaces.json`` so they survive restarts.
Updates are immutable (load → new list → atomic write).
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Workspace:
    id: str
    name: str
    created_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Workspace":
        return Workspace(id=str(d.get("id", "")), name=str(d.get("name", "")),
                         created_at=float(d.get("created_at", 0.0) or 0.0))


def _store_path() -> Path:
    from orchestra import config
    return config.eventlog_db_path().parent / "workspaces.json"


class WorkspaceStore:
    """Thread-safe JSON store of workspaces + the active workspace id."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._seq = 0

    def _file(self) -> Path:
        return self._path if self._path is not None else _store_path()

    def _read(self) -> dict:
        try:
            data = json.loads(self._file().read_text())
        except (FileNotFoundError, ValueError, OSError):
            return {"active": "", "workspaces": []}
        if not isinstance(data, dict):
            return {"active": "", "workspaces": []}
        return {"active": str(data.get("active", "")),
                "workspaces": [Workspace.from_dict(w) for w in data.get("workspaces", [])
                               if isinstance(w, dict)]}

    def _write(self, active: str, workspaces: list) -> None:
        path = self._file()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"active": active,
                                   "workspaces": [w.to_dict() for w in workspaces]}, indent=2))
        os.replace(tmp, path)

    def list(self) -> list:
        with self._lock:
            workspaces: list = self._read()["workspaces"]
            return workspaces

    def get(self, ws_id: str) -> Optional[Workspace]:
        return next((w for w in self.list() if w.id == ws_id), None)

    def create(self, name: str, *, now: Optional[float] = None, activate: bool = True) -> Workspace:
        now = now if now is not None else time.time()
        with self._lock:
            state = self._read()
            self._seq += 1
            ws = Workspace(id=f"ws-{int(now * 1000)}-{self._seq}",
                           name=(name or "New workspace").strip(), created_at=now)
            workspaces = state["workspaces"] + [ws]
            active = ws.id if activate or not state["active"] else state["active"]
            self._write(active, workspaces)
            return ws

    def rename(self, ws_id: str, name: str) -> Optional[Workspace]:
        with self._lock:
            state = self._read()
            updated = None
            out = []
            for w in state["workspaces"]:
                if w.id == ws_id:
                    updated = replace(w, name=(name or w.name).strip())
                    out.append(updated)
                else:
                    out.append(w)
            if updated is not None:
                self._write(state["active"], out)
            return updated

    def delete(self, ws_id: str) -> bool:
        with self._lock:
            state = self._read()
            kept = [w for w in state["workspaces"] if w.id != ws_id]
            if len(kept) == len(state["workspaces"]) or not kept:
                return False           # unknown id, or refusing to delete the last workspace
            active = state["active"]
            if active == ws_id:
                active = kept[0].id
            self._write(active, kept)
            return True

    def get_active(self) -> Optional[Workspace]:
        state = self._read()
        active = next((w for w in state["workspaces"] if w.id == state["active"]), None)
        return active or (state["workspaces"][0] if state["workspaces"] else None)

    def set_active(self, ws_id: str) -> bool:
        with self._lock:
            state = self._read()
            if not any(w.id == ws_id for w in state["workspaces"]):
                return False
            self._write(ws_id, state["workspaces"])
            return True

    def ensure_default(self) -> Workspace:
        """Guarantee at least one workspace exists; return the active one."""
        active = self.get_active()
        if active is not None:
            return active
        return self.create("Default")


_store: Optional[WorkspaceStore] = None
_store_lock = threading.Lock()


def get_workspace_store() -> WorkspaceStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = WorkspaceStore()
        return _store

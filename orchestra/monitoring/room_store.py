"""Persistent store for user-defined task rooms.

A room is a configurable workspace: a name, a role preset, and free-form rules.
Rooms are stored as JSON at ``.orchestra/rooms.json`` and updated immutably (load →
new list → atomic write). Phase 2 will add ``connections`` for routing between rooms.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Room:
    id: str
    name: str
    preset: str = "custom"
    rules: str = ""
    created_at: float = 0.0
    connections: tuple = field(default=())      # Phase 2: downstream room ids
    default_model: str = ""                      # Phase 3: preferred model for hand-offs ("" = auto-cast)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["connections"] = list(self.connections)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Room":
        return Room(
            id=str(d.get("id", "")),
            name=str(d.get("name", "")),
            preset=str(d.get("preset", "custom")),
            rules=str(d.get("rules", "")),
            created_at=float(d.get("created_at", 0.0) or 0.0),
            connections=tuple(d.get("connections", []) or []),
            default_model=str(d.get("default_model", "")),
        )


def _store_path() -> Path:
    from orchestra import config
    return config.eventlog_db_path().parent / "rooms.json"


class RoomStore:
    """Thread-safe JSON-backed room registry. All mutations write atomically."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._seq = 0

    def _file(self) -> Path:
        return self._path if self._path is not None else _store_path()

    def _load(self) -> list[Room]:
        path = self._file()
        try:
            raw = json.loads(path.read_text())
        except (FileNotFoundError, ValueError, OSError):
            return []
        return [Room.from_dict(d) for d in raw if isinstance(d, dict)]

    def _save(self, rooms: list[Room]) -> None:
        path = self._file()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps([r.to_dict() for r in rooms], indent=2))
        os.replace(tmp, path)      # atomic

    def list(self) -> list[Room]:
        with self._lock:
            return self._load()

    def get(self, room_id: str) -> Optional[Room]:
        return next((r for r in self.list() if r.id == room_id), None)

    def create(self, name: str, *, preset: str = "custom", rules: str = "",
               now: Optional[float] = None) -> Room:
        now = now if now is not None else time.time()
        with self._lock:
            rooms = self._load()
            self._seq += 1
            rid = f"room-{int(now * 1000)}-{self._seq}"
            room = Room(id=rid, name=(name or "Untitled room").strip(),
                        preset=preset or "custom", rules=rules or "", created_at=now)
            self._save(rooms + [room])      # new list, no mutation
            return room

    def update(self, room_id: str, **fields) -> Optional[Room]:
        allowed = {"name", "preset", "rules", "connections", "default_model"}
        changes = {k: v for k, v in fields.items() if k in allowed}
        with self._lock:
            rooms = self._load()
            updated: Optional[Room] = None
            new_rooms = []
            for r in rooms:
                if r.id == room_id:
                    if "connections" in changes:
                        changes["connections"] = tuple(changes["connections"])
                    updated = replace(r, **changes)
                    new_rooms.append(updated)
                else:
                    new_rooms.append(r)
            if updated is not None:
                self._save(new_rooms)
            return updated

    def delete(self, room_id: str) -> bool:
        with self._lock:
            rooms = self._load()
            kept = [r for r in rooms if r.id != room_id]
            if len(kept) == len(rooms):
                return False
            # also drop any dangling connections to the removed room
            kept = [replace(r, connections=tuple(c for c in r.connections if c != room_id))
                    for r in kept]
            self._save(kept)
            return True


_store: Optional[RoomStore] = None
_store_lock = threading.Lock()


def get_room_store() -> RoomStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = RoomStore()
        return _store

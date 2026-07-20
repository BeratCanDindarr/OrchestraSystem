from __future__ import annotations
"""Dispatch inbox: track tasks assigned from the fleet dashboard and their results.

When you assign work to a character it lands here as ``running``; when the run
finishes, its output is captured. Completed-but-unread items surface in a tray at
the top of the dashboard until you read them (open the output), then they drop back
to the normal fleet view. Pure in-memory + thread-safe; ``now`` is a parameter so
the ordering/timestamps are deterministic in tests.
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

_RUNNING = "running"
_COMPLETED = "completed"
_FAILED = "failed"


@dataclass
class Dispatch:
    id: str
    character_id: str
    task: str
    status: str = _RUNNING
    output: str = ""
    read: bool = False
    created_at: float = 0.0
    completed_at: Optional[float] = None


class DispatchInbox:
    """In-memory registry of dispatched tasks and their outcomes."""

    def __init__(self) -> None:
        self._items: dict[str, Dispatch] = {}
        self._seq = 0
        self._lock = threading.Lock()

    def add(self, character_id: str, task: str, *, now: Optional[float] = None) -> str:
        now = now if now is not None else time.time()
        with self._lock:
            self._seq += 1
            did = f"d{self._seq}"
            self._items[did] = Dispatch(
                id=did, character_id=character_id, task=task, created_at=now,
            )
        return did

    def complete(self, dispatch_id: str, output: str, *, ok: bool = True,
                 now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        with self._lock:
            d = self._items.get(dispatch_id)
            if d is None:
                return
            d.status = _COMPLETED if ok else _FAILED
            d.output = output
            d.completed_at = now

    def get(self, dispatch_id: str) -> Optional[Dispatch]:
        with self._lock:
            return self._items.get(dispatch_id)

    def mark_read(self, dispatch_id: str) -> None:
        with self._lock:
            d = self._items.get(dispatch_id)
            if d is not None:
                d.read = True

    def tray(self) -> list[Dispatch]:
        """Running + completed-but-unread dispatches, newest first."""
        with self._lock:
            items = [
                d for d in self._items.values()
                if d.status == _RUNNING or (d.status in (_COMPLETED, _FAILED) and not d.read)
            ]
        return sorted(items, key=lambda d: d.created_at, reverse=True)

    def unread_count(self) -> int:
        with self._lock:
            return sum(1 for d in self._items.values()
                       if d.status in (_COMPLETED, _FAILED) and not d.read)

    def recent(self, limit: int = 50) -> list[Dispatch]:
        with self._lock:
            items = list(self._items.values())
        return sorted(items, key=lambda d: d.created_at, reverse=True)[:limit]


_inbox: Optional[DispatchInbox] = None
_inbox_lock = threading.Lock()


def get_inbox() -> DispatchInbox:
    """Process-wide dispatch inbox singleton."""
    global _inbox
    if _inbox is None:
        with _inbox_lock:
            if _inbox is None:
                _inbox = DispatchInbox()
    return _inbox

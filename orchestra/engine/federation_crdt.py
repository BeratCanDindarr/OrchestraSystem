from __future__ import annotations
"""Pure-Python CRDT primitives for federated semantic memory.

No external CRDT framework — LWW-Register, G-Counter, and OR-Set are
implemented from first principles so that merge is guaranteed to be
commutative, associative, and idempotent.
"""

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class LWWRegister(Generic[T]):
    """Last-Write-Wins register."""

    value: T | None = None
    timestamp: float = 0.0
    peer_id: str = ""

    def set(self, value: T, timestamp: float, peer_id: str) -> None:
        if timestamp > self.timestamp:
            self.value = value
            self.timestamp = timestamp
            self.peer_id = peer_id
        elif timestamp == self.timestamp and peer_id > self.peer_id:
            self.value = value
            self.timestamp = timestamp
            self.peer_id = peer_id

    def merge(self, other: "LWWRegister[T]") -> "LWWRegister[T]":
        winner = self if self.timestamp > other.timestamp else other
        if self.timestamp == other.timestamp:
            winner = self if self.peer_id > other.peer_id else other
        return LWWRegister(winner.value, winner.timestamp, winner.peer_id)

    def to_dict(self) -> dict:
        return {"value": self.value, "timestamp": self.timestamp, "peer_id": self.peer_id}

    @classmethod
    def from_dict(cls, data: dict) -> "LWWRegister":
        return cls(value=data.get("value"), timestamp=data.get("timestamp", 0.0), peer_id=data.get("peer_id", ""))


@dataclass
class GCounter:
    """Grow-only counter (component-wise max)."""

    _state: dict[str, int] = field(default_factory=dict)

    def increment(self, peer_id: str, delta: int = 1) -> None:
        self._state[peer_id] = self._state.get(peer_id, 0) + delta

    def value(self) -> int:
        return sum(self._state.values())

    def merge(self, other: "GCounter") -> "GCounter":
        merged = dict(self._state)
        for peer_id, val in other._state.items():
            merged[peer_id] = max(merged.get(peer_id, 0), val)
        return GCounter(merged)

    def to_dict(self) -> dict:
        return dict(self._state)

    @classmethod
    def from_dict(cls, data: dict) -> "GCounter":
        return cls(_state=dict(data))


@dataclass
class ORSet(Generic[T]):
    """Observed-Remove Set (add-wins, tombstones prevent resurrection).

    Each ``add`` generates a unique tag so that concurrent adds win over
    removes — the added tag is not in the remover's tombstone set.
    """

    _adds: dict[str, T] = field(default_factory=dict)
    _removes: set[str] = field(default_factory=set)

    def _new_tag(self) -> str:
        import uuid
        return uuid.uuid4().hex

    def add(self, element: T) -> str:
        tag = self._new_tag()
        self._adds[tag] = element
        self._removes.discard(tag)
        return tag

    def remove(self, element: T) -> None:
        # Tombstone every existing tag for this element
        for tag, val in list(self._adds.items()):
            if val == element:
                self._removes.add(tag)
                self._adds.pop(tag, None)

    def elements(self) -> set[T]:
        return {v for k, v in self._adds.items() if k not in self._removes}

    def merge(self, other: "ORSet[T]") -> "ORSet[T]":
        merged_adds = dict(self._adds)
        merged_adds.update(other._adds)
        merged_removes = self._removes | other._removes
        return ORSet(merged_adds, merged_removes)

    def to_dict(self) -> dict:
        # Serialize as list of (tag, element) pairs so unique tags survive
        return {
            "adds": [[k, v] for k, v in self._adds.items()],
            "removes": list(self._removes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ORSet":
        s = cls()
        for pair in data.get("adds", []):
            if isinstance(pair, list) and len(pair) == 2:
                tag, element = pair
                s._adds[tag] = element
                s._removes.discard(tag)
        for tag in data.get("removes", []):
            s._removes.add(tag)
        return s


@dataclass
class FederatedState:
    """Composite CRDT state for a federation summary."""

    content: LWWRegister[str] = field(default_factory=lambda: LWWRegister(""))
    counter: GCounter = field(default_factory=GCounter)
    tags: ORSet[str] = field(default_factory=lambda: ORSet())

    def merge(self, other: "FederatedState") -> "FederatedState":
        return FederatedState(
            content=self.content.merge(other.content),
            counter=self.counter.merge(other.counter),
            tags=self.tags.merge(other.tags),
        )

    def to_dict(self) -> dict:
        return {
            "content": self.content.to_dict(),
            "counter": self.counter.to_dict(),
            "tags": self.tags.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FederatedState":
        return cls(
            content=LWWRegister.from_dict(data.get("content", {})),
            counter=GCounter.from_dict(data.get("counter", {})),
            tags=ORSet.from_dict(data.get("tags", {})),
        )

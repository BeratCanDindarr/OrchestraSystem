"""Tests for pure-Python CRDT primitives."""
from __future__ import annotations

import pytest

from orchestra.engine.federation_crdt import (
    FederatedState,
    GCounter,
    LWWRegister,
    ORSet,
)


class TestLWWRegister:
    def test_set_newer_timestamp_wins(self) -> None:
        r = LWWRegister("a", 1.0, "p1")
        r.set("b", 2.0, "p2")
        assert r.value == "b"
        assert r.timestamp == 2.0

    def test_set_older_timestamp_ignored(self) -> None:
        r = LWWRegister("a", 2.0, "p1")
        r.set("b", 1.0, "p2")
        assert r.value == "a"

    def test_set_tie_break_by_peer_id(self) -> None:
        r = LWWRegister("a", 1.0, "p1")
        r.set("b", 1.0, "p2")
        assert r.value == "b"

    def test_merge_commutative(self) -> None:
        a = LWWRegister("x", 1.0, "p1")
        b = LWWRegister("y", 2.0, "p2")
        assert a.merge(b).value == b.merge(a).value == "y"

    def test_merge_idempotent(self) -> None:
        a = LWWRegister("x", 1.0, "p1")
        assert a.merge(a).value == "x"
        assert a.merge(a).timestamp == 1.0

    def test_round_trip_dict(self) -> None:
        a = LWWRegister("hello", 42.0, "peer-a")
        b = LWWRegister.from_dict(a.to_dict())
        assert a.value == b.value
        assert a.timestamp == b.timestamp
        assert a.peer_id == b.peer_id


class TestGCounter:
    def test_increment_and_value(self) -> None:
        c = GCounter()
        c.increment("a", 3)
        c.increment("b", 2)
        assert c.value() == 5

    def test_merge_commutative(self) -> None:
        a = GCounter({"x": 1, "y": 2})
        b = GCounter({"y": 3, "z": 4})
        m1 = a.merge(b)
        m2 = b.merge(a)
        assert m1.value() == m2.value() == 8

    def test_merge_idempotent(self) -> None:
        a = GCounter({"x": 5})
        assert a.merge(a).value() == 5

    def test_merge_associative(self) -> None:
        a = GCounter({"x": 1})
        b = GCounter({"x": 2})
        c = GCounter({"x": 3})
        assert a.merge(b).merge(c).value() == a.merge(b.merge(c)).value() == 3

    def test_round_trip_dict(self) -> None:
        c = GCounter({"a": 1, "b": 2})
        d = GCounter.from_dict(c.to_dict())
        assert c.value() == d.value()


class TestORSet:
    def test_add_and_elements(self) -> None:
        s = ORSet()
        s.add("a")
        s.add("b")
        assert s.elements() == {"a", "b"}

    def test_remove(self) -> None:
        s = ORSet()
        s.add("a")
        s.remove("a")
        assert "a" not in s.elements()

    def test_merge_commutative(self) -> None:
        a = ORSet()
        a.add("x")
        b = ORSet()
        b.add("y")
        assert a.merge(b).elements() == b.merge(a).elements() == {"x", "y"}

    def test_merge_idempotent(self) -> None:
        a = ORSet()
        a.add("x")
        assert a.merge(a).elements() == {"x"}

    def test_add_wins_over_remove(self) -> None:
        a = ORSet()
        a.add("x")
        a.remove("x")
        b = ORSet()
        b.add("x")
        merged = a.merge(b)
        assert "x" in merged.elements()

    def test_round_trip_dict(self) -> None:
        s = ORSet()
        s.add("a")
        s.add("b")
        t = ORSet.from_dict(s.to_dict())
        assert s.elements() == t.elements()


class TestFederatedState:
    def test_merge_composition(self) -> None:
        a = FederatedState()
        a.content.set("hello", 1.0, "p1")
        a.counter.increment("p1", 2)
        a.tags.add("python")

        b = FederatedState()
        b.content.set("world", 2.0, "p2")
        b.counter.increment("p2", 3)
        b.tags.add("rust")

        merged = a.merge(b)
        assert merged.content.value == "world"
        assert merged.counter.value() == 5
        assert merged.tags.elements() == {"python", "rust"}

    def test_merge_commutative(self) -> None:
        a = FederatedState()
        a.content.set("x", 1.0, "p1")
        b = FederatedState()
        b.content.set("y", 2.0, "p2")
        m1 = a.merge(b)
        m2 = b.merge(a)
        assert m1.content.value == m2.content.value == "y"

    def test_round_trip_dict(self) -> None:
        a = FederatedState()
        a.content.set("test", 42.0, "peer")
        a.counter.increment("peer", 5)
        a.tags.add("tag1")
        b = FederatedState.from_dict(a.to_dict())
        assert b.content.value == "test"
        assert b.counter.value() == 5
        assert b.tags.elements() == {"tag1"}

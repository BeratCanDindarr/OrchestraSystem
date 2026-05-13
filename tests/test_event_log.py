"""Unit tests for EventLog (Block 1: EventLog Storage)."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import List

import pytest

from orchestra.storage.event_log import (
    Event,
    EventLog,
    InvalidEventTypeError,
    DatabaseError,
    VALID_EVENT_TYPES,
)


class TestEventLogBasics:
    """Basic EventLog functionality tests."""

    def test_init_creates_database(self):
        """EventLog.__init__ creates SQLite database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            event_log = EventLog(db_path)
            assert db_path.exists()
            assert event_log.db_path == str(db_path)

    def test_init_with_default_path(self):
        """EventLog without db_path uses config default."""
        event_log = EventLog()
        assert event_log.db_path is not None
        assert event_log.db_path.endswith(".db")

    def test_append_returns_uuid(self):
        """append() returns a valid UUID string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")
            event_id = event_log.append("run_started", "run-123", {"mode": "ask"})
            assert isinstance(event_id, str)
            assert len(event_id) == 36  # UUID4 format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
            assert event_id.count("-") == 4

    def test_append_invalid_event_type(self):
        """append() raises InvalidEventTypeError for invalid event_type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")
            with pytest.raises(InvalidEventTypeError) as exc_info:
                event_log.append("invalid_event", "run-123", {})
            assert "Invalid event_type 'invalid_event'" in str(exc_info.value)

    def test_append_valid_event_types(self):
        """append() accepts all 30 valid event types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")
            for event_type in VALID_EVENT_TYPES:
                event_id = event_log.append(event_type, "run-123", {})
                assert isinstance(event_id, str)
                assert len(event_id) == 36

    def test_append_stores_payload_as_json(self):
        """append() serializes payload to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")
            payload = {"alias": "cdx-deep", "cost": 1.23, "tags": ["auto"]}
            event_id = event_log.append("run_started", "run-123", payload)

            # Verify via direct SQLite query
            conn = sqlite3.connect(str(event_log.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT payload FROM events WHERE id = ?", (event_id,))
            row = cursor.fetchone()
            assert row is not None
            stored_payload = json.loads(row[0])
            assert stored_payload == payload
            conn.close()

    def test_append_invalid_payload_raises_error(self):
        """append() raises DatabaseError for non-JSON-serializable payload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")

            # Use an object that cannot be JSON serialized
            class NonSerializable:
                pass

            with pytest.raises(DatabaseError) as exc_info:
                event_log.append("run_started", "run-123", {"obj": NonSerializable()})
            assert "Failed to serialize payload" in str(exc_info.value)


class TestEventLogReplay:
    """EventLog replay() functionality tests."""

    def test_replay_empty_run(self):
        """replay() returns empty list for run with no events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")
            events = event_log.replay("run-nonexistent")
            assert events == []

    def test_replay_single_event(self):
        """replay() returns single event with correct fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")
            payload = {"mode": "ask"}
            event_id = event_log.append("run_started", "run-123", payload)

            events = event_log.replay("run-123")
            assert len(events) == 1
            assert events[0].event_id == event_id
            assert events[0].event_type == "run_started"
            assert events[0].run_id == "run-123"
            assert events[0].payload == payload
            assert isinstance(events[0].ts, float)
            assert events[0].ts > 0

    def test_replay_multiple_events_ordered_by_timestamp(self):
        """replay() returns events in timestamp order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")

            # Append 3 events with small delays
            ids = []
            for i in range(3):
                event_id = event_log.append(
                    "agent_started", "run-123", {"agent_num": i}
                )
                ids.append(event_id)
                time.sleep(0.01)  # Ensure timestamp ordering

            events = event_log.replay("run-123")
            assert len(events) == 3
            assert [e.event_id for e in events] == ids
            # Verify strictly increasing timestamps
            for i in range(len(events) - 1):
                assert events[i].ts <= events[i + 1].ts

    def test_replay_filters_by_run_id(self):
        """replay() only returns events for specified run_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")

            # Append events to different runs
            event_log.append("run_started", "run-111", {})
            event_log.append("run_started", "run-222", {})
            event_log.append("agent_started", "run-111", {})

            events_111 = event_log.replay("run-111")
            events_222 = event_log.replay("run-222")

            assert len(events_111) == 2
            assert len(events_222) == 1
            assert all(e.run_id == "run-111" for e in events_111)
            assert all(e.run_id == "run-222" for e in events_222)

    def test_replay_payload_deserialization(self):
        """replay() correctly deserializes JSON payloads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")

            complex_payload = {
                "nested": {"key": "value", "list": [1, 2, 3]},
                "string": "test",
                "number": 42.5,
                "boolean": True,
                "null": None,
            }
            event_log.append("run_started", "run-123", complex_payload)

            events = event_log.replay("run-123")
            assert len(events) == 1
            assert events[0].payload == complex_payload


class TestEventLogGetEvent:
    """EventLog get_event() functionality tests."""

    def test_get_event_by_id(self):
        """get_event() returns Event with correct data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")
            payload = {"key": "value"}
            event_id = event_log.append("run_started", "run-123", payload)

            event = event_log.get_event(event_id)
            assert event is not None
            assert event.event_id == event_id
            assert event.event_type == "run_started"
            assert event.payload == payload

    def test_get_event_nonexistent(self):
        """get_event() returns None for nonexistent event_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")
            event = event_log.get_event("nonexistent-id")
            assert event is None

    def test_get_event_immutable(self):
        """Event returned by get_event() is immutable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")
            event_id = event_log.append("run_started", "run-123", {})
            event = event_log.get_event(event_id)

            with pytest.raises(Exception):  # frozenclass cannot be modified
                event.event_type = "modified"  # type: ignore


class TestEventLogThreadSafety:
    """Thread-safety tests for EventLog."""

    def test_concurrent_appends_no_corruption(self):
        """100+ parallel appends do not corrupt database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")
            num_threads = 10
            events_per_thread = 15
            results: List[List[str]] = [[] for _ in range(num_threads)]
            errors: List[Exception] = []

            def append_events(thread_id: int):
                try:
                    for i in range(events_per_thread):
                        event_id = event_log.append(
                            "agent_started",
                            f"run-{thread_id}",
                            {"iteration": i},
                        )
                        results[thread_id].append(event_id)
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=append_events, args=(i,))
                for i in range(num_threads)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Verify no errors
            assert len(errors) == 0

            # Verify all events stored
            total_appended = sum(len(r) for r in results)
            assert total_appended == num_threads * events_per_thread

            # Verify no duplicates
            all_ids = []
            for r in results:
                all_ids.extend(r)
            assert len(all_ids) == len(set(all_ids))

            # Verify can replay all events
            for thread_id in range(num_threads):
                events = event_log.replay(f"run-{thread_id}")
                assert len(events) == events_per_thread

    def test_replay_during_append(self):
        """replay() works correctly while append() is happening."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")

            # Append initial events
            event_log.append("run_started", "run-123", {})

            results = {"initial": None, "final": None, "errors": []}

            def replay_thread():
                try:
                    results["initial"] = event_log.replay("run-123")
                    time.sleep(0.05)  # Let append thread work
                    results["final"] = event_log.replay("run-123")
                except Exception as e:
                    results["errors"].append(e)

            def append_thread():
                time.sleep(0.01)
                for i in range(10):
                    event_log.append("agent_started", "run-123", {"i": i})
                    time.sleep(0.003)

            t1 = threading.Thread(target=replay_thread)
            t2 = threading.Thread(target=append_thread)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            # Verify no errors
            assert len(results["errors"]) == 0
            # Verify replay results changed (saw more events after appends)
            assert len(results["initial"]) < len(results["final"])


class TestEventLogSchema:
    """Schema validation tests."""

    def test_schema_has_check_constraint(self):
        """Database schema enforces event_type CHECK constraint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            event_log = EventLog(db_path)

            # Try to insert invalid event_type directly (bypassing Python validation)
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            with pytest.raises(sqlite3.IntegrityError):
                cursor.execute(
                    """
                    INSERT INTO events (id, event_type, run_id, payload)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("test-id", "invalid_type", "run-123", "{}"),
                )
            conn.close()

    def test_schema_indexes_exist(self):
        """Database schema has required indexes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            event_log = EventLog(db_path)

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='events'"
            )
            index_names = {row[0] for row in cursor.fetchall()}

            assert "idx_events_run_id" in index_names
            assert "idx_events_ts" in index_names
            assert "idx_events_event_type" in index_names
            conn.close()

    def test_wal_mode_enabled(self):
        """Database uses WAL mode for concurrency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            event_log = EventLog(db_path)

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            assert mode.upper() == "WAL"
            conn.close()


class TestEventDataclass:
    """Event dataclass tests."""

    def test_event_to_dict(self):
        """Event.to_dict() returns dictionary representation."""
        event = Event(
            event_id="uuid-123",
            event_type="run_started",
            run_id="run-456",
            payload={"key": "value"},
            ts=1715000000.123,
            seq=1,
        )
        d = event.to_dict()
        assert d == {
            "event_id": "uuid-123",
            "event_type": "run_started",
            "run_id": "run-456",
            "payload": {"key": "value"},
            "ts": 1715000000.123,
            "seq": 1,
        }

    def test_event_immutable(self):
        """Event is frozen and immutable."""
        event = Event(
            event_id="uuid-123",
            event_type="run_started",
            run_id="run-456",
            payload={},
            ts=1715000000.123,
            seq=1,
        )
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            event.event_type = "modified"  # type: ignore


class TestEventLogErrorHandling:
    """Error handling tests."""

    def test_database_error_on_connection_failure(self):
        """DatabaseError raised on connection failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a read-only directory to test permission errors
            readonly_dir = Path(tmpdir) / "readonly"
            readonly_dir.mkdir()
            db_path = readonly_dir / "db.db"

            # Write initial database
            event_log = EventLog(db_path)
            event_log.append("run_started", "run-123", {})

            # Make directory read-only
            import os
            os.chmod(readonly_dir, 0o444)

            try:
                # Attempting to append to read-only db should fail
                with pytest.raises(DatabaseError):
                    event_log.append("run_started", "run-123", {})
            finally:
                # Restore permissions for cleanup
                os.chmod(readonly_dir, 0o755)

    def test_corrupt_json_payload_handled_on_replay(self):
        """replay() handles corrupt JSON gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            event_log = EventLog(db_path)

            # Insert a valid event first
            event_log.append("run_started", "run-123", {"valid": "data"})

            # Manually corrupt a payload (seq is now NOT NULL, so provide it)
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO events (id, event_type, run_id, payload, ts, seq) VALUES (?, ?, ?, ?, unixepoch('subsec'), (SELECT COALESCE(MAX(seq), 0) + 1 FROM events))",
                ("corrupt-id", "run_completed", "run-123", "INVALID JSON{{{"),
            )
            conn.commit()
            conn.close()

            # replay() should handle corrupt data gracefully
            events = event_log.replay("run-123")
            assert len(events) == 2
            # The corrupted event should have empty payload
            assert events[1].payload == {}


class TestValidEventTypes:
    """VALID_EVENT_TYPES constant tests."""

    def test_valid_event_types_count(self):
        """VALID_EVENT_TYPES has at least 26 types (30 per spec)."""
        assert len(VALID_EVENT_TYPES) >= 26

    def test_valid_event_types_includes_spec_types(self):
        """VALID_EVENT_TYPES includes all types from specification."""
        required_types = {
            "run_started", "run_completed", "run_failed",
            "agent_started", "agent_completed", "agent_failed", "agent_retrying",
            "budget_check", "budget_exceeded",
            "approval_requested", "approval_approved", "approval_rejected",
            "suspension_created", "suspension_resumed",
            "checkpoint_written", "checkpoint_loaded",
            "synthesis_started", "synthesis_completed",
            "cache_hit", "cache_miss",
            "error_logged", "validation_failed",
            "span_started", "span_completed",
            "cost_tracked", "token_count_recorded",
        }
        assert required_types.issubset(VALID_EVENT_TYPES)

    def test_valid_event_types_includes_extensions(self):
        """VALID_EVENT_TYPES includes 4 extensibility slots."""
        extension_types = {"ext_1", "ext_2", "ext_3", "ext_4"}
        assert extension_types.issubset(VALID_EVENT_TYPES)

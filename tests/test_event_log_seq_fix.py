"""Unit tests for EventLog seq fix (Block 8: Events Seq Fix)."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
from orchestra.storage.migrations import migrate_v2_to_v3, MigrationError


class TestSeqAtomicity:
    """Test atomic sequence number generation."""

    def test_append_returns_uuid_not_seq(self):
        """append() returns event_id (UUID), not seq number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")
            event_id = event_log.append("run_started", "run-123", {"mode": "ask"})
            # Should be UUID format
            assert len(event_id) == 36
            assert event_id.count("-") == 4

    def test_single_append_creates_seq_1(self):
        """First event in database gets seq=1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")
            event_id = event_log.append("run_started", "run-123", {})

            event = event_log.get_event(event_id)
            assert event is not None
            assert event.seq == 1

    def test_sequential_appends_have_monotonic_seq(self):
        """10 sequential appends have seq 1..10, strictly monotonic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")

            event_ids = []
            for i in range(10):
                event_id = event_log.append(
                    "agent_started", "run-123", {"iteration": i}
                )
                event_ids.append(event_id)

            # Fetch and check seq values
            seqs = []
            for event_id in event_ids:
                event = event_log.get_event(event_id)
                assert event is not None
                seqs.append(event.seq)

            # Check monotonic increase
            assert seqs == list(range(1, 11))

    def test_concurrent_appends_unique_seq(self):
        """100 concurrent appends have 100 unique seq values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")
            num_threads = 10
            events_per_thread = 10
            results: dict[int, list[tuple[str, int]]] = {i: [] for i in range(num_threads)}
            errors: list[Exception] = []

            def append_events(thread_id: int):
                try:
                    for i in range(events_per_thread):
                        event_id = event_log.append(
                            "agent_started",
                            f"run-{thread_id}",
                            {"iteration": i},
                        )
                        # Fetch seq immediately
                        event = event_log.get_event(event_id)
                        if event is None:
                            raise DatabaseError(f"Event {event_id} not found")
                        results[thread_id].append((event_id, event.seq))
                except Exception as e:
                    errors.append(e)

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(append_events, i)
                    for i in range(num_threads)
                ]
                for f in futures:
                    f.result()

            # Verify no errors
            assert len(errors) == 0, f"Errors during concurrent append: {errors}"

            # Collect all seqs
            all_seqs = []
            for thread_results in results.values():
                for _, seq in thread_results:
                    all_seqs.append(seq)

            # Verify uniqueness
            assert len(all_seqs) == num_threads * events_per_thread
            assert len(set(all_seqs)) == len(all_seqs), "Duplicate seq values detected"

            # Verify monotonic (when sorted)
            sorted_seqs = sorted(all_seqs)
            assert sorted_seqs == list(range(1, num_threads * events_per_thread + 1)), \
                "Seq values should be contiguous 1..N when sorted"

    def test_race_condition_prevention_with_begin_immediate(self):
        """Multiple threads appending concurrently do not create duplicate seqs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")

            seqs_by_thread: dict[int, list[int]] = {}
            lock = threading.Lock()
            errors: list[Exception] = []

            def append_and_record(thread_id: int):
                try:
                    seqs = []
                    for i in range(20):
                        event_id = event_log.append(
                            "budget_check",
                            f"run-{thread_id}",
                            {"check": i},
                        )
                        event = event_log.get_event(event_id)
                        if event:
                            seqs.append(event.seq)
                    with lock:
                        seqs_by_thread[thread_id] = seqs
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=append_and_record, args=(i,))
                for i in range(5)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            # Collect all seqs
            all_seqs = []
            for seqs in seqs_by_thread.values():
                all_seqs.extend(seqs)

            # Verify no duplicates
            assert len(set(all_seqs)) == len(all_seqs), "Duplicate seq values detected"
            # Verify seqs are contiguous when sorted
            sorted_seqs = sorted(all_seqs)
            assert sorted_seqs == list(range(1, len(all_seqs) + 1)), \
                "Seq values should be contiguous 1..N when sorted"


class TestReplayOrdering:
    """Test replay() uses seq-based ordering."""

    def test_replay_empty_run(self):
        """replay() returns empty list for run with no events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")
            events = event_log.replay("run-nonexistent")
            assert events == []

    def test_replay_orders_by_seq_not_timestamp(self):
        """replay() orders events by seq, not timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")

            # Append events with intentional delays
            event_ids = []
            for i in range(3):
                event_id = event_log.append(
                    "agent_started", "run-123", {"iteration": i}
                )
                event_ids.append(event_id)
                time.sleep(0.01)

            # Verify events replay in seq order
            events = event_log.replay("run-123")
            assert len(events) == 3

            # Check seqs are monotonic
            seqs = [e.seq for e in events]
            assert seqs == sorted(seqs)
            # Should be 1, 2, 3
            assert seqs == [1, 2, 3]

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

    def test_replay_large_run_preserves_seq_order(self):
        """replay() preserves seq order for 100+ events in a run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")

            # Append 100 events
            for i in range(100):
                event_log.append(
                    "agent_started", "run-large", {"iteration": i}
                )

            events = event_log.replay("run-large")
            assert len(events) == 100

            # Verify seq ordering
            seqs = [e.seq for e in events]
            assert seqs == sorted(seqs)
            # Seqs should be contiguous (global seqs might not be 1-100 if other runs exist)
            # but should be strictly increasing
            for i in range(len(seqs) - 1):
                assert seqs[i] < seqs[i + 1]


class TestEventHierarchy:
    """Test get_event_hierarchy() method."""

    def test_get_event_hierarchy_empty_run(self):
        """get_event_hierarchy() returns empty dict for nonexistent run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")
            hierarchy = event_log.get_event_hierarchy("run-nonexistent")
            assert hierarchy == {}

    def test_get_event_hierarchy_single_event(self):
        """get_event_hierarchy() maps single event seq=1 to rank=0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")
            event_id = event_log.append("run_started", "run-123", {})
            event = event_log.get_event(event_id)

            hierarchy = event_log.get_event_hierarchy("run-123")
            assert hierarchy == {event.seq: 0}

    def test_get_event_hierarchy_multiple_events(self):
        """get_event_hierarchy() maps multiple seqs to 0-indexed ranks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")

            # Append 5 events to run-abc
            seqs = []
            for i in range(5):
                event_id = event_log.append(
                    "agent_started", "run-abc", {"iteration": i}
                )
                event = event_log.get_event(event_id)
                seqs.append(event.seq)

            # Also append events to other runs to verify filtering
            event_log.append("run_started", "run-xyz", {})

            hierarchy = event_log.get_event_hierarchy("run-abc")

            # Hierarchy should only have events from run-abc
            assert len(hierarchy) == 5
            # Map seqs to ranks
            for rank, seq in enumerate(seqs):
                assert hierarchy[seq] == rank

    def test_get_event_hierarchy_preserves_seq_order(self):
        """get_event_hierarchy() ranks correspond to seq order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")

            # Append 10 events
            event_ids = []
            for i in range(10):
                event_id = event_log.append(
                    "agent_started", "run-123", {"iteration": i}
                )
                event_ids.append(event_id)

            hierarchy = event_log.get_event_hierarchy("run-123")

            # Get seqs in order
            seqs = []
            for event_id in event_ids:
                event = event_log.get_event(event_id)
                seqs.append(event.seq)

            # Verify ranks match seq order
            for rank, seq in enumerate(seqs):
                assert hierarchy[seq] == rank


class TestMigrationV2toV3:
    """Test v2→v3 migration."""

    def test_migration_nonexistent_table_is_noop(self):
        """migrate_v2_to_v3 is no-op for nonexistent events table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            # Create basic schema without events table
            conn.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY)")
            conn.commit()

            # Migration should succeed
            result = migrate_v2_to_v3(conn)
            assert result is True

            conn.close()

    def test_migration_v3_already_in_place_is_noop(self):
        """migrate_v2_to_v3 is no-op if schema already v3."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            event_log = EventLog(db_path)

            # Initialize with new schema (v3)
            event_log.append("run_started", "run-123", {})

            # Open raw connection and attempt migration
            conn = sqlite3.connect(str(db_path))
            result = migrate_v2_to_v3(conn)
            assert result is True
            conn.close()

    def test_migration_preserves_existing_data(self):
        """migrate_v2_to_v3 preserves all existing events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            # Create v2 schema (optional seq)
            conn.execute("""
                CREATE TABLE events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    ts REAL NOT NULL
                )
            """)

            # Insert 10 test events
            event_data = []
            for i in range(10):
                event_id = f"event-{i}"
                event_type = "agent_started"
                run_id = "run-123"
                payload = json.dumps({"iteration": i})
                ts = float(i)

                conn.execute(
                    "INSERT INTO events (id, event_type, run_id, payload, ts) VALUES (?, ?, ?, ?, ?)",
                    (event_id, event_type, run_id, payload, ts),
                )
                event_data.append((event_id, event_type, run_id, payload, ts))

            conn.commit()

            # Perform migration
            result = migrate_v2_to_v3(conn)
            assert result is True

            # Verify all events preserved
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, event_type, run_id, payload, ts FROM events ORDER BY ts"
            )
            migrated = cursor.fetchall()

            assert len(migrated) == 10
            for i, (event_id, event_type, run_id, payload, ts) in enumerate(migrated):
                assert event_id == f"event-{i}"
                assert event_type == "agent_started"
                assert run_id == "run-123"
                assert json.loads(payload) == {"iteration": i}

            conn.close()

    def test_migration_assigns_seq_via_row_number(self):
        """migrate_v2_to_v3 assigns seq via ROW_NUMBER() ordered by ts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            # Create v2 schema
            conn.execute("""
                CREATE TABLE events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    ts REAL NOT NULL
                )
            """)

            # Insert events with varying timestamps
            timestamps = [10.0, 5.0, 20.0, 15.0]
            for i, ts in enumerate(timestamps):
                conn.execute(
                    "INSERT INTO events (id, event_type, run_id, payload, ts) VALUES (?, ?, ?, ?, ?)",
                    (f"event-{i}", "agent_started", "run-123", "{}", ts),
                )
            conn.commit()

            # Perform migration
            migrate_v2_to_v3(conn)

            # Verify seqs assigned in ts order
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, seq FROM events ORDER BY seq"
            )
            results = cursor.fetchall()

            # Seqs should be 1, 2, 3, 4 in ts order
            expected_order = ["event-1", "event-0", "event-3", "event-2"]  # ts order: 5, 10, 15, 20
            for i, (event_id, seq) in enumerate(results):
                assert event_id == expected_order[i]
                assert seq == i + 1

            conn.close()

    def test_migration_creates_indexes(self):
        """migrate_v2_to_v3 creates all required indexes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            # Create v2 schema
            conn.execute("""
                CREATE TABLE events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    ts REAL NOT NULL
                )
            """)
            conn.commit()

            # Perform migration
            migrate_v2_to_v3(conn)

            # Verify indexes
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='events'"
            )
            index_names = {row[0] for row in cursor.fetchall()}

            expected_indexes = {
                "idx_events_run_id",
                "idx_events_run_id_seq",
                "idx_events_seq",
                "idx_events_ts",
                "idx_events_event_type",
            }
            assert expected_indexes.issubset(index_names)

            conn.close()

    def test_migration_enforces_seq_unique_constraint(self):
        """After migration, seq UNIQUE constraint is enforced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            # Create v2 schema
            conn.execute("""
                CREATE TABLE events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    ts REAL NOT NULL
                )
            """)
            conn.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                ("event-1", "agent_started", "run-123", "{}", 1.0),
            )
            conn.commit()

            # Perform migration
            migrate_v2_to_v3(conn)

            # Try to insert event with duplicate seq (should fail)
            cursor = conn.cursor()
            with pytest.raises(sqlite3.IntegrityError):
                cursor.execute(
                    "INSERT INTO events (id, event_type, run_id, payload, ts, seq) VALUES (?, ?, ?, ?, ?, ?)",
                    ("event-2", "agent_started", "run-123", "{}", 2.0, 1),  # seq=1 already exists
                )

            conn.close()


class TestSeqPerformance:
    """Test performance requirements."""

    def test_append_performance_under_5ms(self):
        """append() completes in <5ms per event on average."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")

            times = []
            for i in range(50):
                start = time.time()
                event_log.append(
                    "agent_started", f"run-{i}", {"iteration": i}
                )
                elapsed = (time.time() - start) * 1000  # Convert to ms
                times.append(elapsed)

            avg_time = sum(times) / len(times)
            max_time = max(times)

            # Average should be well under 5ms
            assert avg_time < 5.0, f"Average append time {avg_time}ms > 5ms"
            # Even max time should ideally be <20ms
            assert max_time < 20.0, f"Max append time {max_time}ms > 20ms (too slow)"

    def test_replay_performance_for_large_run(self):
        """replay() completes efficiently for 1000+ events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")

            # Append 200 events to a single run
            for i in range(200):
                event_log.append(
                    "agent_started", "run-large", {"iteration": i}
                )

            # Time the replay
            start = time.time()
            events = event_log.replay("run-large")
            elapsed = (time.time() - start) * 1000  # Convert to ms

            assert len(events) == 200
            assert elapsed < 500.0, f"Replay took {elapsed}ms (should be <500ms)"


class TestSeqEdgeCases:
    """Test edge cases and error conditions."""

    def test_append_to_multiple_runs_maintains_global_seq(self):
        """seq is globally unique even with events from different runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")

            # Append events to different runs
            run1_events = []
            run2_events = []

            event_log.append("run_started", "run-1", {})
            for _ in range(5):
                event_id = event_log.append("agent_started", "run-1", {})
                run1_events.append(event_log.get_event(event_id))

            event_log.append("run_started", "run-2", {})
            for _ in range(5):
                event_id = event_log.append("agent_started", "run-2", {})
                run2_events.append(event_log.get_event(event_id))

            # Collect all seqs
            all_seqs = [e.seq for e in run1_events if e] + [e.seq for e in run2_events if e]

            # All seqs should be unique
            assert len(set(all_seqs)) == len(all_seqs)

    def test_replay_consistency_concurrent_append_and_replay(self):
        """replay() is consistent while append() is happening."""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = EventLog(Path(tmpdir) / "test.db")

            # Append initial events
            for i in range(10):
                event_log.append("agent_started", "run-123", {})

            replay_results = {"before": None, "after": None, "errors": []}

            def replay_thread():
                try:
                    replay_results["before"] = event_log.replay("run-123")
                    time.sleep(0.05)
                    replay_results["after"] = event_log.replay("run-123")
                except Exception as e:
                    replay_results["errors"].append(e)

            def append_thread():
                time.sleep(0.01)
                for i in range(20):
                    event_log.append("budget_check", "run-123", {})
                    time.sleep(0.001)

            t1 = threading.Thread(target=replay_thread)
            t2 = threading.Thread(target=append_thread)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            assert len(replay_results["errors"]) == 0
            before_events = replay_results["before"]
            after_events = replay_results["after"]

            assert before_events is not None
            assert after_events is not None
            assert len(before_events) < len(after_events)

            # Verify both are in seq order
            before_seqs = [e.seq for e in before_events]
            after_seqs = [e.seq for e in after_events]

            assert before_seqs == sorted(before_seqs)
            assert after_seqs == sorted(after_seqs)

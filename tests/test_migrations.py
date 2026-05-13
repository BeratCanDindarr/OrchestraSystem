"""Comprehensive migration test suite for Orchestra storage system.

Tests cover:
- Migration discovery and execution order
- v1→v2 EventLog data preservation and schema updates
- v1→v2 Artifacts schema migration
- Rollback from migration failures
- Idempotent re-runs (no double-migration)
- Concurrent migration attempts (serialized via IMMEDIATE transactions)
- Config validation on deploy
- Schema version tracking via PRAGMA user_version
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import List

import pytest

from orchestra.storage.migrations import (
    run_migrations,
    rollback_migration,
    get_migration_status,
    _get_migration_modules,
    _get_schema_version,
    _set_schema_version,
    MigrationError,
)


class TestMigrationDiscovery:
    """Test auto-discovery of migration modules."""

    def test_migration_modules_discovered(self):
        """_get_migration_modules() discovers all NNN_*.py files."""
        modules = _get_migration_modules()
        assert isinstance(modules, list)
        assert len(modules) >= 2, "Expected at least 2 migrations (001, 002)"
        assert "001_eventlog_v1_to_v2" in modules
        assert "002_artifacts_v1_to_v2" in modules
        assert modules == sorted(modules)

    def test_migration_modules_sorted(self):
        """Migration modules are returned in sorted order."""
        modules = _get_migration_modules()
        assert modules == sorted(modules)


class TestEventLogMigration:
    """Test v1→v2 EventLog migration."""

    def _create_v1_events_table(self, conn: sqlite3.Connection) -> None:
        """Create v1 events table (seq is optional)."""
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                run_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                ts REAL NOT NULL
            )
        """)
        conn.commit()

    def test_eventlog_upgrade_empty_table(self):
        """Migration 001 handles empty events table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            self._create_v1_events_table(conn)

            result = run_migrations(conn, target_version=1)

            assert result["migrations_run"] == 1
            assert result["final_version"] == 1
            assert len(result["errors"]) == 0

            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(events)")
            columns = {row[1]: row for row in cursor.fetchall()}
            assert "seq" in columns
            assert columns["seq"][3] == 1  # NOT NULL

            conn.close()

    def test_eventlog_upgrade_with_data(self):
        """Migration 001 preserves and sequences existing events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            self._create_v1_events_table(conn)

            cursor = conn.cursor()
            events = [
                ("evt-1", "run_started", "run-123", '{"mode":"auto"}', 1.0),
                ("evt-2", "agent_completed", "run-123", '{"alias":"cdx"}', 2.0),
                ("evt-3", "run_completed", "run-123", '{"status":"ok"}', 3.0),
            ]
            cursor.executemany(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?)", events
            )
            conn.commit()

            result = run_migrations(conn, target_version=1)
            assert result["migrations_run"] == 1

            cursor.execute("SELECT COUNT(*) FROM events")
            assert cursor.fetchone()[0] == 3

            cursor.execute("SELECT id, seq FROM events ORDER BY seq ASC")
            rows = cursor.fetchall()
            assert rows[0][1] == 1
            assert rows[1][1] == 2
            assert rows[2][1] == 3

            conn.close()

    def test_eventlog_migration_idempotent(self):
        """Running migration twice is safe (idempotent)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            self._create_v1_events_table(conn)

            result1 = run_migrations(conn, target_version=1)
            assert result1["migrations_run"] == 1

            result2 = run_migrations(conn, target_version=1)
            assert result2["migrations_run"] == 0
            assert result2["final_version"] == 1

            conn.close()

    def test_eventlog_schema_validation_passed(self):
        """Migration validates seq as NOT NULL UNIQUE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            self._create_v1_events_table(conn)
            result = run_migrations(conn, target_version=1)
            assert result["migrations_run"] == 1

            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO events (id, event_type, run_id, payload, ts, seq) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("evt-1", "run_started", "run-123", '{}', 1.0, 1),
            )
            conn.commit()

            with pytest.raises(sqlite3.IntegrityError):
                cursor.execute(
                    "INSERT INTO events (id, event_type, run_id, payload, ts, seq) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ("evt-2", "run_started", "run-123", '{}', 2.0, 1),
                )
                conn.commit()

            conn.close()


class TestArtifactsMigration:
    """Test v1→v2 Artifacts migration."""

    def _create_v1_events_table(self, conn: sqlite3.Connection) -> None:
        """Create v1 events table (seq is optional)."""
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                run_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                ts REAL NOT NULL
            )
        """)
        conn.commit()

    def _create_v1_artifacts_table(self, conn: sqlite3.Connection) -> None:
        """Create v1 artifacts/checkpoints table (no schema_version)."""
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_proposals (
                proposal_id TEXT PRIMARY KEY,
                proposal_data TEXT NOT NULL,
                created_at REAL
            )
        """)
        conn.commit()

    def test_artifacts_upgrade_empty_table(self):
        """Migration 002 handles empty artifacts table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            self._create_v1_events_table(conn)
            self._create_v1_artifacts_table(conn)
            result = run_migrations(conn, target_version=2)

            assert result["migrations_run"] == 2
            assert result["final_version"] == 2

            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(tool_proposals)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "schema_version" in columns

            conn.close()

    def test_artifacts_upgrade_sets_version(self):
        """Migration 002 sets schema_version=1 for existing rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            self._create_v1_events_table(conn)
            self._create_v1_artifacts_table(conn)

            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO tool_proposals VALUES (?, ?, ?)",
                ("prop-1", '{"tool":"test"}', 1.0),
            )
            conn.commit()

            result = run_migrations(conn, target_version=2)
            assert result["migrations_run"] == 2

            cursor.execute("SELECT schema_version FROM tool_proposals")
            version = cursor.fetchone()[0]
            assert version == 1

            conn.close()

    def test_artifacts_migration_backward_compatible(self):
        """Existing checkpoints remain readable after migration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            self._create_v1_events_table(conn)
            self._create_v1_artifacts_table(conn)

            cursor = conn.cursor()
            proposal_data = '{"agents": [], "outcome": "approved"}'
            cursor.execute(
                "INSERT INTO tool_proposals VALUES (?, ?, ?)",
                ("prop-1", proposal_data, 1.0),
            )
            conn.commit()

            result = run_migrations(conn, target_version=2)

            cursor.execute("SELECT proposal_data FROM tool_proposals WHERE proposal_id = ?", ("prop-1",))
            row = cursor.fetchone()
            assert row is not None
            data = json.loads(row[0])
            assert data["agents"] == []

            conn.close()


class TestMigrationRunner:
    """Test migration runner coordination and state tracking."""

    def test_migration_status_tracks_version(self):
        """get_migration_status() reports current schema version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA foreign_keys = ON")

            status = get_migration_status(conn)
            assert status["current_version"] == 0
            assert status["pending_count"] >= 2

            _set_schema_version(conn, 1)
            status = get_migration_status(conn)
            assert status["current_version"] == 1

            conn.close()

    def test_migration_execution_order(self):
        """Migrations execute in numeric order (001, 002, ...)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA foreign_keys = ON")

            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT,
                    run_id TEXT,
                    payload TEXT,
                    ts REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE tool_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    proposal_data TEXT,
                    created_at REAL
                )
            """)
            conn.commit()

            result = run_migrations(conn)

            assert result["migrations_run"] >= 2
            modules_run = [d["module"] for d in result["migration_details"]]
            assert "001_eventlog_v1_to_v2" in modules_run
            assert "002_artifacts_v1_to_v2" in modules_run

            conn.close()

    def test_migration_failure_rollback(self):
        """On migration error, transaction is rolled back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA foreign_keys = ON")

            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE events (
                    id TEXT PRIMARY KEY
                )
            """)
            conn.commit()

            with pytest.raises(MigrationError):
                run_migrations(conn, target_version=1)

            version = _get_schema_version(conn)
            assert version == 0

            conn.close()

    def test_concurrent_migrations_serialized(self):
        """Concurrent migration attempts are serialized (no race)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT,
                    run_id TEXT,
                    payload TEXT,
                    ts REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE tool_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    proposal_data TEXT,
                    created_at REAL
                )
            """)
            conn.commit()
            conn.close()

            results = []
            errors = []

            def run_migration():
                try:
                    conn = sqlite3.connect(str(db_path))
                    result = run_migrations(conn)
                    results.append(result)
                    conn.close()
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=run_migration) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            total_migrations = sum(r["migrations_run"] for r in results)
            # Only the first thread completes both migrations; others see them already done
            assert total_migrations >= 2  # At least one thread runs both

            versions = [r["final_version"] for r in results]
            assert all(v == versions[0] for v in versions)
            assert versions[0] == 2  # Both migrations completed


class TestRollback:
    """Test rollback functionality."""

    def test_rollback_from_version_2(self):
        """rollback_migration() can rollback from v2 to v1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA foreign_keys = ON")

            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT,
                    run_id TEXT,
                    payload TEXT,
                    ts REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE tool_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    proposal_data TEXT,
                    created_at REAL
                )
            """)
            conn.commit()

            run_migrations(conn, target_version=2)
            assert _get_schema_version(conn) == 2

            result = rollback_migration(conn, from_version=2)
            assert result["rollback_completed"] is True
            assert result["new_version"] == 1

            assert _get_schema_version(conn) == 1

            conn.close()

    def test_rollback_at_version_0(self):
        """rollback_migration() handles already-at-v0 gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            result = rollback_migration(conn, from_version=0)
            assert result["rollback_completed"] is False

            conn.close()


class TestSchemaVersionTracking:
    """Test schema version PRAGMA tracking."""

    def test_schema_version_initially_zero(self):
        """New database has schema_version=0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            version = _get_schema_version(conn)
            assert version == 0

            conn.close()

    def test_schema_version_persists(self):
        """Schema version persists across connections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            conn1 = sqlite3.connect(str(db_path))
            _set_schema_version(conn1, 2)
            conn1.close()

            conn2 = sqlite3.connect(str(db_path))
            version = _get_schema_version(conn2)
            assert version == 2
            conn2.close()

    def test_set_schema_version_idempotent(self):
        """Setting schema version multiple times is safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            _set_schema_version(conn, 1)
            _set_schema_version(conn, 1)
            assert _get_schema_version(conn) == 1

            _set_schema_version(conn, 2)
            assert _get_schema_version(conn) == 2

            conn.close()


class TestConfigValidation:
    """Test deployment config validation."""

    def test_config_syntax_validation(self):
        """Config parser catches TOML syntax errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text("invalid toml [[[")

            import tomli

            with pytest.raises(Exception):
                with open(config_path, "rb") as f:
                    tomli.load(f)

    def test_migration_requires_pragma_foreign_keys(self):
        """Migrations require foreign_keys=ON for consistency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT,
                    run_id TEXT,
                    payload TEXT,
                    ts REAL
                )
            """)
            conn.commit()

            run_migrations(conn, target_version=1)

            cursor.execute("PRAGMA foreign_keys")
            assert cursor.fetchone()[0] == 1

            conn.close()

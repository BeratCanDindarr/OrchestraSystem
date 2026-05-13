"""Thread-safe EventLog with SQLite backend for atomic event storage and replay."""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from orchestra import config

logger = logging.getLogger(__name__)

# 30 valid event types per spec
VALID_EVENT_TYPES = {
    "run_started",
    "run_completed",
    "run_failed",
    "agent_started",
    "agent_completed",
    "agent_failed",
    "agent_retrying",
    "budget_check",
    "budget_exceeded",
    "approval_requested",
    "approval_approved",
    "approval_rejected",
    "suspension_created",
    "suspension_resumed",
    "checkpoint_written",
    "checkpoint_loaded",
    "synthesis_started",
    "synthesis_completed",
    "cache_hit",
    "cache_miss",
    "error_logged",
    "validation_failed",
    "span_started",
    "span_completed",
    "cost_tracked",
    "token_count_recorded",
    # 4 reserved for extensibility
    "ext_1",
    "ext_2",
    "ext_3",
    "ext_4",
}


class InvalidEventTypeError(Exception):
    """Raised when event_type is not in VALID_TYPES."""
    pass


class DatabaseError(Exception):
    """Raised on SQLite I/O failure."""
    pass


@dataclass(frozen=True)
class Event:
    """Immutable event record from database."""
    event_id: str
    event_type: str
    run_id: str
    payload: dict
    ts: float
    seq: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "run_id": self.run_id,
            "payload": self.payload,
            "ts": self.ts,
            "seq": self.seq,
        }


class EventLog:
    """
    Thread-safe event log with SQLite backend.

    Provides atomic append() and thread-safe replay() operations.
    Uses WAL mode + IMMEDIATE transactions for durability.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize EventLog with optional custom db_path.

        Args:
            db_path: Optional SQLite database path. Defaults to .orchestra/orchestra.db
        """
        self._db_path = db_path or config.db_path()
        self._ensure_directory()
        self._init_schema()

    def _ensure_directory(self) -> None:
        """Create parent directories if needed."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_schema(self) -> None:
        """Initialize database schema if not exists."""
        try:
            conn = self._get_connection()
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA busy_timeout = 5000")

            # Create events table with strict validation and seq AUTOINCREMENT
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    ts REAL NOT NULL DEFAULT (unixepoch('subsec')),
                    seq INTEGER UNIQUE NOT NULL,
                    created_at REAL DEFAULT (unixepoch('subsec')),
                    CHECK (event_type IN (
                        'run_started','run_completed','run_failed',
                        'agent_started','agent_completed','agent_failed','agent_retrying',
                        'budget_check','budget_exceeded',
                        'approval_requested','approval_approved','approval_rejected',
                        'suspension_created','suspension_resumed',
                        'checkpoint_written','checkpoint_loaded',
                        'synthesis_started','synthesis_completed',
                        'cache_hit','cache_miss',
                        'error_logged','validation_failed',
                        'span_started','span_completed',
                        'cost_tracked','token_count_recorded',
                        'ext_1','ext_2','ext_3','ext_4'
                    ))
                )
                """
            )

            # Create indexes for common queries and seq-based ordering
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_run_id_seq ON events(run_id, seq)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_seq ON events(seq)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type)")

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to initialize schema: {e}") from e

    def _get_connection(self) -> sqlite3.Connection:
        """Get a SQLite connection with proper configuration."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to connect to database: {e}") from e

    def append(self, event_type: str, run_id: str, payload: dict) -> str:
        """
        Append event atomically to the event log with auto-incremented seq.

        Args:
            event_type: One of VALID_EVENT_TYPES
            run_id: Unique run identifier
            payload: Arbitrary JSON-serializable dict

        Returns:
            event_id (UUID string)

        Raises:
            InvalidEventTypeError: If event_type not in VALID_TYPES
            DatabaseError: On SQLite I/O failure
        """
        # Validate event_type
        if event_type not in VALID_EVENT_TYPES:
            raise InvalidEventTypeError(
                f"Invalid event_type '{event_type}'. "
                f"Must be one of: {sorted(VALID_EVENT_TYPES)}"
            )

        # Generate event ID
        event_id = str(uuid.uuid4())

        # Serialize payload
        try:
            payload_json = json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            raise DatabaseError(f"Failed to serialize payload: {e}") from e

        # Atomic insert with atomic seq generation via RETURNING
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                # Use RETURNING to atomically get the next seq value.
                # SQLite 3.35+ supports RETURNING; for older versions,
                # we use a workaround with SELECT MAX(seq) inside the transaction.
                result = cursor.execute(
                    """
                    INSERT INTO events (id, event_type, run_id, payload, ts, seq)
                    VALUES (?, ?, ?, ?, unixepoch('subsec'), (SELECT COALESCE(MAX(seq), 0) + 1 FROM events))
                    RETURNING seq
                    """,
                    (event_id, event_type, run_id, payload_json),
                )

                row = result.fetchone()
                seq = row[0] if row else None

                if seq is None:
                    conn.rollback()
                    raise DatabaseError("Failed to retrieve seq after insert")

                conn.commit()
                logger.debug(f"Appended event: {event_id} ({event_type}) to run {run_id} with seq={seq}")
                return event_id
            except sqlite3.Error as e:
                conn.rollback()
                raise DatabaseError(f"Failed to insert event: {e}") from e
        finally:
            conn.close()

    def replay(self, run_id: str) -> List[Event]:
        """
        Load all events for a run_id in strict seq order.

        Thread-safe: reads only from the main database.
        Events are ordered by seq (global sequence number), ensuring
        strict consistency across concurrent appends.

        Args:
            run_id: Run identifier to replay

        Returns:
            List of Event namedtuples, ordered by seq ASC
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, event_type, run_id, payload, ts, seq
                FROM events
                WHERE run_id = ?
                ORDER BY seq ASC
                """,
                (run_id,),
            )
            events = []
            for row in cursor.fetchall():
                try:
                    payload = json.loads(row["payload"])
                except (json.JSONDecodeError, ValueError):
                    payload = {}
                events.append(
                    Event(
                        event_id=row["id"],
                        event_type=row["event_type"],
                        run_id=row["run_id"],
                        payload=payload,
                        ts=row["ts"],
                        seq=row["seq"],
                    )
                )
            return events
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to replay events: {e}") from e
        finally:
            conn.close()

    def get_event(self, event_id: str) -> Optional[Event]:
        """
        Fetch a single event by ID.

        Args:
            event_id: UUID of event to fetch

        Returns:
            Event if found, None otherwise
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, event_type, run_id, payload, ts, seq
                FROM events
                WHERE id = ?
                LIMIT 1
                """,
                (event_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            try:
                payload = json.loads(row["payload"])
            except (json.JSONDecodeError, ValueError):
                payload = {}
            return Event(
                event_id=row["id"],
                event_type=row["event_type"],
                run_id=row["run_id"],
                payload=payload,
                ts=row["ts"],
                seq=row["seq"],
            )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get event: {e}") from e
        finally:
            conn.close()

    def get_event_hierarchy(self, run_id: str) -> dict:
        """
        Return {seq: rank} mapping for events in a run.

        Maps seq (global sequence number) to rank (0-indexed position in run).
        Useful for understanding event ordering in multi-run scenarios.

        Example:
            run_abc events have seq: 1, 5, 12, 18
            Returns: {1: 0, 5: 1, 12: 2, 18: 3}

        Args:
            run_id: Run identifier

        Returns:
            Dictionary mapping seq → rank
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT seq, ROW_NUMBER() OVER (ORDER BY seq) as rank
                FROM events
                WHERE run_id = ?
                ORDER BY seq ASC
                """,
                (run_id,),
            )

            hierarchy = {}
            for row in cursor.fetchall():
                seq, rank = row
                hierarchy[seq] = rank - 1  # Convert to 0-indexed
            return hierarchy
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get event hierarchy: {e}") from e
        finally:
            conn.close()

    @property
    def db_path(self) -> str:
        """Return the SQLite database file path."""
        return str(self._db_path)

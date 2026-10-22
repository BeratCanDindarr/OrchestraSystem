"""The one way Orchestra opens a SQLite store (orchestra.db, memory.db, the event and replay logs).

Every connection gets the same settings: write-ahead logging, so readers do not block the
writer; a 10 s wait for a lock instead of failing with "database is locked" (runs, their steps
and the job worker write at the same time); and NORMAL sync, which WAL makes safe.
No imports from orchestra: every storage module may use it.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

BUSY_TIMEOUT_MS = 10_000


def connect(path: str | Path, *, row_factory=None) -> sqlite3.Connection:
    """Open the store at `path` with the shared settings. The caller closes it."""
    connection = sqlite3.connect(str(path), timeout=BUSY_TIMEOUT_MS / 1000)
    connection.row_factory = row_factory
    if str(path) != ":memory:":
        connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS};")
    return connection


@contextmanager
def transaction(path: str | Path, *, row_factory=None) -> Iterator[sqlite3.Connection]:
    """A connection that commits on success, rolls back on an error, and is always closed.

    `with sqlite3.connect(...) as conn` commits but never closes: every call left a connection open.
    """
    connection = connect(path, row_factory=row_factory)
    try:
        with connection:
            yield connection
    finally:
        connection.close()

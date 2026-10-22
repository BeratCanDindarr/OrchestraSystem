from __future__ import annotations
"""FtsSearchCache — Semantic search cache using SQLite FTS5 with BM25 scoring.

Caches prompt/response pairs with PII scrubbing to prevent information leakage.
Uses BM25 scoring for semantic similarity search and prompt deduplication via hashing
to reduce redundant API calls and token costs.
"""

import hashlib
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

from orchestra.engine.pii_scrubber import PiiScrubber


def _is_in_memory(db_path: str) -> bool:
    """Return True for SQLite in-memory sentinels that must never touch the filesystem."""
    return db_path == ":memory:" or db_path == "" or db_path.startswith("file:")


class CacheError(Exception):
    """Error during cache operations."""
    pass


@dataclass
class CacheResult:
    """Result of cache lookup via semantic search."""
    id: str
    run_id: str
    response: str
    score: float  # BM25 score (-N to 0, more negative = better)
    cost_saved_usd: float
    similarity: float  # Converted to 0-1 range


class FtsSearchCache:
    """FTS5-backed semantic cache with PII scrubbing and deduplication.

    Features:
    - SQLite FTS5 virtual table with BM25 scoring
    - Automatic PII scrubbing before storage (no sensitive data leaked)
    - Prompt deduplication via SHA-256 hash
    - Thread-safe concurrent access via BEGIN IMMEDIATE
    - Hit rate tracking based on duplicate prompts
    - TTL-based cleanup for expired entries
    """

    def __init__(self, db_path: str | None = None):
        """Initialize cache with SQLite FTS5 backend.

        Args:
            db_path: Path to SQLite database file (None to load from config)
        """
        if db_path is None:
            from orchestra import config
            db_path = config.fts_cache_db_path()
        import os
        if _is_in_memory(db_path):
            if db_path.startswith("file:"):
                self.conn = sqlite3.connect(db_path, uri=True, check_same_thread=False)
            else:
                self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            expanded = os.path.expanduser(db_path)
            dirname = os.path.dirname(expanded)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            self.conn = sqlite3.connect(expanded, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-64000")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA mmap_size=30000000")
        self._init_schema()
        self.pii_scrubber = PiiScrubber()

    def _init_schema(self):
        """Create FTS5 virtual table and metadata table if not exist."""
        cursor = self.conn.cursor()

        # FTS5 virtual table with porter tokenizer for English stemming
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_cache USING fts5(
                id UNINDEXED,
                run_id UNINDEXED,
                prompt_scrubbed,
                response_scrubbed,
                tokenize='porter'
            )
        """)

        # Metadata table for non-indexed fields and deduplication
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fts_cache_metadata (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                prompt_hash TEXT UNIQUE NOT NULL,
                cost_usd REAL DEFAULT 0.0,
                tokens INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                expires_at REAL
            )
        """)

        # Indices for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fts_cache_hash
            ON fts_cache_metadata(prompt_hash)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fts_cache_run_id
            ON fts_cache_metadata(run_id)
        """)

        self.conn.commit()

    @staticmethod
    def _entry_key(prompt_scrubbed: str, alias: str) -> str:
        """Key of an entry: the scrubbed prompt, and the alias that answered it when given."""
        keyed = f"{alias}\n{prompt_scrubbed}" if alias else prompt_scrubbed
        return hashlib.sha256(keyed.encode()).hexdigest()

    def lookup(self, prompt: str, alias: str, ttl_hours: float | None = None) -> Optional[CacheResult]:
        """Return the stored answer to exactly this prompt from exactly this alias.

        This is what a run may be answered from. `search` ranks entries by shared words and
        is not a test of "the same question": a short prompt whose words all occur in an
        earlier, different prompt matches it.

        Args:
            prompt: Raw prompt text (will be scrubbed)
            alias: The agent alias being asked
            ttl_hours: Ignore entries older than this; None means no age limit
        """
        key = self._entry_key(self.pii_scrubber.scrub_text(prompt), alias)
        row = self.conn.execute(
            """
            SELECT m.id, m.run_id, f.response_scrubbed, m.cost_usd, m.created_at
            FROM fts_cache_metadata m JOIN fts_cache f ON f.id = m.id
            WHERE m.prompt_hash = ?
            """,
            (key,),
        ).fetchone()
        if row is None:
            return None
        entry_id, run_id, response, cost_usd, created_at = row
        if ttl_hours is not None and created_at < time.time() - ttl_hours * 3600:
            return None
        return CacheResult(id=entry_id, run_id=run_id, response=response, score=0.0,
                           cost_saved_usd=cost_usd, similarity=1.0)

    def store(self, run_id: str, prompt: str, response: str,
              cost_usd: float = 0.0, tokens: int = 0, alias: str = "") -> str:
        """Store prompt/response pair after PII scrubbing.

        One entry per scrubbed prompt (and alias, when given). Without an alias an
        existing entry is kept and its ID returned; with an alias the newer answer
        replaces the stored one, so a run that asked again is what later runs get.

        Args:
            run_id: Run identifier for multi-run isolation
            prompt: Raw prompt text (will be scrubbed)
            response: Raw response text (will be scrubbed)
            cost_usd: Cost of this API execution
            tokens: Total tokens used (in + out)
            alias: The agent alias that gave the response

        Returns:
            Cache entry ID (UUID string)

        Raises:
            CacheError: If database operation fails
        """
        # Scrub PII before storage
        prompt_scrubbed = self.pii_scrubber.scrub_text(prompt)
        response_scrubbed = self.pii_scrubber.scrub_text(response)

        # Hash for deduplication (using scrubbed prompt to detect true duplicates)
        prompt_hash = self._entry_key(prompt_scrubbed, alias)

        # Check if already cached
        cursor = self.conn.cursor()
        existing = cursor.execute(
            "SELECT id FROM fts_cache_metadata WHERE prompt_hash = ?",
            (prompt_hash,)
        ).fetchone()

        if existing and not alias:
            return str(existing[0])

        # Insert new entry
        entry_id = str(uuid.uuid4())
        now = time.time()

        cursor.execute("BEGIN IMMEDIATE")
        try:
            if existing:
                # The same alias answered the same prompt again: the newer answer replaces the old.
                for table in ("fts_cache", "fts_cache_metadata"):
                    cursor.execute(f"DELETE FROM {table} WHERE id = ?", (existing[0],))
            # FTS5 virtual table (text columns for search)
            cursor.execute("""
                INSERT INTO fts_cache (id, run_id, prompt_scrubbed, response_scrubbed)
                VALUES (?, ?, ?, ?)
            """, (entry_id, run_id, prompt_scrubbed, response_scrubbed))

            # Metadata table (for deduplication and tracking)
            cursor.execute("""
                INSERT INTO fts_cache_metadata
                (id, run_id, prompt_hash, cost_usd, tokens, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (entry_id, run_id, prompt_hash, cost_usd, tokens, now))

            self.conn.commit()
            return entry_id

        except Exception as e:
            self.conn.rollback()
            raise CacheError(f"Failed to store cache entry: {str(e)}")


    def get_hit_rate(self, run_id: str) -> float:
        """Return cache hit rate for a run.

        Hit rate is computed as: (duplicate prompt uses) / (total entries)

        Example: If 10 entries exist but only 3 unique prompts, hit rate = 0.7
        (7 out of 10 were cache hits due to deduplication)

        Args:
            run_id: Run identifier to analyze

        Returns:
            Hit rate as float 0.0-1.0
        """
        cursor = self.conn.cursor()

        # Total entries for this run
        total = cursor.execute(
            "SELECT COUNT(*) FROM fts_cache_metadata WHERE run_id = ?",
            (run_id,)
        ).fetchone()[0]

        if total == 0:
            return 0.0

        # Count unique prompts
        unique = cursor.execute(
            "SELECT COUNT(DISTINCT prompt_hash) FROM fts_cache_metadata WHERE run_id = ?",
            (run_id,)
        ).fetchone()[0]

        # Hit rate: (total - unique) / total
        # Entries beyond the first per prompt = cache hits
        if unique == 0:
            return 0.0

        hit_count = total - unique
        return hit_count / total if total > 0 else 0.0

    def forget_run(self, run_id: str) -> None:
        """Remove the answers a run stored."""
        with self.conn:
            for table in ("fts_cache", "fts_cache_metadata"):
                self.conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))

    def cleanup_expired(self, ttl_hours: int = 168):
        """Remove cache entries older than TTL.

        Default TTL is 7 days (168 hours).

        Args:
            ttl_hours: Time-to-live in hours

        Raises:
            CacheError: If cleanup fails
        """
        cursor = self.conn.cursor()
        expires_before = time.time() - (ttl_hours * 3600)

        cursor.execute("BEGIN IMMEDIATE")
        try:
            # Get IDs to delete
            expired_ids = cursor.execute(
                "SELECT id FROM fts_cache_metadata WHERE created_at < ?",
                (expires_before,)
            ).fetchall()

            # Delete from both tables
            for (entry_id,) in expired_ids:
                cursor.execute("DELETE FROM fts_cache WHERE id = ?", (entry_id,))
                cursor.execute("DELETE FROM fts_cache_metadata WHERE id = ?", (entry_id,))

            self.conn.commit()

        except Exception as e:
            self.conn.rollback()
            raise CacheError(f"Cleanup failed: {str(e)}")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

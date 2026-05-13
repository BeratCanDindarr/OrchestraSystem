"""FtsSearchCache — Semantic search cache using SQLite FTS5 with BM25 scoring.

Caches prompt/response pairs with PII scrubbing to prevent information leakage.
Uses BM25 scoring for semantic similarity search and prompt deduplication via hashing
to reduce redundant API calls and token costs.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

from orchestra.engine.pii_scrubber import PiiScrubber


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

    def __init__(self, db_path: str = ":memory:"):
        """Initialize cache with SQLite FTS5 backend.

        Args:
            db_path: Path to SQLite database file (":memory:" for in-process)
        """
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
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

    def store(self, run_id: str, prompt: str, response: str,
              cost_usd: float = 0.0, tokens: int = 0) -> str:
        """Store prompt/response pair after PII scrubbing.

        Automatically deduplicates based on scrubbed prompt hash.
        If the same prompt (after scrubbing) was cached before, returns
        the existing cache entry ID instead of creating a duplicate.

        Args:
            run_id: Run identifier for multi-run isolation
            prompt: Raw prompt text (will be scrubbed)
            response: Raw response text (will be scrubbed)
            cost_usd: Cost of this API execution
            tokens: Total tokens used (in + out)

        Returns:
            Cache entry ID (UUID string)

        Raises:
            CacheError: If database operation fails
        """
        # Scrub PII before storage
        prompt_scrubbed = self.pii_scrubber.scrub_text(prompt)
        response_scrubbed = self.pii_scrubber.scrub_text(response)

        # Hash for deduplication (using scrubbed prompt to detect true duplicates)
        prompt_hash = hashlib.sha256(prompt_scrubbed.encode()).hexdigest()

        # Check if already cached
        cursor = self.conn.cursor()
        existing = cursor.execute(
            "SELECT id FROM fts_cache_metadata WHERE prompt_hash = ?",
            (prompt_hash,)
        ).fetchone()

        if existing:
            return existing[0]

        # Insert new entry
        entry_id = str(uuid.uuid4())
        now = time.time()

        cursor.execute("BEGIN IMMEDIATE")
        try:
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

    def search(self, query: str, topk: int = 3,
               score_threshold: float = -0.8) -> List[CacheResult]:
        """Search cache for similar prompts using BM25 scoring.

        Scrubs query PII before search, then returns BM25-scored results
        filtered by score threshold and sorted by relevance (highest first).

        Args:
            query: Query prompt text (will be scrubbed)
            topk: Maximum number of top results to return
            score_threshold: Minimum BM25 score (-1.0 to 0.0, default -0.8 for moderate matches).
                            BM25 scores range from -inf to 0, where closer to 0 = better match.
                            Use -2.0 for strong matches only.

        Returns:
            List of CacheResult sorted by score descending (most relevant first)
        """
        # Scrub query to match storage format
        query_scrubbed = self.pii_scrubber.scrub_text(query)

        cursor = self.conn.cursor()

        # FTS5 MATCH with BM25 scoring
        # bm25(fts_cache, 1) scores the 'response_scrubbed' column
        results = cursor.execute("""
            SELECT
                f.id,
                m.run_id,
                f.response_scrubbed,
                bm25(fts_cache, 1) as score,
                m.cost_usd,
                m.tokens
            FROM fts_cache f
            JOIN fts_cache_metadata m ON f.id = m.id
            WHERE fts_cache MATCH ?
            ORDER BY score DESC
            LIMIT ?
        """, (query_scrubbed, topk)).fetchall()

        cache_results = []
        for entry_id, run_id, response, bm25_score, cost_usd, tokens in results:
            # Filter by threshold (more negative = better, so check if below threshold)
            if bm25_score < score_threshold:
                continue

            # Convert BM25 score (-N to 0) to similarity (0 to 1)
            # Formula: 1 / (1 + |bm25_score|)
            # -0.8 → similarity ~0.556
            # -2.0 → similarity ~0.333
            # 0 → similarity = 1.0 (perfect match)
            similarity = 1.0 / (1.0 + abs(bm25_score))

            cache_results.append(CacheResult(
                id=entry_id,
                run_id=run_id,
                response=response,
                score=bm25_score,
                cost_saved_usd=cost_usd,
                similarity=similarity
            ))

        return cache_results

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

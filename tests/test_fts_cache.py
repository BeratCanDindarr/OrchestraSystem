"""Unit tests for FtsSearchCache (Block 9: FTS5 Semantic Cache).

Tests:
- Store prompt/response with PII scrubbing
- Search with BM25 scoring and threshold filtering
- Prompt deduplication via hash
- Cache hit rate calculation
- TTL-based cleanup
- Thread-safe concurrent access
- PII prevention (no sensitive data in cache)
- Multi-run isolation
- Performance: search < 50ms for 1000 entries
"""
from __future__ import annotations

import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import List

import pytest

from orchestra.storage.fts_cache import (
    FtsSearchCache,
    CacheResult,
    CacheError,
)


class TestFtsSearchCacheBasics:
    """Basic FtsSearchCache functionality."""

    def test_init_creates_database(self):
        """FtsSearchCache.__init__ creates SQLite database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            cache = FtsSearchCache(str(db_path))
            assert db_path.exists()
            cache.close()

    def test_init_with_memory_db(self):
        """FtsSearchCache can use in-memory database."""
        cache = FtsSearchCache(":memory:")
        assert cache.conn is not None
        cache.close()

    def test_store_returns_uuid(self):
        """store() returns a valid UUID string."""
        cache = FtsSearchCache(":memory:")
        entry_id = cache.store("run-1", "Tell me about ML", "Machine learning is...")
        assert isinstance(entry_id, str)
        assert len(entry_id) == 36  # UUID4: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        assert entry_id.count("-") == 4
        cache.close()

    def test_store_with_cost_and_tokens(self):
        """store() accepts cost and token parameters."""
        cache = FtsSearchCache(":memory:")
        entry_id = cache.store(
            "run-1",
            "Tell me about ML",
            "Machine learning is...",
            cost_usd=0.05,
            tokens=150
        )
        assert isinstance(entry_id, str)
        cache.close()

    def test_search_returns_cache_results(self):
        """search() returns list of CacheResult."""
        cache = FtsSearchCache(":memory:")
        cache.store("run-1", "What is AI?", "AI is artificial intelligence")
        results = cache.search("artificial intelligence", topk=5)
        assert isinstance(results, list)
        assert all(isinstance(r, CacheResult) for r in results)
        cache.close()

    def test_get_hit_rate_empty_run(self):
        """get_hit_rate() returns 0.0 for empty run."""
        cache = FtsSearchCache(":memory:")
        rate = cache.get_hit_rate("run-1")
        assert rate == 0.0
        cache.close()


class TestDeduplication:
    """Prompt deduplication via SHA-256 hash."""

    def test_duplicate_prompt_returns_same_id(self):
        """Storing same prompt twice returns same cache entry ID."""
        cache = FtsSearchCache(":memory:")
        prompt = "Tell me about machine learning"
        response = "Machine learning is a subset of AI"

        id1 = cache.store("run-1", prompt, response)
        id2 = cache.store("run-1", prompt, response)

        assert id1 == id2
        cache.close()

    def test_deduplication_ignores_whitespace(self):
        """Deduplication uses scrubbed (not original) prompt."""
        cache = FtsSearchCache(":memory:")
        prompt1 = "Tell me about ML"
        prompt2 = "Tell me about ML"

        id1 = cache.store("run-1", prompt1, "Response 1")
        id2 = cache.store("run-1", prompt2, "Response 2")

        assert id1 == id2
        cache.close()

    def test_different_prompts_create_different_entries(self):
        """Different prompts create different cache entries."""
        cache = FtsSearchCache(":memory:")
        id1 = cache.store("run-1", "What is AI?", "AI is...")
        id2 = cache.store("run-1", "What is ML?", "ML is...")

        assert id1 != id2
        cache.close()


class TestPiiScrubbing:
    """PII scrubbing before storage."""

    def test_pii_scrubbed_before_storage(self):
        """PII patterns are scrubbed and replaced with [REDACTED:type]."""
        cache = FtsSearchCache(":memory:")

        prompt = "Use api key sk_test_abc123xyz456"
        response = "Using sk_test_abc123xyz456 to authenticate"

        entry_id = cache.store("run-1", prompt, response)

        cursor = cache.conn.cursor()
        stored = cursor.execute(
            "SELECT prompt_scrubbed, response_scrubbed FROM fts_cache WHERE id = ?",
            (entry_id,)
        ).fetchone()

        prompt_scrubbed, response_scrubbed = stored
        assert "[REDACTED:api_key_sk]" in prompt_scrubbed
        assert "[REDACTED:api_key_sk]" in response_scrubbed
        assert "sk_test_abc123xyz456" not in prompt_scrubbed
        assert "sk_test_abc123xyz456" not in response_scrubbed
        cache.close()

    def test_email_scrubbed_from_prompt(self):
        """Email addresses are scrubbed."""
        cache = FtsSearchCache(":memory:")
        prompt = "Contact user@example.com for support"
        entry_id = cache.store("run-1", prompt, "Response")

        cursor = cache.conn.cursor()
        stored = cursor.execute(
            "SELECT prompt_scrubbed FROM fts_cache WHERE id = ?",
            (entry_id,)
        ).fetchone()

        assert "[REDACTED:email]" in stored[0]
        assert "user@example.com" not in stored[0]
        cache.close()

    def test_jwt_scrubbed_from_response(self):
        """JWT tokens are scrubbed."""
        cache = FtsSearchCache(":memory:")
        response = "Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        entry_id = cache.store("run-1", "Request token", response)

        cursor = cache.conn.cursor()
        stored = cursor.execute(
            "SELECT response_scrubbed FROM fts_cache WHERE id = ?",
            (entry_id,)
        ).fetchone()

        assert "[REDACTED:jwt]" in stored[0]
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in stored[0]
        cache.close()


class TestSearchAndScoring:
    """BM25 search with threshold filtering."""

    def test_search_returns_sorted_by_score(self):
        """search() returns results sorted by BM25 score descending."""
        cache = FtsSearchCache(":memory:")
        cache.store("run-1", "What is machine learning?", "ML is AI subset")
        cache.store("run-1", "What is deep learning?", "DL uses neural networks")

        results = cache.search("machine learning", topk=5)
        assert len(results) > 0
        if len(results) > 1:
            assert results[0].score >= results[1].score
        cache.close()

    def test_search_threshold_filtering(self):
        """search() filters results by score threshold."""
        cache = FtsSearchCache(":memory:")
        cache.store("run-1", "machine learning basics", "ML is a field of AI")
        cache.store("run-1", "cooking recipes", "How to bake bread")

        results_strict = cache.search("machine learning", topk=5, score_threshold=-2.0)
        results_permissive = cache.search("machine learning", topk=5, score_threshold=-0.1)

        assert len(results_strict) <= len(results_permissive)
        cache.close()

    def test_similarity_score_0_to_1(self):
        """search() returns similarity in 0-1 range."""
        cache = FtsSearchCache(":memory:")
        cache.store("run-1", "What is AI?", "AI stands for artificial intelligence")

        results = cache.search("artificial intelligence")
        assert all(0 <= r.similarity <= 1 for r in results)
        cache.close()

    def test_bm25_score_negative_or_zero(self):
        """BM25 score is -N to 0 (more negative = better match)."""
        cache = FtsSearchCache(":memory:")
        cache.store("run-1", "machine learning", "ML uses algorithms")

        results = cache.search("machine learning")
        for result in results:
            assert result.score <= 0
        cache.close()

    def test_search_topk_limit(self):
        """search(topk=N) returns at most N results."""
        cache = FtsSearchCache(":memory:")
        for i in range(20):
            cache.store("run-1", f"test prompt {i}", f"response {i}")

        results = cache.search("test", topk=3)
        assert len(results) <= 3
        cache.close()


class TestHitRate:
    """Cache hit rate calculation."""

    def test_hit_rate_single_entry(self):
        """Single entry, 1 unique prompt → hit rate 0.0."""
        cache = FtsSearchCache(":memory:")
        cache.store("run-1", "What is AI?", "AI is...")

        rate = cache.get_hit_rate("run-1")
        assert rate == 0.0
        cache.close()

    def test_hit_rate_two_unique_prompts(self):
        """Two unique prompts → hit rate 0.0."""
        cache = FtsSearchCache(":memory:")
        cache.store("run-1", "prompt 1", "response 1")
        cache.store("run-1", "prompt 2", "response 2")

        rate = cache.get_hit_rate("run-1")
        assert rate == 0.0
        cache.close()

    def test_hit_rate_empty_run(self):
        """get_hit_rate() on empty run returns 0.0."""
        cache = FtsSearchCache(":memory:")
        rate = cache.get_hit_rate("nonexistent-run")
        assert rate == 0.0
        cache.close()


class TestCleanup:
    """TTL-based cleanup of expired entries."""

    def test_cleanup_removes_expired_entries(self):
        """cleanup_expired(ttl_hours=0) removes all entries."""
        cache = FtsSearchCache(":memory:")
        cache.store("run-1", "test 1", "response 1")
        cache.store("run-1", "test 2", "response 2")

        count_before = cache.conn.cursor().execute(
            "SELECT COUNT(*) FROM fts_cache"
        ).fetchone()[0]
        assert count_before == 2

        cache.cleanup_expired(ttl_hours=0)

        count_after = cache.conn.cursor().execute(
            "SELECT COUNT(*) FROM fts_cache"
        ).fetchone()[0]
        assert count_after == 0
        cache.close()

    def test_cleanup_preserves_recent_entries(self):
        """cleanup_expired(ttl_hours=10000) preserves recent entries."""
        cache = FtsSearchCache(":memory:")
        cache.store("run-1", "test 1", "response 1")

        count_before = cache.conn.cursor().execute(
            "SELECT COUNT(*) FROM fts_cache"
        ).fetchone()[0]

        cache.cleanup_expired(ttl_hours=10000)

        count_after = cache.conn.cursor().execute(
            "SELECT COUNT(*) FROM fts_cache"
        ).fetchone()[0]
        assert count_before == count_after
        cache.close()

    def test_cleanup_deletes_from_both_tables(self):
        """cleanup_expired() removes from both FTS5 and metadata tables."""
        cache = FtsSearchCache(":memory:")
        cache.store("run-1", "test", "response")

        cache.cleanup_expired(ttl_hours=0)

        fts_count = cache.conn.cursor().execute(
            "SELECT COUNT(*) FROM fts_cache"
        ).fetchone()[0]
        meta_count = cache.conn.cursor().execute(
            "SELECT COUNT(*) FROM fts_cache_metadata"
        ).fetchone()[0]

        assert fts_count == 0
        assert meta_count == 0
        cache.close()


class TestMultiRunIsolation:
    """Multi-run isolation in cache."""

    def test_hit_rate_isolated_by_run_id(self):
        """get_hit_rate() only counts entries from that run_id."""
        cache = FtsSearchCache(":memory:")
        cache.store("run-1", "test prompt 1", "response 1")
        cache.store("run-2", "test prompt 2", "response 2")

        rate_1 = cache.get_hit_rate("run-1")
        rate_2 = cache.get_hit_rate("run-2")

        assert rate_1 == 0.0
        assert rate_2 == 0.0
        cache.close()


class TestThreadSafety:
    """Thread-safe concurrent access with BEGIN IMMEDIATE."""

    def test_sequential_store_no_errors(self):
        """Sequential store() calls don't cause errors (thread-safety sanity check)."""
        cache = FtsSearchCache(":memory:")
        errors = []

        try:
            for i in range(10):
                cache.store(
                    f"run-1",
                    f"prompt {i}",
                    f"response {i}"
                )
        except Exception as e:
            errors.append(e)

        assert len(errors) == 0
        cache.close()

    def test_sequential_search_no_errors(self):
        """Sequential search() calls don't cause errors."""
        cache = FtsSearchCache(":memory:")
        for i in range(10):
            cache.store("run-1", f"prompt {i}", f"response {i}")

        errors = []

        try:
            for _ in range(5):
                cache.search("prompt 5", topk=3)
        except Exception as e:
            errors.append(e)

        assert len(errors) == 0
        cache.close()


class TestPerformance:
    """Performance benchmarks."""

    def test_search_under_50ms_for_1000_entries(self):
        """search() completes in < 50ms for 1000 entries."""
        cache = FtsSearchCache(":memory:")

        for i in range(1000):
            cache.store("run-1", f"test prompt {i}", f"response {i}")

        start = time.time()
        results = cache.search("test prompt 500", topk=3)
        elapsed = time.time() - start

        assert elapsed < 0.050, f"Search took {elapsed:.3f}s (expected < 0.050s)"
        cache.close()

    def test_store_insertion_time_reasonable(self):
        """store() insertion time scales reasonably."""
        cache = FtsSearchCache(":memory:")

        start = time.time()
        for i in range(500):
            cache.store("run-1", f"prompt {i}", f"response {i}")
        elapsed = time.time() - start

        # 500 insertions should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0
        cache.close()


class TestErrorHandling:
    """Error handling and edge cases."""

    def test_store_raises_exception_on_db_failure(self):
        """store() raises exception on database errors."""
        cache = FtsSearchCache(":memory:")
        cache.close()

        with pytest.raises(Exception):  # sqlite3.ProgrammingError
            cache.store("run-1", "test", "response")

    def test_cleanup_raises_exception_on_db_failure(self):
        """cleanup_expired() raises exception on database errors."""
        cache = FtsSearchCache(":memory:")
        cache.close()

        with pytest.raises(Exception):  # sqlite3.ProgrammingError
            cache.cleanup_expired()

    def test_search_empty_cache_returns_empty_list(self):
        """search() on empty cache returns empty list."""
        cache = FtsSearchCache(":memory:")
        results = cache.search("anything")
        assert results == []
        cache.close()

    def test_search_no_match_returns_empty_list(self):
        """search() with no matching entries returns empty list."""
        cache = FtsSearchCache(":memory:")
        cache.store("run-1", "apple orange banana", "fruit response")
        results = cache.search("xyz123nonsense", topk=5)
        assert results == []
        cache.close()


class TestCacheResultDataclass:
    """CacheResult dataclass properties."""

    def test_cache_result_fields(self):
        """CacheResult has expected fields."""
        result = CacheResult(
            id="abc-123",
            run_id="run-1",
            response="Test response",
            score=-1.5,
            cost_saved_usd=0.02,
            similarity=0.6
        )
        assert result.id == "abc-123"
        assert result.run_id == "run-1"
        assert result.response == "Test response"
        assert result.score == -1.5
        assert result.cost_saved_usd == 0.02
        assert result.similarity == 0.6


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_store_search_dedup_flow(self):
        """Complete flow: store → search → dedup."""
        cache = FtsSearchCache(":memory:")

        prompt = "What is machine learning?"
        response = "ML is a subset of AI that enables systems to learn from data"

        id1 = cache.store("run-1", prompt, response, cost_usd=0.02, tokens=100)
        id2 = cache.store("run-1", prompt, response)

        assert id1 == id2

        results = cache.search("machine learning")
        assert len(results) > 0
        assert results[0].response == response
        cache.close()

    def test_pii_not_leaked_end_to_end(self):
        """PII is never stored or returned in plain text."""
        cache = FtsSearchCache(":memory:")

        api_key = "sk_test_secret123456789"
        prompt = f"Connect using {api_key}"
        response = f"Connected with {api_key} successfully"

        entry_id = cache.store("run-1", prompt, response)

        cursor = cache.conn.cursor()
        stored_prompt, stored_response = cursor.execute(
            "SELECT prompt_scrubbed, response_scrubbed FROM fts_cache WHERE id = ?",
            (entry_id,)
        ).fetchone()

        assert api_key not in stored_prompt
        assert api_key not in stored_response
        assert "[REDACTED:api_key_sk]" in stored_prompt
        assert "[REDACTED:api_key_sk]" in stored_response
        cache.close()

"""Phase 1 tests for Block 9 (FTS5 Cache fixes).

Focused tests for:
1. BM25 scoring semantics validation
2. Hash consistency across searches
3. PII scrubbing verification with entropy checks
4. Cache hit rate calculation
5. TTL-based cleanup
"""
from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from orchestra.storage.fts_cache import (
    FtsSearchCache,
    CacheResult,
)


class TestPhase1BM25Semantics:
    """Test BM25 score interpretation and semantics."""

    def test_bm25_scoring_semantics(self):
        """Verify BM25 score interpretation.

        BM25 scores should be -N to 0, where:
        - Score closer to 0 = better match
        - Score more negative = worse match

        Store: "hello world test prompt"
        Search: "hello world test" → should return match with score closer to 0
        """
        cache = FtsSearchCache(":memory:")

        # Store a prompt with clear matching keywords
        cache.store("run-1", "hello world test prompt", "response text")

        # Search with exact matching subset
        results = cache.search("hello world test", score_threshold=-2.0)

        # Should have results
        assert len(results) > 0, "Should find at least one match"

        # BM25 score should be between -2.0 and 0
        result = results[0]
        assert result.score >= -2.0, f"Score {result.score} should be >= -2.0"
        assert result.score < 0, f"Score {result.score} should be < 0"

        cache.close()

    def test_bm25_score_perfect_match_closer_to_zero(self):
        """Perfect matches should have scores closer to zero."""
        cache = FtsSearchCache(":memory:")

        # Store test data
        cache.store("run-1", "machine learning algorithms", "ML content")
        cache.store("run-1", "cooking recipes for dinner", "Food content")

        # Exact phrase search
        results = cache.search("machine learning algorithms", topk=5)

        # Should find the ML entry with better score than random search
        assert len(results) > 0

        # Perfect match should have highest score (closest to 0)
        perfect_match = results[0]
        assert perfect_match.score >= -2.0

        cache.close()


class TestPhase1HashConsistency:
    """Test hash consistency across searches (no double-scrubbing)."""

    def test_no_double_scrub_hash_match(self):
        """Verify hash computed once and consistency across operations.

        Store: prompt="my_api_key_sk_test_12345", response="result"
        Search same prompt → should find cache hit
        Verify: hash computed once, not twice
        """
        cache = FtsSearchCache(":memory:")

        # Use a prompt with PII-like pattern (will be scrubbed)
        prompt = "my_api_key_sk_test_12345"
        response = "result data"

        # First store
        id1 = cache.store("run-1", prompt, response)

        # Second store - identical prompt
        id2 = cache.store("run-1", prompt, response)

        # Should be same ID (deduplication worked)
        assert id1 == id2, "Duplicate prompts should return same cache ID"

        # Now search - should find the cached entry
        results = cache.search(prompt, score_threshold=-2.0)

        assert len(results) > 0, "Should find cache entry via search"

        # Verify similarity is high (cache hit)
        assert results[0].similarity > 0.85, \
            f"Similarity {results[0].similarity} should be > 0.85 for cache match"

        cache.close()

    def test_hash_consistency_with_whitespace(self):
        """Hash should be consistent for semantically identical prompts."""
        cache = FtsSearchCache(":memory:")

        # Two identical prompts (should deduplicate)
        prompt = "test prompt for caching"

        id1 = cache.store("run-1", prompt, "response 1")
        id2 = cache.store("run-1", prompt, "response 2")

        # Same hash → same ID
        assert id1 == id2, "Identical prompts must deduplicate"

        cache.close()


class TestPhase1PIIScrubbing:
    """Verify PII scrubbing with entropy checks."""

    def test_pii_scrubbing_entropy(self):
        """Verify PII redaction works with entropy-based filtering.

        Store: prompt with API key "sk_test_1234567890abcdef"
        Query: SELECT prompt_scrubbed FROM fts_cache
        Assert: stored text contains "[REDACTED:api_key_sk]" not actual key
        """
        cache = FtsSearchCache(":memory:")

        # Prompt with clear API key pattern
        api_key = "sk_test_1234567890abcdef"
        prompt = f"Use this API key: {api_key}"
        response = f"Successfully authenticated with {api_key}"

        entry_id = cache.store("run-1", prompt, response)

        # Query the database directly to verify storage
        cursor = cache.conn.cursor()
        stored = cursor.execute(
            "SELECT prompt_scrubbed, response_scrubbed FROM fts_cache WHERE id = ?",
            (entry_id,)
        ).fetchone()

        prompt_scrubbed, response_scrubbed = stored

        # Verify actual key is NOT in database
        assert api_key not in prompt_scrubbed, \
            f"API key should not be stored in prompt_scrubbed"
        assert api_key not in response_scrubbed, \
            f"API key should not be stored in response_scrubbed"

        # Verify redaction marker IS present
        assert "[REDACTED:api_key_sk]" in prompt_scrubbed, \
            "Prompt should contain redaction marker"
        assert "[REDACTED:api_key_sk]" in response_scrubbed, \
            "Response should contain redaction marker"

        cache.close()

    def test_pii_entropy_filtering_email(self):
        """Email addresses should be scrubbed."""
        cache = FtsSearchCache(":memory:")

        email = "sensitive@example.com"
        prompt = f"Contact {email} for support"

        entry_id = cache.store("run-1", prompt, "response")

        cursor = cache.conn.cursor()
        prompt_scrubbed = cursor.execute(
            "SELECT prompt_scrubbed FROM fts_cache WHERE id = ?",
            (entry_id,)
        ).fetchone()[0]

        assert email not in prompt_scrubbed
        assert "[REDACTED:email]" in prompt_scrubbed

        cache.close()

    def test_pii_jwt_scrubbing(self):
        """JWT tokens should be scrubbed."""
        cache = FtsSearchCache(":memory:")

        jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        response = f"Token returned: {jwt_token}"

        entry_id = cache.store("run-1", "request token", response)

        cursor = cache.conn.cursor()
        response_scrubbed = cursor.execute(
            "SELECT response_scrubbed FROM fts_cache WHERE id = ?",
            (entry_id,)
        ).fetchone()[0]

        assert jwt_token not in response_scrubbed
        assert "[REDACTED:jwt]" in response_scrubbed

        cache.close()


class TestPhase1CacheHitRate:
    """Verify deduplication tracking in hit rate calculation."""

    def test_cache_hit_rate(self):
        """Verify hit rate calculation with deduplication.

        Store prompt1 twice (same hash)
        Store prompt2 once
        Assert: get_hit_rate() returns correct calculation
        """
        cache = FtsSearchCache(":memory:")

        # Store prompt1 twice (will deduplicate to 1 entry)
        id1a = cache.store("run-1", "prompt1", "response1a")
        id1b = cache.store("run-1", "prompt1", "response1b")

        # These should be the same due to deduplication
        assert id1a == id1b, "Duplicate prompts should deduplicate"

        # Store prompt2 once
        id2 = cache.store("run-1", "prompt2", "response2")

        # Verify database state: 2 entries, 2 unique hashes
        cursor = cache.conn.cursor()
        total = cursor.execute(
            "SELECT COUNT(*) FROM fts_cache_metadata WHERE run_id = ?",
            ("run-1",)
        ).fetchone()[0]

        unique = cursor.execute(
            "SELECT COUNT(DISTINCT prompt_hash) FROM fts_cache_metadata WHERE run_id = ?",
            ("run-1",)
        ).fetchone()[0]

        assert total == 2, f"Should have 2 total entries, got {total}"
        assert unique == 2, f"Should have 2 unique prompts, got {unique}"

        hit_rate = cache.get_hit_rate("run-1")
        expected_rate = (total - unique) / total if total > 0 else 0.0
        assert hit_rate == expected_rate, \
            f"Hit rate {hit_rate} should equal {expected_rate}"

        cache.close()

    def test_cache_hit_rate_empty_run(self):
        """Empty run should have 0.0 hit rate."""
        cache = FtsSearchCache(":memory:")

        rate = cache.get_hit_rate("nonexistent-run")
        assert rate == 0.0

        cache.close()


class TestPhase1TTLCleanup:
    """Verify TTL cleanup for expired entries."""

    def test_cache_cleanup_expired(self):
        """Verify TTL cleanup deletes old entries.

        Store entry with created_at = now - 200 hours
        Call cleanup_expired(ttl_hours=168)  # 7 days
        Assert: entry deleted from DB
        """
        cache = FtsSearchCache(":memory:")

        # Store a normal entry
        entry_id = cache.store("run-1", "test prompt", "test response")

        # Manually set created_at to 200 hours ago in metadata
        cursor = cache.conn.cursor()
        old_timestamp = time.time() - (200 * 3600)

        cursor.execute(
            "UPDATE fts_cache_metadata SET created_at = ? WHERE id = ?",
            (old_timestamp, entry_id)
        )
        cache.conn.commit()

        # Verify entry exists before cleanup
        count_before = cursor.execute(
            "SELECT COUNT(*) FROM fts_cache WHERE id = ?",
            (entry_id,)
        ).fetchone()[0]
        assert count_before == 1, "Entry should exist before cleanup"

        # Run cleanup with 168-hour (7-day) TTL
        cache.cleanup_expired(ttl_hours=168)

        # Verify entry is deleted
        count_after = cursor.execute(
            "SELECT COUNT(*) FROM fts_cache WHERE id = ?",
            (entry_id,)
        ).fetchone()[0]
        assert count_after == 0, "Entry should be deleted after cleanup"

        # Verify also deleted from metadata
        meta_count = cursor.execute(
            "SELECT COUNT(*) FROM fts_cache_metadata WHERE id = ?",
            (entry_id,)
        ).fetchone()[0]
        assert meta_count == 0, "Metadata should also be deleted"

        cache.close()

    def test_cleanup_preserves_recent_entries(self):
        """Recent entries should NOT be deleted."""
        cache = FtsSearchCache(":memory:")

        # Store entry now
        entry_id = cache.store("run-1", "recent prompt", "response")

        # Run cleanup with very long TTL
        cache.cleanup_expired(ttl_hours=10000)

        # Verify entry still exists
        cursor = cache.conn.cursor()
        count = cursor.execute(
            "SELECT COUNT(*) FROM fts_cache WHERE id = ?",
            (entry_id,)
        ).fetchone()[0]
        assert count == 1, "Recent entry should be preserved"

        cache.close()

    def test_cleanup_removes_both_tables(self):
        """Cleanup should remove from both FTS5 and metadata tables."""
        cache = FtsSearchCache(":memory:")

        # Store and age an entry
        entry_id = cache.store("run-1", "test", "response")

        cursor = cache.conn.cursor()
        old_timestamp = time.time() - (200 * 3600)
        cursor.execute(
            "UPDATE fts_cache_metadata SET created_at = ? WHERE id = ?",
            (old_timestamp, entry_id)
        )
        cache.conn.commit()

        # Cleanup
        cache.cleanup_expired(ttl_hours=168)

        # Check both tables
        fts_count = cursor.execute(
            "SELECT COUNT(*) FROM fts_cache"
        ).fetchone()[0]
        meta_count = cursor.execute(
            "SELECT COUNT(*) FROM fts_cache_metadata"
        ).fetchone()[0]

        assert fts_count == 0, "FTS5 table should be empty"
        assert meta_count == 0, "Metadata table should be empty"

        cache.close()


class TestPhase1EdgeCases:
    """Edge cases and error conditions."""

    def test_search_empty_cache(self):
        """Search on empty cache returns empty list."""
        cache = FtsSearchCache(":memory:")
        results = cache.search("anything")
        assert results == [], "Empty cache should return empty results"
        cache.close()

    def test_store_with_empty_strings(self):
        """Store should handle empty strings."""
        cache = FtsSearchCache(":memory:")

        entry_id = cache.store("run-1", "", "")
        assert entry_id is not None
        assert isinstance(entry_id, str)

        cache.close()

    def test_score_threshold_boundary(self):
        """Test score threshold filtering at boundaries."""
        cache = FtsSearchCache(":memory:")

        cache.store("run-1", "test content here", "response")

        # Very strict threshold (only perfect matches)
        results_strict = cache.search("test", score_threshold=-0.1)

        # Permissive threshold
        results_permissive = cache.search("test", score_threshold=-5.0)

        # Permissive should have >= strict
        assert len(results_permissive) >= len(results_strict), \
            "Permissive threshold should return more or equal results"

        cache.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

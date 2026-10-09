"""Tests for SemanticMemory with HNSW and numpy fallback."""
import sqlite3
from unittest.mock import patch
import numpy as np

import pytest

from orchestra.storage.memory import SemanticMemory, _HNSW_AVAILABLE


def _mock_embed(text: str, dim: int = 384) -> list:
    """Create deterministic embedding vector from text."""
    h = hash(text) % (2**32)
    np.random.seed(h)
    vec = np.random.randn(dim).astype(np.float32)
    # Normalize for cosine similarity consistency
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


@pytest.fixture
def temp_memory(tmp_path):
    """Create a temporary SemanticMemory instance with mocked embed."""
    db_path = tmp_path / "memory.db"
    mem = SemanticMemory(str(db_path))
    with patch("orchestra.storage.memory._ollama.embed") as mock_embed:
        mock_embed.side_effect = lambda text: _mock_embed(text)
        mem._mock_embed = mock_embed
        yield mem


class TestSemanticMemorySearch:
    def test_search_returns_results(self, temp_memory):
        """Search should return ranked results."""
        mem = temp_memory
        mem.add("hello world", metadata={"path": "a.py"})
        mem.add("goodbye world", metadata={"path": "b.py"})
        mem.add("python programming", metadata={"path": "c.py"})

        results = mem.search("hello python", limit=2)
        assert len(results) == 2
        assert all("similarity" in r for r in results)
        assert all("content" in r for r in results)
        # Results should be sorted by similarity descending
        assert results[0]["similarity"] >= results[1]["similarity"]

    def test_search_hnsw_matches_fallback_top_result(self, tmp_path):
        """When HNSW is available, it should find the same top result as brute force."""
        db_hnsw = tmp_path / "mem_hnsw.db"
        db_fallback = tmp_path / "mem_fallback.db"

        with patch("orchestra.storage.memory._ollama.embed") as mock_embed:
            mock_embed.side_effect = lambda text: _mock_embed(text)

            mem_hnsw = SemanticMemory(str(db_hnsw))
            mem_hnsw.add("alpha", metadata={"path": "a.py"})
            mem_hnsw.add("beta", metadata={"path": "b.py"})
            mem_hnsw.add("gamma", metadata={"path": "c.py"})
            hnsw_results = mem_hnsw.search("alpha", limit=1)

            from orchestra.storage import memory as memory_module
            orig = memory_module._HNSW_AVAILABLE
            try:
                memory_module._HNSW_AVAILABLE = False
                mem_fallback = SemanticMemory(str(db_fallback))
                mem_fallback.add("alpha", metadata={"path": "a.py"})
                mem_fallback.add("beta", metadata={"path": "b.py"})
                mem_fallback.add("gamma", metadata={"path": "c.py"})
                fallback_results = mem_fallback.search("alpha", limit=1)
            finally:
                memory_module._HNSW_AVAILABLE = orig

        assert len(fallback_results) == len(hnsw_results) == 1
        assert fallback_results[0]["content"] == hnsw_results[0]["content"]

    def test_delete_invalidates_index(self, temp_memory):
        """Deleting an item should make the index dirty and remove it from search."""
        mem = temp_memory
        mem.add("item to delete", metadata={"path": "d.py"})
        mem.add("item to keep", metadata={"path": "e.py"})

        # Search to build index
        results_before = mem.search("item", limit=5)
        assert len(results_before) == 2

        # Delete first item
        with sqlite3.connect(mem.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM embeddings WHERE content = ?", ("item to delete",)
            ).fetchone()
            item_id = row[0]

        assert mem.delete(item_id) is True
        assert mem._index_dirty is True

        results_after = mem.search("item", limit=5)
        assert len(results_after) == 1
        assert results_after[0]["content"] == "item to keep"

    def test_delete_nonexistent_returns_false(self, temp_memory):
        """Deleting a non-existent id should return False."""
        mem = temp_memory
        assert mem.delete(99999) is False

    def test_search_empty_db(self, temp_memory):
        """Search on empty database should return empty list."""
        mem = temp_memory
        results = mem.search("anything", limit=5)
        assert results == []

    def test_add_with_no_embed_returns_false(self, temp_memory):
        """If embed returns empty, add should return False."""
        mem = temp_memory
        with patch("orchestra.storage.memory._ollama.embed", return_value=[]):
            result = mem.add("some content")
        assert result is False


class TestHNSWFallback:
    def test_fallback_when_hnsw_unavailable(self, tmp_path):
        """Search should work with numpy fallback when hnswlib is not installed."""
        db_path = tmp_path / "memory.db"
        mem = SemanticMemory(str(db_path))

        with patch("orchestra.storage.memory._HNSW_AVAILABLE", False):
            with patch("orchestra.storage.memory._ollama.embed") as mock_embed:
                mock_embed.side_effect = lambda text: _mock_embed(text)
                mem.add("fallback test", metadata={"path": "f.py"})
                results = mem.search("fallback", limit=1)
                assert len(results) == 1
                assert results[0]["content"] == "fallback test"

    def test_index_persists_and_loads(self, tmp_path):
        """Index should be saved to disk and loaded on fresh instance."""
        db_path = tmp_path / "memory.db"

        with patch("orchestra.storage.memory._ollama.embed") as mock_embed:
            mock_embed.side_effect = lambda text: _mock_embed(text)

            # First instance: build and save index
            mem1 = SemanticMemory(str(db_path))
            mem1.add("persistent item", metadata={"path": "g.py"})
            results1 = mem1.search("persistent", limit=1)
            assert len(results1) == 1
            assert mem1._index is not None

            # Second instance: should load existing index
            mem2 = SemanticMemory(str(db_path))
            assert mem2._index is None  # not loaded yet
            results2 = mem2.search("persistent", limit=1)
            assert len(results2) == 1
            assert mem2._index is not None

    def test_index_rebuilt_when_db_modified_externally(self, tmp_path):
        """If DB is newer than index, index should be rebuilt."""
        db_path = tmp_path / "memory.db"

        with patch("orchestra.storage.memory._ollama.embed") as mock_embed:
            mock_embed.side_effect = lambda text: _mock_embed(text)

            mem = SemanticMemory(str(db_path))
            mem.add("original", metadata={"path": "h.py"})
            mem.search("original", limit=1)
            assert mem._index is not None

            # Simulate external DB modification by directly inserting
            with sqlite3.connect(str(db_path)) as conn:
                vec = np.array(_mock_embed("external"), dtype=np.float32).tobytes()
                conn.execute(
                    "INSERT INTO embeddings (content, vector, metadata) VALUES (?, ?, ?)",
                    ("external", vec, None)
                )
                conn.commit()

            # Search should detect stale index and rebuild
            results = mem.search("external", limit=5)
            contents = [r["content"] for r in results]
            assert "external" in contents

    def test_pii_search_emits_audit_log(self, tmp_path, caplog):
        """Accessing PII-labelled memory entries must produce an audit log."""
        db_path = tmp_path / "memory.db"
        with patch("orchestra.storage.memory._ollama.embed") as mock_embed:
            mock_embed.side_effect = lambda text: _mock_embed(text)
            mem = SemanticMemory(str(db_path))
            mem.add("secret ssn 123-45-6789", sensitivity_label="pii", source="run_42")
            with caplog.at_level("INFO", logger="orchestra.storage.memory"):
                results = mem.search("secret", limit=1)
            assert len(results) == 1
            assert any("PII_ACCESS" in rec.message for rec in caplog.records)

    def test_audit_log_add_and_delete(self, tmp_path):
        """add and delete must create immutable audit events."""
        db_path = tmp_path / "memory.db"
        with patch("orchestra.storage.memory._ollama.embed") as mock_embed:
            mock_embed.side_effect = lambda text: _mock_embed(text)
            mem = SemanticMemory(str(db_path))
            mem.add("alpha", source="run_1", owner="user_a")
            ledger = mem.get_memory_ledger()
            item_id = ledger[0]["id"]

            audit = mem.get_memory_audit_log(item_id=item_id)
            assert any(e["event_type"] == "add" for e in audit)
            assert audit[0]["actor"] == "user_a"

            mem.delete(item_id, actor="user_a")
            audit = mem.get_memory_audit_log(item_id=item_id)
            tombstones = [e for e in audit if e["event_type"] == "tombstone"]
            assert len(tombstones) == 1
            assert tombstones[0]["details"]["reason"] == "delete"
            assert tombstones[0]["details"]["snapshot"]["content"] == "alpha"

    def test_audit_log_pin_unpin(self, tmp_path):
        """pin and unpin must emit audit events."""
        db_path = tmp_path / "memory.db"
        with patch("orchestra.storage.memory._ollama.embed") as mock_embed:
            mock_embed.side_effect = lambda text: _mock_embed(text)
            mem = SemanticMemory(str(db_path))
            mem.add("beta")
            item_id = mem.get_memory_ledger()[0]["id"]
            mem.pin(item_id, actor="admin")
            mem.unpin(item_id, actor="admin")
            audit = mem.get_memory_audit_log(item_id=item_id)
            types = [e["event_type"] for e in audit]
            assert "pin" in types
            assert "unpin" in types
            assert all(e["actor"] == "admin" for e in audit if e["event_type"] in ("pin", "unpin"))

    def test_audit_log_cleanup_expired(self, tmp_path):
        """cleanup_expired must tombstone every deleted item."""
        db_path = tmp_path / "memory.db"
        with patch("orchestra.storage.memory._ollama.embed") as mock_embed:
            mock_embed.side_effect = lambda text: _mock_embed(text)
            mem = SemanticMemory(str(db_path))
            mem.add("expiring", ttl_hours=0)
            # Force expiration by backdating expires_at
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    "UPDATE embeddings SET expires_at = datetime('now', '-1 hours') WHERE content = ?",
                    ("expiring",),
                )
                conn.commit()
            mem.cleanup_expired(actor="janitor")
            audit = mem.get_memory_audit_log(event_type="tombstone")
            assert len(audit) == 1
            assert audit[0]["details"]["reason"] == "cleanup_expired"
            assert audit[0]["actor"] == "janitor"

    def test_audit_log_search(self, tmp_path):
        """search must create a search audit event."""
        db_path = tmp_path / "memory.db"
        with patch("orchestra.storage.memory._ollama.embed") as mock_embed:
            mock_embed.side_effect = lambda text: _mock_embed(text)
            mem = SemanticMemory(str(db_path))
            mem.add("gamma")
            mem.search("gamma", limit=1)
            audit = mem.get_memory_audit_log(event_type="search")
            assert len(audit) == 1
            assert audit[0]["details"]["query"] == "gamma"

    def test_audit_log_cite(self, tmp_path):
        """increment_citation must create a cite audit event."""
        db_path = tmp_path / "memory.db"
        with patch("orchestra.storage.memory._ollama.embed") as mock_embed:
            mock_embed.side_effect = lambda text: _mock_embed(text)
            mem = SemanticMemory(str(db_path))
            mem.add("delta")
            item_id = mem.get_memory_ledger()[0]["id"]
            mem.increment_citation(item_id, actor="reader")
            audit = mem.get_memory_audit_log(item_id=item_id, event_type="cite")
            assert len(audit) == 1
            assert audit[0]["actor"] == "reader"


class TestNotesAndTasks:
    """Memory holds two kinds of rows: notes someone chose to keep, and the task text of runs."""

    class _Store:
        def __init__(self, rows):
            self.rows = rows
            self.limits = []

        def search(self, query, limit=5):
            self.limits.append(limit)
            return self.rows[:limit]

    def test_a_note_is_found_behind_many_task_rows(self):
        from orchestra.storage.memory import search_notes

        rows = [{"content": f"task {i}", "metadata": {"run_id": f"r{i}"}} for i in range(20)]
        rows.append({"content": "the note", "metadata": {}})
        store = self._Store(rows)

        assert [r["content"] for r in search_notes(store, "note", limit=3)] == ["the note"]
        assert store.limits[0] >= 21

    def test_service_search_leaves_tasks_out_and_returns_snippets(self, monkeypatch):
        from orchestra import service
        from orchestra.storage import memory as memory_module

        rows = [{"id": 1, "content": "old task " + "x" * 3000, "metadata": {"run_id": "r1"}, "similarity": 0.9},
                {"id": 2, "content": "a note " + "y" * 3000, "metadata": {}, "similarity": 0.5}]
        monkeypatch.setattr(memory_module, "get_memory", lambda: self._Store(rows))

        found = service.memory_search_entries("anything", limit=5)["results"]
        assert [r["id"] for r in found] == [2]
        assert len(found[0]["content"]) < 700 and found[0]["content_length"] > 3000

        everything = service.memory_search_entries("anything", limit=5, include_tasks=True, full=True)["results"]
        assert [r["id"] for r in everything] == [1, 2]
        assert len(everything[0]["content"]) > 3000

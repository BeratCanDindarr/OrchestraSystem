"""Tests for parameterized SemanticMemory, TypedMemory facade, and recency."""
import sqlite3
from datetime import datetime
from unittest.mock import patch
import numpy as np
import pytest

from orchestra import config
from orchestra.storage.memory import SemanticMemory


def _mock_embed(text: str, dim: int = 384) -> list:
    h = hash(text) % (2**32)
    np.random.seed(h)
    vec = np.random.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def test_custom_table_isolates_rows(tmp_path):
    """A SemanticMemory bound to a custom table writes only to that table."""
    db = str(tmp_path / "memory.db")
    with patch("orchestra.storage.memory._ollama.embed", side_effect=_mock_embed):
        mem = SemanticMemory(
            db, embeddings_table="embeddings_episodic",
            index_path=str(tmp_path / "epi_hnsw.idx"),
        )
        assert mem.add("an episode") is True
        results = mem.search("episode", limit=1)
        assert len(results) == 1 and results[0]["content"] == "an episode"
    with sqlite3.connect(db) as conn:
        epi = conn.execute("SELECT COUNT(*) FROM embeddings_episodic").fetchone()[0]
        # A store bound to a custom table creates ONLY that table — the default
        # 'embeddings' table is never touched by this instance.
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert epi == 1
    assert "embeddings_episodic" in tables
    assert "embeddings" not in tables


def test_recency_boost_decays_with_age(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_recency_enabled", lambda: True)
    monkeypatch.setattr(config, "memory_recency_half_life_hours", lambda: 10.0)
    monkeypatch.setattr(config, "memory_recency_weight", lambda: 0.2)
    fixed_now = datetime(2026, 7, 17, 12, 0, 0)
    mem = SemanticMemory(str(tmp_path / "m.db"), now_fn=lambda: fixed_now)
    fresh = mem._recency_boost("2026-07-17 12:00:00")   # age 0h
    one_half_life = mem._recency_boost("2026-07-17 02:00:00")  # age 10h == half-life
    assert fresh == pytest.approx(0.2)
    assert one_half_life == pytest.approx(0.1, abs=1e-6)


def test_recency_boost_zero_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_recency_enabled", lambda: False)
    mem = SemanticMemory(str(tmp_path / "m.db"))
    assert mem._recency_boost("2026-07-17 12:00:00") == 0.0


def test_plain_memory_ignores_memory_type(tmp_path):
    with patch("orchestra.storage.memory._ollama.embed", side_effect=_mock_embed):
        mem = SemanticMemory(str(tmp_path / "m.db"))
        assert mem.add("x content", memory_type="episodic") is True
        results = mem.search("x content", limit=1, memory_type="episodic")
        assert len(results) == 1 and results[0]["content"] == "x content"


def test_add_routes_to_typed_table(tmp_path):
    from orchestra.storage.typed_memory import TypedMemory
    db = str(tmp_path / "memory.db")
    with patch("orchestra.storage.memory._ollama.embed", side_effect=_mock_embed):
        tm = TypedMemory(db)
        assert tm.add("a lesson", memory_type="episodic") is True
        assert tm.add("a fact", memory_type="semantic") is True
    with sqlite3.connect(db) as conn:
        sem = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        epi = conn.execute("SELECT COUNT(*) FROM embeddings_episodic").fetchone()[0]
    assert sem == 1 and epi == 1


def test_search_merges_across_types(tmp_path):
    from orchestra.storage.typed_memory import TypedMemory
    with patch("orchestra.storage.memory._ollama.embed", side_effect=_mock_embed):
        tm = TypedMemory(str(tmp_path / "memory.db"))
        tm.add("alpha semantic", memory_type="semantic")
        tm.add("alpha episodic", memory_type="episodic")
        contents = {r["content"] for r in tm.search("alpha", limit=5)}
    assert "alpha semantic" in contents and "alpha episodic" in contents


def test_cross_store_write_does_not_invalidate_sibling_index(tmp_path):
    """Regression (review HIGH): writing to one typed store must NOT mark another
    store's HNSW index stale just because they share the memory.db file mtime."""
    from orchestra.storage.memory import _HNSW_AVAILABLE
    if not _HNSW_AVAILABLE:
        pytest.skip("hnswlib not installed")
    from orchestra.storage.typed_memory import TypedMemory
    with patch("orchestra.storage.memory._ollama.embed", side_effect=_mock_embed):
        tm = TypedMemory(str(tmp_path / "memory.db"))
        sem, epi = tm._stores["semantic"], tm._stores["episodic"]
        sem.add("a semantic fact")
        sem.search("semantic")          # builds + persists the semantic index
        assert sem._is_index_fresh() is True
        epi.add("an episodic event")    # bumps the shared memory.db mtime
        assert sem._is_index_fresh() is True  # sibling write must NOT invalidate


def test_search_filters_by_type(tmp_path):
    from orchestra.storage.typed_memory import TypedMemory
    with patch("orchestra.storage.memory._ollama.embed", side_effect=_mock_embed):
        tm = TypedMemory(str(tmp_path / "memory.db"))
        tm.add("only semantic", memory_type="semantic")
        tm.add("only episodic", memory_type="episodic")
        contents = {r["content"] for r in tm.search("only", limit=5, memory_type="episodic")}
    assert contents == {"only episodic"}


def test_get_memory_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from orchestra.storage.memory import get_memory
    from orchestra.storage.typed_memory import TypedMemory

    monkeypatch.setattr(config, "memory_typed_enabled", lambda: True)
    assert isinstance(get_memory(), TypedMemory)

    monkeypatch.setattr(config, "memory_typed_enabled", lambda: False)
    assert isinstance(get_memory(), SemanticMemory)


def test_record_lesson_tags_episodic(monkeypatch):
    from orchestra.engine import reflection
    monkeypatch.setattr(reflection.config, "reflection_record_lessons", lambda: True)
    captured = {}

    class FakeMem:
        def add(self, content, **kwargs):
            captured["content"] = content
            captured.update(kwargs)
            return True

    reflection._record_lesson("some task", ["was wrong"], "do better", FakeMem())
    assert captured["memory_type"] == "episodic"


def test_memory_add_entry_tags_semantic(monkeypatch):
    captured = {}

    class FakeMem:
        def add(self, content, metadata=None, summary=None, **kwargs):
            captured.update(kwargs)
            return True

    monkeypatch.setattr("orchestra.storage.memory.get_memory", lambda: FakeMem())
    from orchestra.service import memory_add_entry
    memory_add_entry("hello world")
    assert captured["memory_type"] == "semantic"

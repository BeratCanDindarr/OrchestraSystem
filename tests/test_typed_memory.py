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

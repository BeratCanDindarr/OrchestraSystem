"""Unit tests for orchestra.engine.rag (automatic retrieval + prompt-prefix formatting)."""
from __future__ import annotations

from unittest.mock import patch

from orchestra import config
from orchestra.engine import rag


class _FakeMemory:
    def __init__(self, results):
        self._results = results
        self.queries = []

    def search(self, query, limit=5):
        self.queries.append((query, limit))
        return self._results[:limit]


def test_rag_config_defaults():
    assert config.rag_enabled() is False
    assert config.rag_retrieval_limit() == 3
    assert config.rag_max_chars() == 2000


def test_retrieve_context_empty_when_disabled():
    mem = _FakeMemory([{"summary": "x", "similarity": 0.9}])
    # rag disabled by default -> no search performed at all
    assert rag.retrieve_context("q", memory=mem) == []
    assert mem.queries == []


def test_retrieve_context_returns_results_when_enabled():
    mem = _FakeMemory([{"summary": "prior work", "similarity": 0.91}])
    with patch("orchestra.config.rag_enabled", return_value=True):
        out = rag.retrieve_context("q", limit=2, memory=mem)
    assert out and out[0]["summary"] == "prior work"
    assert [query for query, _ in mem.queries] == ["q"]


def test_retrieve_context_swallows_search_errors():
    class Boom:
        def search(self, *a, **k):
            raise RuntimeError("embedding backend down")

    with patch("orchestra.config.rag_enabled", return_value=True):
        assert rag.retrieve_context("q", memory=Boom()) == []


def test_format_context_block_empty_for_no_results():
    assert rag.format_context_block([]) == ""
    # entries with no usable text -> empty
    assert rag.format_context_block([{"summary": "", "content": ""}]) == ""


def test_format_context_block_renders_entries():
    block = rag.format_context_block(
        [{"summary": "did X", "similarity": 0.88}, {"content": "did Y"}],
        max_chars=500,
    )
    assert block.startswith("[Retrieved Context")
    assert "did X" in block and "did Y" in block
    assert "similarity 0.88" in block
    assert block.endswith("\n\n")


def test_format_context_block_respects_max_chars():
    long_text = "z" * 5000
    block = rag.format_context_block([{"summary": long_text}], max_chars=100)
    assert len(block) <= 120           # cap + ellipsis + trailing newlines
    assert block.rstrip().endswith("…")


def test_format_context_block_shows_reflection_lesson():
    block = rag.format_context_block(
        [{"content": "some task", "similarity": 0.9,
          "metadata": {"reflection_lesson": "avoid: X. do Y"}}],
        max_chars=500,
    )
    assert "some task" in block
    assert "Lesson learned last time: avoid: X. do Y" in block


def test_format_context_block_shows_lesson_even_without_text():
    block = rag.format_context_block(
        [{"metadata": {"reflection_lesson": "avoid: Z"}}],
        max_chars=500,
    )
    assert "avoid: Z" in block


def test_rag_prefix_empty_when_disabled():
    mem = _FakeMemory([{"summary": "x", "similarity": 0.9}])
    assert rag.rag_prefix("q", memory=mem) == ""


def test_rag_prefix_builds_block_when_enabled():
    mem = _FakeMemory([{"summary": "relevant prior task", "similarity": 0.9}])
    with patch("orchestra.config.rag_enabled", return_value=True):
        prefix = rag.rag_prefix("q", memory=mem)
    assert "relevant prior task" in prefix and prefix.endswith("\n\n")

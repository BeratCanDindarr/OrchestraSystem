"""Tests for Tiered Context Memory — ContextMemory three-layer degradation."""
from __future__ import annotations

import pytest

from orchestra.engine.context_memory import ContextMemory


def _counter(text: str) -> int:
    """Simple token estimator: 1 token per 4 chars."""
    return len(text) // 4


def _make_note(alias: str, content: str) -> dict:
    return {"alias": alias, "content": content, "tags": [], "ts": "2026-05-08T12:00:00+00:00"}


ROLE = "You are a senior engineer."
PRIORITY = "[POLICY]\nFollow project rules.\n"
PROMPT = "Fix the bug in payment.py."


# ---------------------------------------------------------------------------
# Layer 1: Protected — no notes
# ---------------------------------------------------------------------------

def test_no_notes_returns_protected():
    ctx = ContextMemory([], recent_n=10)
    result = ctx.build(
        role=ROLE,
        priority_context=PRIORITY,
        prompt=PROMPT,
        max_tokens=9999,
        token_counter=_counter,
    )
    assert ROLE in result
    assert PROMPT in result
    assert "BLACKBOARD" not in result


# ---------------------------------------------------------------------------
# Layer 2: Recent — fits in budget
# ---------------------------------------------------------------------------

def test_recent_notes_included_when_budget_allows():
    notes = [_make_note("cdx-deep", "cache miss on key auth_token")]
    ctx = ContextMemory(notes, recent_n=10)
    result = ctx.build(
        role=ROLE,
        priority_context=PRIORITY,
        prompt=PROMPT,
        max_tokens=9999,
        token_counter=_counter,
    )
    assert "cache miss on key auth_token" in result
    assert "GLOBAL BLACKBOARD" in result


def test_recent_limits_to_recent_n():
    notes = [_make_note("a", f"note {i}") for i in range(20)]
    ctx = ContextMemory(notes, recent_n=5)
    result = ctx.build(
        role=ROLE,
        priority_context=PRIORITY,
        prompt=PROMPT,
        max_tokens=9999,
        token_counter=_counter,
    )
    assert "note 19" in result
    assert "note 15" in result
    assert "note 0" not in result
    assert "note 9" not in result


# ---------------------------------------------------------------------------
# Layer 3: Compressed — recent doesn't fit, use summarize_fn
# ---------------------------------------------------------------------------

def test_compressed_layer_uses_summarize_fn():
    big_content = "x" * 2000
    notes = [_make_note("cdx-deep", big_content)]
    ctx = ContextMemory(notes, recent_n=10)

    summary_called = []

    def mock_summarize(text: str) -> str:
        summary_called.append(text)
        return "Short summary: found a big cache issue."

    result = ctx.build(
        role=ROLE,
        priority_context=PRIORITY,
        prompt=PROMPT,
        max_tokens=100,
        token_counter=_counter,
        summarize_fn=mock_summarize,
    )
    assert summary_called, "summarize_fn should have been called"
    assert ROLE in result


def test_summarize_fn_exception_falls_back():
    big_content = "y" * 3000
    notes = [_make_note("gmn-pro", big_content)]
    ctx = ContextMemory(notes, recent_n=10)

    def failing_summarize(text: str) -> str:
        raise RuntimeError("LLM unavailable")

    result = ctx.build(
        role=ROLE,
        priority_context=PRIORITY,
        prompt=PROMPT,
        max_tokens=50,
        token_counter=_counter,
        summarize_fn=failing_summarize,
    )
    assert ROLE in result
    assert PROMPT in result


# ---------------------------------------------------------------------------
# Layer 1 fallback
# ---------------------------------------------------------------------------

def test_falls_back_to_protected_when_all_else_fails():
    notes = [_make_note("cdx-deep", "z" * 500)]
    ctx = ContextMemory(notes, recent_n=10)
    result = ctx.build(
        role=ROLE,
        priority_context=PRIORITY,
        prompt=PROMPT,
        max_tokens=20,
        token_counter=_counter,
        summarize_fn=None,
    )
    assert ROLE in result
    assert PROMPT in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_notes_no_summarize_called():
    summarize_called = []
    ctx = ContextMemory([], recent_n=10)
    result = ctx.build(
        role=ROLE,
        priority_context=PRIORITY,
        prompt=PROMPT,
        max_tokens=9999,
        token_counter=_counter,
        summarize_fn=lambda t: summarize_called.append(t) or "summary",
    )
    assert not summarize_called
    assert PROMPT in result

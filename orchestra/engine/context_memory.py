"""Tiered Context Memory — three degradation layers for agent prompt assembly.

Layers (degraded in order when token budget is exceeded):
  1. Protected  — role + system policies + current prompt (always fits)
  2. Recent     — Protected + last N blackboard events (raw, no truncation)
  3. Compressed — Protected + LLM-summarized blackboard (when Recent > budget)

Usage (called from runner._prefixed):
    ctx = ContextMemory(notes, recent_n=10)
    prompt = ctx.build(
        role=role,
        priority_context=priority,
        prompt=task_prompt,
        max_tokens=max_tokens,
        token_counter=_estimate_tokens,
        summarize_fn=lambda text: _llm_summarize(text),  # optional
    )
"""
from __future__ import annotations

from typing import Callable, Sequence


class ContextMemory:
    """Assembles agent prompts using a 3-tier degradation strategy."""

    def __init__(
        self,
        notes: Sequence[dict],
        *,
        recent_n: int = 10,
    ) -> None:
        self._notes = list(notes)
        self._recent_n = recent_n

    def _format_notes(self, notes: list[dict]) -> str:
        if not notes:
            return ""
        lines = ["\n[GLOBAL BLACKBOARD]\n"]
        for note in notes:
            lines.append(f"- [{note.get('alias', '?')}]: {note.get('content', '')}")
        return "\n".join(lines) + "\n"

    def build(
        self,
        *,
        role: str,
        priority_context: str,
        prompt: str,
        max_tokens: int,
        token_counter: Callable[[str], int],
        summarize_fn: Callable[[str], str] | None = None,
    ) -> str:
        """Build the final prompt string, degrading context layers to fit max_tokens."""
        protected = f"{role}\n\n{priority_context}\n\n{prompt}"

        if not self._notes:
            return protected

        # ── Layer 2: Recent (last N notes, full text) ──────────────────────
        recent_notes = self._notes[-self._recent_n :]
        recent_block = self._format_notes(recent_notes)
        candidate_recent = f"{role}\n\n{priority_context}{recent_block}\n\n{prompt}"
        if token_counter(candidate_recent) <= max_tokens:
            return candidate_recent

        # ── Layer 3: Compressed (LLM-summarized or truncated fallback) ─────
        if summarize_fn is not None:
            try:
                raw_text = self._format_notes(self._notes)
                summary = summarize_fn(raw_text)
            except Exception:
                summary = self._truncate_fallback(recent_block, max_tokens, token_counter, protected)
        else:
            summary = self._truncate_fallback(recent_block, max_tokens, token_counter, protected)

        if summary:
            compressed_block = f"\n[BLACKBOARD SUMMARY]\n{summary}\n"
            candidate_compressed = f"{role}\n\n{priority_context}{compressed_block}\n\n{prompt}"
            if token_counter(candidate_compressed) <= max_tokens:
                return candidate_compressed

        # ── Layer 1: Protected only (drop all blackboard) ──────────────────
        return protected

    def _truncate_fallback(
        self,
        block: str,
        max_tokens: int,
        token_counter: Callable[[str], int],
        protected: str,
    ) -> str:
        """Simple line-based truncation when LLM summary is unavailable."""
        lines = [line for line in block.splitlines() if line.strip()]
        kept: list[str] = []
        budget = max_tokens - token_counter(protected) - 50  # 50 token headroom
        total = 0
        for line in reversed(lines):
            total += len(line) // 4 + 1  # rough token estimate
            if total > budget:
                break
            kept.append(line)
        kept.reverse()
        return "\n".join(kept) if kept else ""

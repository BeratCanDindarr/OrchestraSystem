"""Automatic RAG: retrieve relevant prior context from semantic memory and format it as a
bounded prompt-prefix block, injected before each agent run.

Config-gated (``rag.enabled``, default off). The memory handle is injectable so retrieval is
testable without a real DB / embedding backend. Retrieval failures (e.g. embedding backend
down) degrade to an empty block rather than breaking the run.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from orchestra import config

logger = logging.getLogger(__name__)


def retrieve_context(prompt: str, *, limit: Optional[int] = None, memory: Any = None) -> list[dict]:
    """Return up to ``limit`` relevant memory entries for ``prompt``.

    Empty list when RAG is disabled, memory is unavailable, or search fails.
    """
    if not config.rag_enabled():
        return []
    limit = limit if limit is not None else config.rag_retrieval_limit()
    try:
        if memory is None:
            from orchestra.storage.memory import get_memory
            memory = get_memory()
        return memory.search(prompt, limit=limit) or []
    except Exception:
        logger.warning("RAG retrieval failed", exc_info=True)
        return []


def format_context_block(results: list[dict], *, max_chars: Optional[int] = None) -> str:
    """Format retrieved entries into a bounded ``[Retrieved Context]`` prefix block.

    Returns "" when there is nothing usable. The block is advisory: the model is told it may
    ignore it, so a spurious retrieval does not derail the task.
    """
    if not results:
        return ""
    max_chars = max_chars if max_chars is not None else config.rag_max_chars()
    lines = ["[Retrieved Context — relevant prior work; use if helpful, ignore if not]"]
    for i, res in enumerate(results, 1):
        text = (res.get("summary") or res.get("content") or "").strip()
        lesson = (res.get("metadata") or {}).get("reflection_lesson")
        if not text and not lesson:
            continue
        sim = res.get("similarity")
        tag = f" (similarity {sim:.2f})" if isinstance(sim, (int, float)) else ""
        entry = f"{i}.{tag} {text}".rstrip()
        if lesson:
            entry += f"\n   Lesson learned last time: {lesson}"
        lines.append(entry)
    if len(lines) == 1:                      # header only, no usable entries
        return ""
    block = "\n".join(lines)
    if len(block) > max_chars:
        block = block[:max_chars].rstrip() + " …"
    return block + "\n\n"


def rag_prefix(prompt: str, *, memory: Any = None) -> str:
    """Retrieve + format context for ``prompt``; "" when disabled or nothing relevant."""
    return format_context_block(retrieve_context(prompt, memory=memory))

"""PASTE — Speculative Tool Execution prefetch.

Prefetches results for read-only tools that are likely to be needed,
based on intent analysis of the task. Runs in background before the
main agent starts, so results are warm in cache when agent calls them.

SAFETY CONTRACT:
  - ONLY read-only tools are ever executed speculatively
  - Write/mutate/delete tools are NEVER called here
  - Results are cached in a SpeculativeCache; agents read from cache first
  - If agent calls a tool not in cache, it falls through to normal execution
  - Cache is scoped per run_id and discarded after run completes

Supported read-only tool categories (configurable via [speculation] in config):
  filesystem:   read_file, list_dir, glob, grep
  search:       find_symbol, search_code
  editor_state: get_editor_state, read_console (Unity)
  git:          git_log, git_status, git_diff (read-only)

Usage (called from runner.py before dispatching agents):
    executor = SpeculativeExecutor(run_id=run.run_id)
    executor.prefetch(task)
    # ... later, agent reads from executor.cache
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

from orchestra import config


# ---------------------------------------------------------------------------
# Read-only tool registry
# ---------------------------------------------------------------------------

_READ_ONLY_TOOLS: frozenset[str] = frozenset(
    {
        # Filesystem
        "read_file", "list_dir", "glob", "grep", "find_file",
        # Code navigation
        "find_symbol", "find_declaration", "find_implementations",
        "find_referencing_symbols", "search_for_pattern",
        "get_symbols_overview",
        # Editor state (Unity MCP)
        "get_editor_state", "read_console", "project_info",
        # Git (read-only)
        "git_log", "git_status", "git_diff",
        # Web/docs (read-only)
        "web_fetch", "web_search", "unity_docs",
    }
)

# Intent → likely tools mapping (heuristic prefetch hints)
_INTENT_TOOL_MAP: dict[str, list[str]] = {
    "find":    ["find_file", "find_symbol", "search_for_pattern"],
    "read":    ["read_file", "get_symbols_overview"],
    "list":    ["list_dir", "glob"],
    "search":  ["search_for_pattern", "grep", "find_symbol"],
    "explain": ["read_file", "find_symbol", "get_symbols_overview"],
    "review":  ["read_file", "find_symbol", "grep"],
    "debug":   ["read_console", "read_file", "find_symbol"],
    "unity":   ["get_editor_state", "read_console", "find_file"],
    "docs":    ["unity_docs", "web_fetch"],
    "git":     ["git_log", "git_status", "git_diff"],
}

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "find":    ["find", "where", "locate", "which file", "bul"],
    "read":    ["show", "what is", "read", "content", "oku"],
    "list":    ["list", "all files", "enumerate"],
    "search":  ["search", "grep", "contains", "ara"],
    "explain": ["explain", "what does", "how does", "açıkla", "nasıl"],
    "review":  ["review", "incele", "check", "look at"],
    "debug":   ["error", "bug", "fix", "hata", "crash", "fail"],
    "unity":   ["unity", "editor", "scene", "gameobject", "component"],
    "docs":    ["docs", "documentation", "api", "reference"],
    "git":     ["commit", "git", "history", "blame", "branch"],
}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@dataclass
class SpeculativeCacheEntry:
    tool: str
    args: dict
    result: object
    error: str | None = None
    hit: bool = False


@dataclass
class SpeculativeCache:
    """Per-run in-memory cache of speculatively fetched tool results."""
    run_id: str
    _entries: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _key(self, tool: str, args: dict) -> str:
        import json
        return f"{tool}:{json.dumps(args, sort_keys=True)}"

    def put(self, tool: str, args: dict, result: object, error: str | None = None) -> None:
        key = self._key(tool, args)
        with self._lock:
            self._entries[key] = SpeculativeCacheEntry(
                tool=tool, args=args, result=result, error=error
            )

    def get(self, tool: str, args: dict) -> SpeculativeCacheEntry | None:
        key = self._key(tool, args)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.hit = True
            return entry

    def stats(self) -> dict:
        with self._lock:
            total = len(self._entries)
            hits = sum(1 for e in self._entries.values() if e.hit)
        return {"total": total, "hits": hits, "miss": total - hits}


# ---------------------------------------------------------------------------
# Intent extraction
# ---------------------------------------------------------------------------

def extract_intent_hints(task: str) -> list[str]:
    """Return list of intent keys that match the task text."""
    text = task.lower()
    matched: list[str] = []
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            matched.append(intent)
    return matched


def tools_for_intents(intents: list[str]) -> list[str]:
    """Map intent keys to a deduplicated list of read-only tool names."""
    seen: set[str] = set()
    result: list[str] = []
    for intent in intents:
        for tool in _INTENT_TOOL_MAP.get(intent, []):
            if tool not in seen and tool in _READ_ONLY_TOOLS:
                seen.add(tool)
                result.append(tool)
    return result


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class SpeculativeExecutor:
    """Prefetch read-only tool results before the main agent runs.

    Tools are executed in background daemon threads so they don't block
    the main agent dispatch. Results are stored in self.cache.
    """

    def __init__(self, run_id: str, max_tools: int | None = None, enabled: bool | None = None) -> None:
        self.run_id = run_id
        self.cache = SpeculativeCache(run_id=run_id)
        cfg = config.section("speculation") if hasattr(config, "section") else {}
        self._max_tools = max_tools if max_tools is not None else int(cfg.get("max_prefetch_tools", 5))
        if enabled is not None:
            self._enabled = enabled
        else:
            self._enabled = bool(cfg.get("enabled", True))
        self._tool_executors: dict[str, Callable] = {}

    def register_tool(self, name: str, fn: Callable) -> None:
        """Register an executable tool function. Only read-only tools accepted."""
        if name in _READ_ONLY_TOOLS:
            self._tool_executors[name] = fn

    def register_tool_unsafe(self, name: str, fn: Callable) -> None:
        """Register a tool without read-only guard. Internal use only — do not expose."""
        raise RuntimeError(
            f"register_tool_unsafe is forbidden. Only read-only tools may be registered. "
            f"Attempted: {name!r}"
        )

    def prefetch(self, task: str) -> list[threading.Thread]:
        """Speculatively execute likely tools for task in background threads.

        Returns launched threads (caller may join if sync behavior needed).
        """
        if not self._enabled:
            return []

        intents = extract_intent_hints(task)
        if not intents:
            return []

        candidate_tools = tools_for_intents(intents)[: self._max_tools]
        threads: list[threading.Thread] = []
        for tool_name in candidate_tools:
            executor_fn = self._tool_executors.get(tool_name)
            if executor_fn is None:
                continue
            t = threading.Thread(
                target=self._run_tool,
                args=(tool_name, executor_fn, task),
                daemon=True,
                name=f"speculative-{tool_name}",
            )
            t.start()
            threads.append(t)
        return threads

    def _run_tool(self, tool_name: str, fn: Callable, task: str) -> None:
        try:
            result = fn(task=task)
            self.cache.put(tool_name, {"task": task}, result)
        except Exception as exc:
            self.cache.put(tool_name, {"task": task}, None, error=str(exc))

    def lookup(self, tool: str, args: dict) -> SpeculativeCacheEntry | None:
        """Check cache before executing a tool. Returns None on cache miss."""
        return self.cache.get(tool, args)

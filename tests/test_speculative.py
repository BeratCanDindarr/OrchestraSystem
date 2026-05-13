"""Tests for PASTE Speculative Tool Execution — SpeculativeExecutor."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from orchestra.engine.speculative import (
    SpeculativeCache,
    SpeculativeExecutor,
    extract_intent_hints,
    tools_for_intents,
    _READ_ONLY_TOOLS,
)


# ---------------------------------------------------------------------------
# extract_intent_hints
# ---------------------------------------------------------------------------

def test_extract_intent_find():
    intents = extract_intent_hints("find the authentication class")
    assert "find" in intents


def test_extract_intent_debug():
    intents = extract_intent_hints("fix the crash in payment module")
    assert "debug" in intents


def test_extract_intent_unity():
    intents = extract_intent_hints("check the unity scene for missing components")
    assert "unity" in intents


def test_extract_intent_empty_task():
    intents = extract_intent_hints("")
    assert intents == []


def test_extract_intent_multiple():
    intents = extract_intent_hints("find and explain the auth bug")
    assert "find" in intents
    assert "explain" in intents


# ---------------------------------------------------------------------------
# tools_for_intents
# ---------------------------------------------------------------------------

def test_tools_for_intents_deduplicates():
    # Both "find" and "search" map to find_symbol — should appear only once
    tools = tools_for_intents(["find", "search"])
    assert tools.count("find_symbol") == 1


def test_tools_for_intents_read_only_only():
    tools = tools_for_intents(["find", "debug", "unity", "git"])
    for tool in tools:
        assert tool in _READ_ONLY_TOOLS, f"{tool} is not in READ_ONLY_TOOLS"


def test_tools_for_intents_empty_intents():
    tools = tools_for_intents([])
    assert tools == []


# ---------------------------------------------------------------------------
# SpeculativeCache
# ---------------------------------------------------------------------------

def test_cache_put_and_get():
    cache = SpeculativeCache(run_id="test-run")
    cache.put("read_file", {"task": "read auth.py"}, "file content")
    entry = cache.get("read_file", {"task": "read auth.py"})
    assert entry is not None
    assert entry.result == "file content"
    assert entry.error is None


def test_cache_miss_returns_none():
    cache = SpeculativeCache(run_id="test-run")
    entry = cache.get("read_file", {"task": "nonexistent"})
    assert entry is None


def test_cache_hit_flag_set():
    cache = SpeculativeCache(run_id="test-run")
    cache.put("grep", {"task": "search imports"}, ["line1", "line2"])
    entry = cache.get("grep", {"task": "search imports"})
    assert entry.hit is True


def test_cache_stats():
    cache = SpeculativeCache(run_id="test-run")
    cache.put("read_file", {"task": "t1"}, "result1")
    cache.put("grep", {"task": "t2"}, "result2")
    cache.get("read_file", {"task": "t1"})  # one hit
    stats = cache.stats()
    assert stats["total"] == 2
    assert stats["hits"] == 1
    assert stats["miss"] == 1


def test_cache_thread_safe():
    cache = SpeculativeCache(run_id="test-run")
    errors = []

    def writer(i):
        try:
            cache.put(f"tool_{i}", {"i": i}, f"result_{i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert cache.stats()["total"] == 50


# ---------------------------------------------------------------------------
# SpeculativeExecutor.register_tool
# ---------------------------------------------------------------------------

def test_register_read_only_tool_allowed():
    executor = SpeculativeExecutor(run_id="run-001")
    fn = MagicMock(return_value="content")
    executor.register_tool("read_file", fn)
    assert "read_file" in executor._tool_executors


def test_register_write_tool_rejected():
    executor = SpeculativeExecutor(run_id="run-001")
    fn = MagicMock()
    executor.register_tool("write_file", fn)  # not in _READ_ONLY_TOOLS → silently ignored
    assert "write_file" not in executor._tool_executors


def test_register_tool_unsafe_raises():
    executor = SpeculativeExecutor(run_id="run-001")
    with pytest.raises(RuntimeError, match="forbidden"):
        executor.register_tool_unsafe("write_file", MagicMock())


# ---------------------------------------------------------------------------
# SpeculativeExecutor.prefetch
# ---------------------------------------------------------------------------

def test_prefetch_calls_registered_tools():
    executor = SpeculativeExecutor(run_id="run-001", max_tools=5, enabled=True)
    results = []
    lock = threading.Lock()

    def mock_find(task):
        with lock:
            results.append("find_file")
        return ["auth.py"]

    executor.register_tool("find_file", mock_find)

    threads = executor.prefetch("find the auth file")
    for t in threads:
        t.join(timeout=2.0)

    assert "find_file" in results


def test_prefetch_stores_result_in_cache():
    executor = SpeculativeExecutor(run_id="run-001", max_tools=5, enabled=True)

    def mock_read(task):
        return "file contents here"

    executor.register_tool("read_file", mock_read)

    threads = executor.prefetch("show me what is in the config file")
    for t in threads:
        t.join(timeout=2.0)

    entry = executor.cache.get("read_file", {"task": "show me what is in the config file"})
    assert entry is not None
    assert entry.result == "file contents here"


def test_prefetch_stores_error_on_exception():
    executor = SpeculativeExecutor(run_id="run-001", max_tools=5, enabled=True)

    def failing_tool(task):
        raise RuntimeError("tool exploded")

    executor.register_tool("read_file", failing_tool)

    threads = executor.prefetch("show me what is in config")
    for t in threads:
        t.join(timeout=2.0)

    entry = executor.cache.get("read_file", {"task": "show me what is in config"})
    assert entry is not None
    assert entry.error == "tool exploded"
    assert entry.result is None


def test_prefetch_disabled_runs_nothing():
    with patch("orchestra.config.section", return_value={"enabled": False, "max_prefetch_tools": 5}):
        executor = SpeculativeExecutor(run_id="run-001")

    fn = MagicMock(return_value="data")
    executor.register_tool("read_file", fn)
    threads = executor.prefetch("show me the file")
    assert threads == []
    fn.assert_not_called()


def test_prefetch_respects_max_tools_cap():
    executor = SpeculativeExecutor(run_id="run-001", max_tools=2)
    called = []
    lock = threading.Lock()

    for tool in ["read_file", "find_file", "get_symbols_overview"]:
        def make_fn(name):
            def fn(task):
                with lock:
                    called.append(name)
                return name
            return fn
        executor.register_tool(tool, make_fn(tool))

    threads = executor.prefetch("explain how auth works")
    for t in threads:
        t.join(timeout=2.0)

    assert len(called) <= 2


# ---------------------------------------------------------------------------
# SpeculativeExecutor.lookup
# ---------------------------------------------------------------------------

def test_lookup_hits_cache():
    executor = SpeculativeExecutor(run_id="run-001", max_tools=5, enabled=True)

    def mock_read(task):
        return "cached content"

    executor.register_tool("read_file", mock_read)
    threads = executor.prefetch("show me what is in config file")
    for t in threads:
        t.join(timeout=2.0)

    hit = executor.lookup("read_file", {"task": "show me what is in config file"})
    assert hit is not None
    assert hit.result == "cached content"


def test_lookup_returns_none_on_miss():
    executor = SpeculativeExecutor(run_id="run-001")
    result = executor.lookup("read_file", {"task": "never prefetched"})
    assert result is None

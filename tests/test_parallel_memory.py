"""Unit tests for orchestra.engine.parallel and orchestra.storage.memory."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from orchestra.engine.parallel import _make_table, _mark_run_cancelled, _mark_run_cancelled_async
from orchestra.models import OrchestraRun, AgentRun, AgentStatus, RunStatus
from orchestra.storage.memory import SemanticMemory

def test_make_table():
    run = OrchestraRun(mode="ask", task="test", run_id="r1")
    run.agents = [AgentRun(alias="a1", provider="p1", model="m1")]
    table = _make_table(run)
    assert table is not None
    # table.title is a Renderable, convert to string might need console.render_str or similar
    # but we can just check if it exists

def test_mark_run_cancelled():
    run = OrchestraRun(mode="ask", task="test", run_id="r1", status=RunStatus.RUNNING)
    run.agents = [AgentRun(alias="a1", provider="p1", model="m1", status=AgentStatus.STARTED)]
    lock = threading.Lock()
    
    with patch("orchestra.engine.artifacts.write_manifest"), \
         patch("orchestra.engine.artifacts.append_event"):
        _mark_run_cancelled(run, lock, "test reason")
        assert run.status == RunStatus.CANCELLED
        assert run.agents[0].status == AgentStatus.CANCELLED

@pytest.mark.asyncio
async def test_mark_run_cancelled_async():
    run = OrchestraRun(mode="ask", task="test", run_id="r1", status=RunStatus.RUNNING)
    run.agents = [AgentRun(alias="a1", provider="p1", model="m1", status=AgentStatus.STARTED)]
    lock = threading.Lock()
    
    with patch("orchestra.engine.artifacts.write_manifest"), \
         patch("orchestra.engine.artifacts.append_event"):
        await _mark_run_cancelled_async(run, lock, "test reason")
        assert run.status == RunStatus.CANCELLED

def test_memory_basic(tmp_path):
    db_path = tmp_path / "memory.db"
    mem = SemanticMemory(db_path=str(db_path))
    
    # Mock embedding
    with patch("orchestra.storage.memory._ollama.embed", return_value=[0.1]*128):
        mem.add("content", summary="summary", metadata={"a": 1})
        
        results = mem.search("content", limit=1)
        assert len(results) == 1
        assert results[0]["content"] == "content"
        assert results[0]["summary"] == "summary"

def test_memory_clear(tmp_path):
    db_path = tmp_path / "memory.db"
    mem = SemanticMemory(db_path=str(db_path))
    with patch("orchestra.storage.memory._ollama.embed", return_value=[0.1]*128):
        mem.add("c")
        
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("DELETE FROM embeddings")
            conn.commit()
        
        # SemanticMemory doesn't have clear(), so we manually delete and check
        results = mem.search("c")
        assert len(results) == 0

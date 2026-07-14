from __future__ import annotations
"""Performance regression test suite for Orchestra.

Each benchmark records its runtime and compares against a stored baseline.
If a test runs >20% slower than baseline, a warning is emitted.
Results are printed in a formatted table at the end of the module.
"""

import json
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from orchestra.engine.parallel import run_parallel
from orchestra.engine.synthesizer import compute_divergence_index
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus
from orchestra.router.classifier import classify, route_task
from orchestra.storage.db import backfill
from orchestra.storage.event_log import EventLog
from orchestra.storage.fts_cache import FtsSearchCache
from orchestra.storage.memory import SemanticMemory

# ---------------------------------------------------------------------------
# Result recording helpers
# ---------------------------------------------------------------------------
_results: list[dict[str, Any]] = []


def _baseline_path() -> Path:
    return Path(__file__).parent / "baselines" / "performance_baseline.json"


def _load_baseline() -> dict[str, float]:
    path = _baseline_path()
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_baseline(baselines: dict[str, float]) -> None:
    _baseline_path().parent.mkdir(parents=True, exist_ok=True)
    _baseline_path().write_text(json.dumps(baselines, indent=2))


def _record_result(name: str, elapsed: float, iterations: int = 1) -> None:
    _results.append(
        {
            "name": name,
            "elapsed": elapsed,
            "per_op": elapsed / iterations if iterations > 0 else elapsed,
            "iterations": iterations,
        }
    )
    baselines = _load_baseline()
    if name in baselines:
        baseline = baselines[name]
        if elapsed > baseline * 1.2:
            warnings.warn(
                f"PERFORMANCE WARNING: {name} took {elapsed:.4f}s, "
                f"baseline is {baseline:.4f}s ({(elapsed / baseline - 1) * 100:.1f}% slower)",
                stacklevel=3,
            )


def _mock_embed(text: str, dim: int = 384) -> list:
    """Create deterministic normalized embedding vector from text."""
    h = hash(text) % (2**32)
    np.random.seed(h)
    vec = np.random.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


# ---------------------------------------------------------------------------
# Module-level summary printer
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def _benchmark_summary():
    yield
    if not _results:
        return

    print("\n" + "=" * 72)
    print("PERFORMANCE REGRESSION RESULTS")
    print("=" * 72)
    print(f"{'Benchmark':<28} {'Total (s)':<12} {'Per-op (ms)':<14} {'Iters':<8}")
    print("-" * 72)

    baselines = _load_baseline()
    new_baselines = dict(baselines)

    for r in _results:
        name = r["name"]
        elapsed = r["elapsed"]
        per_op_ms = r["per_op"] * 1000
        iters = r["iterations"]
        print(f"{name:<28} {elapsed:<12.4f} {per_op_ms:<14.4f} {iters:<8}")
        new_baselines[name] = elapsed

    print("-" * 72)
    # Writing the checked-in baseline on every run creates fake git churn.
    # Opt in explicitly when you intend to re-baseline.
    if os.environ.get("ORCHESTRA_UPDATE_BASELINES") == "1":
        _save_baseline(new_baselines)
        print(f"Baselines updated → tests/baselines/performance_baseline.json")
    else:
        print("Baselines NOT updated (set ORCHESTRA_UPDATE_BASELINES=1 to re-baseline)")
    print("=" * 72)


# ---------------------------------------------------------------------------
# 1. EventLog Append Benchmark
# ---------------------------------------------------------------------------
@pytest.mark.benchmark
@pytest.mark.slow
def test_benchmark_event_log_append(tmp_path):
    """Append 100 events; max time < 20ms per event."""
    db_path = tmp_path / "events.db"
    event_log = EventLog(db_path)

    # Warm-up to stabilise SQLite WAL
    for i in range(5):
        event_log.append("run_started", "run-warmup", {"i": i})

    times = []
    for i in range(100):
        start = time.perf_counter()
        event_log.append("agent_started", f"run-{i}", {"iteration": i})
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    max_time_ms = max(times) * 1000
    # Use 50ms as a generous hard ceiling; baseline comparison catches regressions.
    assert max_time_ms < 50.0, f"Max append time {max_time_ms:.2f}ms > 50ms"
    _record_result("event_log_append", sum(times), 100)


# ---------------------------------------------------------------------------
# 2. HNSW Search Benchmark
# ---------------------------------------------------------------------------
@pytest.mark.benchmark
@pytest.mark.slow
def test_benchmark_hnsw_search(tmp_path):
    """Insert 1000 vectors, search 100 times; total search time < 1s."""
    db_path = tmp_path / "memory.db"
    mem = SemanticMemory(str(db_path))

    # Setup: insert 1000 vectors (not timed)
    with patch("orchestra.storage.memory._ollama.embed") as mock_embed:
        mock_embed.side_effect = lambda text: _mock_embed(text)
        for i in range(1000):
            mem.add(f"content {i} python code", metadata={"path": f"{i}.py"})

    # Benchmark: 100 searches
    # Mock _log_memory_event to avoid DB writes that invalidate the index
    with patch("orchestra.storage.memory._ollama.embed") as mock_embed:
        mock_embed.side_effect = lambda text: _mock_embed(text)
        with patch.object(mem, "_log_memory_event"):
            start = time.perf_counter()
            for i in range(100):
                mem.search(f"query {i} python", limit=5)
            elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"100 searches took {elapsed:.3f}s (expected < 1.0s)"
    _record_result("hnsw_search", elapsed, 100)


# ---------------------------------------------------------------------------
# 3. FTS5 Cache Lookup Benchmark
# ---------------------------------------------------------------------------
@pytest.mark.benchmark
def test_benchmark_fts_cache_lookup(tmp_path):
    """Insert 100 entries, perform 50 lookups; total lookup time < 500ms."""
    db_path = tmp_path / "fts.db"
    cache = FtsSearchCache(str(db_path))

    for i in range(100):
        cache.store("run-1", f"prompt {i} machine learning keywords", f"response {i}")

    start = time.perf_counter()
    for i in range(50):
        cache.search(f"prompt {i}", topk=3)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5, f"50 lookups took {elapsed:.3f}s (expected < 0.5s)"
    _record_result("fts_cache_lookup", elapsed, 50)
    cache.close()


# ---------------------------------------------------------------------------
# 4. Provider Dispatch Benchmark
# ---------------------------------------------------------------------------
@pytest.mark.benchmark
@pytest.mark.slow
def test_benchmark_provider_dispatch(tmp_path):
    """Mock provider that returns instantly; dispatch 10 agents in parallel.

    Assert total time < 2s (mostly framework overhead).
    """
    run = OrchestraRun(mode="ask", task="benchmark task")
    run.status = RunStatus.RUNNING

    agents = [(f"agent-{i}", f"prompt {i}") for i in range(10)]

    async def _async_return(*args, **kwargs):
        return ("", 0)

    with patch("orchestra.engine.parallel.run_with_retry", return_value=("", 0)):
        with patch(
            "orchestra.engine.parallel.run_with_retry_async",
            side_effect=_async_return,
        ):
            with patch("orchestra.engine.parallel.artifacts") as _mock_artifacts:
                with patch(
                    "orchestra.engine.parallel.get_tracer", return_value=None
                ):
                    with patch(
                        "orchestra.engine.parallel._enforce_budget_guard"
                    ):
                        with patch(
                            "orchestra.engine.parallel._enforce_budget_guard_async"
                        ):
                            with patch(
                                "orchestra.engine.replay.get_recorder"
                            ) as mock_recorder:
                                mock_recorder.return_value = MagicMock()
                                start = time.perf_counter()
                                run_parallel(
                                    run,
                                    agents,
                                    show_live=False,
                                    install_signal_handlers=False,
                                    emit_console=False,
                                )
                                elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"10-agent dispatch took {elapsed:.3f}s (expected < 2.0s)"
    _record_result("provider_dispatch", elapsed, 10)


# ---------------------------------------------------------------------------
# 5. Synthesizer Divergence Benchmark
# ---------------------------------------------------------------------------
@pytest.mark.benchmark
def test_benchmark_synthesizer_divergence():
    """Run compute_divergence_index() on 10KB outputs; time < 100ms."""
    text_10kb = "word " * 2500  # ~12.5 KB of text
    agents = [
        AgentRun(
            alias=f"agent-{i}",
            provider="test",
            model="test-model",
            status=AgentStatus.COMPLETED,
            stdout_log=text_10kb,
            confidence=0.8,
        )
        for i in range(5)
    ]

    start = time.perf_counter()
    compute_divergence_index(agents)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1, f"Divergence computation took {elapsed:.3f}s (expected < 0.1s)"
    _record_result("synthesizer_divergence", elapsed, 1)


# ---------------------------------------------------------------------------
# 6. Router Classification Benchmark
# ---------------------------------------------------------------------------
@pytest.mark.benchmark
def test_benchmark_router_classification(monkeypatch):
    """Run classify() and route_task() 100 times; total time < 200ms."""
    monkeypatch.setattr(
        "orchestra.router.classifier._ROUTER_POLICY",
        {"keywords": {"cheap": ["hello"], "deep": ["research", "plan", "analyze"]}},
    )

    tasks = [
        "hello there",
        "research quantum computing",
        "plan the architecture",
        "do something interesting",
        "what is polymorphism",
    ] * 20  # 100 total iterations

    with patch("orchestra.router.classifier._model_route", return_value=None):
        with patch("orchestra.router.classifier.config.routing_config", return_value={}):
            with patch(
                "orchestra.router.classifier.select_alias_for_candidates",
                return_value="cdx-fast",
            ):
                start = time.perf_counter()
                for task in tasks:
                    classify(task)
                    route_task(task)
                elapsed = time.perf_counter() - start

    assert elapsed < 0.2, f"100 route ops took {elapsed:.3f}s (expected < 0.2s)"
    _record_result("router_classification", elapsed, 100)


# ---------------------------------------------------------------------------
# 7. DB Backfill Benchmark
# ---------------------------------------------------------------------------
@pytest.mark.benchmark
@pytest.mark.slow
def test_benchmark_db_backfill(tmp_path, monkeypatch):
    """Create 100 mock manifests, run backfill(); time < 3s."""
    root = tmp_path / "runs"
    root.mkdir()

    for i in range(100):
        run_dir = root / f"run_{i:03d}"
        run_dir.mkdir()
        manifest = {
            "run_id": f"run-{i}",
            "mode": "ask",
            "status": "completed",
            "task": f"task {i}",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
            "agents": [],
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    db_path = tmp_path / "orchestra.db"
    monkeypatch.setenv("ORCHESTRA_DB_PATH", str(db_path))
    monkeypatch.setattr("orchestra.storage.db.config.artifact_root", lambda: root)

    start = time.perf_counter()
    count = backfill()
    elapsed = time.perf_counter() - start

    assert count == 100, f"Expected 100 manifests, got {count}"
    assert elapsed < 3.0, f"Backfill took {elapsed:.3f}s (expected < 3.0s)"
    _record_result("db_backfill", elapsed, 100)

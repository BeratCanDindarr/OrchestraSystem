"""Core skill: benchmark MCP Fast Lane vs JSON-RPC overhead."""
from __future__ import annotations

import json
import statistics
import time
from typing import Any

from orchestra.engine.mcp_fastlane import FASTLANE
from orchestra.engine.mcp_registry import REGISTRY
from orchestra.engine.skills.base import SkillContext, SkillResult


def can_execute(context: SkillContext) -> bool:
    return True


def execute(context: SkillContext) -> SkillResult:
    iterations = int(context.args.get("iterations", 1000))
    tools = context.args.get("tools", "orchestra_get_version").split(",")
    results: dict[str, Any] = {}

    for tool_name in tools:
        tool_name = tool_name.strip()
        # REGISTRY has no .get() method; check availability via invoke (returns
        # None for unregistered tools) or is_fastlane for fastlane status.
        if REGISTRY.invoke(tool_name, {}) is None:
            results[tool_name] = {"error": "not registered"}
            continue

        fastlane_times: list[float] = []
        jsonrpc_times: list[float] = []

        # Warm-up
        for _ in range(10):
            FASTLANE.dispatch(tool_name, {})
            _jsonrpc_roundtrip(tool_name, {})

        # Benchmark Fast Lane
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            FASTLANE.dispatch(tool_name, {})
            t1 = time.perf_counter_ns()
            fastlane_times.append((t1 - t0) / 1_000_000)  # ms

        # Benchmark JSON-RPC serde
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            _jsonrpc_roundtrip(tool_name, {})
            t1 = time.perf_counter_ns()
            jsonrpc_times.append((t1 - t0) / 1_000_000)  # ms

        median_diff = statistics.median(jsonrpc_times) - statistics.median(fastlane_times)
        results[tool_name] = {
            "iterations": iterations,
            "fastlane_median_ms": round(statistics.median(fastlane_times), 4),
            "jsonrpc_median_ms": round(statistics.median(jsonrpc_times), 4),
            "saving_median_ms": round(median_diff, 4),
            "p95_saving_ms": round(
                sorted(jsonrpc_times)[int(iterations * 0.95)] -
                sorted(fastlane_times)[int(iterations * 0.95)], 4
            ),
            "passes_gate": median_diff > 5.0,
        }

    return SkillResult(
        success=all(r.get("passes_gate", False) for r in results.values() if "error" not in r),
        output=json.dumps(results, ensure_ascii=False, indent=2),
        metrics={"tools_tested": len(tools)},
    )


def _jsonrpc_roundtrip(name: str, arguments: dict) -> dict:
    """Simulate JSON-RPC serde cost: dict → JSON string → parse → dispatch."""
    payload = {"jsonrpc": "2.0", "method": name, "params": arguments}
    text = json.dumps(payload, ensure_ascii=False)
    parsed = json.loads(text)
    return REGISTRY.invoke(parsed["method"], parsed["params"]) or {}

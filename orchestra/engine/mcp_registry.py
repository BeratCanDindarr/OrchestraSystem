from __future__ import annotations
import inspect
import logging
import json
import sys
import traceback
from typing import Any, Callable

logger = logging.getLogger(__name__)

def _scrub_dict(payload: dict) -> dict:
    from orchestra.engine.pii_scrubber import PiiScrubber
    return PiiScrubber.scrub_dict(payload)

def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def _tool_success(payload: dict) -> dict:
    scrubbed = _scrub_dict(payload)
    return {
        "content": [{"type": "text", "text": _json_dumps(scrubbed)}],
        "structuredContent": scrubbed,
        "isError": False,
    }

def _tool_error(message: str) -> dict:
    return {
        "content": [{"type": "text", "text": str(message)}],
        "isError": True,
    }

def _structured_run_result(payload: dict) -> dict:
    if not isinstance(payload, dict): return {}
    run = payload.get("run")
    if not isinstance(run, dict): run = {}
    
    return {
        "run_id": str(run.get("run_id", "")),
        "status": str(run.get("status", "unknown")),
        "mode": str(run.get("mode", "unknown")),
        "task": str(run.get("task", "")),
        "summary": str(run.get("summary", ""))[:500],
        "agent_count": len(run.get("agents", []) or []),
    }

def _tool_success_run(payload: dict) -> dict:
    if not isinstance(payload, dict): return _tool_error("Tool returned non-dict payload")
    structured = dict(payload)
    structured["result"] = _structured_run_result(payload)
    return _tool_success(structured)

_RUN_TOOLS = {"orchestra_submit_run", "orchestra_run", "orchestra_continue", "orchestra_wait_for_run", "orchestra_execute_job", "orchestra_get_run"}

def _wrap_result(name: str, result: Any) -> dict:
    if isinstance(result, dict):
        if name in _RUN_TOOLS:
            return _tool_success_run(result)
        return _tool_success(result)
    return _tool_success({"result": result})

class McpRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[Callable, bool]] = {}

    def register(self, name: str, fn: Callable, *, fastlane: bool = True) -> None:
        self._tools[name] = (fn, fastlane)

    def invoke(self, name: str, arguments: dict) -> dict | None:
        entry = self._tools.get(name)
        if entry is None: return None
        fn, _ = entry
        try:
            # 🛡️ Hardened arguments check
            if not isinstance(arguments, dict): arguments = {}
            result = fn(arguments)
            return _wrap_result(name, result)
        except Exception as exc:
            err_msg = f"Error in {name}: {exc}"
            sys.stderr.write(f"FAIL {name}: {exc}\n{traceback.format_exc()}\n")
            return _tool_error(err_msg)

REGISTRY = McpRegistry()

def _auto_wrap(fn: Callable) -> Callable:
    import inspect
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    def wrapper(arguments: dict) -> Any:
        # 🛡️ Defensive arg extraction
        if not isinstance(arguments, dict): arguments = {}
        kwargs = {p: arguments[p] for p in params if p in arguments}
        return fn(**kwargs)
    return wrapper

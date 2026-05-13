"""Tool registry for speculative execution.

Provides bridge between MCP/CLI tools and SpeculativeExecutor.
Registers read-only tool executors that can run speculatively.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Any

from orchestra import config


def _make_find_symbol_executor() -> Callable:
    """Create executor for find_symbol tool (uses Serena LSP)."""
    def executor(task: str, **kwargs) -> Any:
        try:
            from orchestra.tools.find_symbol import find_symbol
            return find_symbol(task)
        except ImportError:
            return {"error": "find_symbol tool not available"}
        except Exception as e:
            return {"error": str(e)}
    return executor


def _make_read_file_executor() -> Callable:
    """Create executor for read_file tool (filesystem access)."""
    def executor(task: str, **kwargs) -> Any:
        try:
            from orchestra.tools.read_file import read_file
            return read_file(task)
        except ImportError:
            return {"error": "read_file tool not available"}
        except Exception as e:
            return {"error": str(e)}
    return executor


def _make_git_log_executor() -> Callable:
    """Create executor for git_log tool (read-only git access)."""
    def executor(task: str, **kwargs) -> Any:
        try:
            from orchestra.tools.git_tools import git_log
            return git_log(task)
        except ImportError:
            return {"error": "git_log tool not available"}
        except Exception as e:
            return {"error": str(e)}
    return executor


def _make_web_search_executor() -> Callable:
    """Create executor for web_search tool (documentation lookup)."""
    def executor(task: str, **kwargs) -> Any:
        try:
            from orchestra.tools.web_search import web_search
            return web_search(task)
        except ImportError:
            return {"error": "web_search tool not available"}
        except Exception as e:
            return {"error": str(e)}
    return executor


_TOOL_EXECUTORS: Dict[str, Callable] = {
    "find_symbol": _make_find_symbol_executor(),
    "read_file": _make_read_file_executor(),
    "git_log": _make_git_log_executor(),
    "web_search": _make_web_search_executor(),
}


def register_speculative_tools(executor) -> int:
    """Register all available tools with a SpeculativeExecutor.

    Args:
        executor: SpeculativeExecutor instance

    Returns:
        Number of tools successfully registered
    """
    registered = 0
    for tool_name, executor_fn in _TOOL_EXECUTORS.items():
        try:
            executor.register_tool(tool_name, executor_fn)
            registered += 1
        except Exception:
            pass
    return registered

"""Unit tests for orchestra.mcp_server."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from orchestra.mcp_server import _tool_success, _tool_error, _auto_wrap

def test_tool_helpers():
    succ = _tool_success({"a": 1})
    # _json_dumps uses indent=2
    assert "a" in succ["content"][0]["text"]
    assert "1" in succ["content"][0]["text"]
    
    err = _tool_error("msg")
    assert err["content"][0]["text"] == "msg"

def test_auto_wrap():
    def my_func(a: int):
        return {"res": a}
    
    wrapped = _auto_wrap(my_func)
    res = wrapped({"a": 5})
    assert res["res"] == 5

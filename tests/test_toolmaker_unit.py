import pytest
from unittest.mock import MagicMock, patch
from orchestra.engine.toolmaker import ToolMaker
from pathlib import Path

def test_toolmaker_propose(tmp_path):
    tm = ToolMaker(tools_dir=str(tmp_path))
    path = tm.propose_new_tool("test_tool", "print('hello')", "test tool desc")
    assert Path(path).exists()
    assert "test_tool" in path
    assert path.endswith(".py")

def test_toolmaker_syntax_check(tmp_path):
    tm = ToolMaker(tools_dir=str(tmp_path))
    path = tm.propose_new_tool("valid", "print('ok')", "desc")
    
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        ok, msg = tm.test_tool_syntax(path)
        assert ok is True
        
        mock_run.return_value = MagicMock(returncode=1, stderr="Syntax Error")
        ok, msg = tm.test_tool_syntax(path)
        assert ok is False
        assert "Syntax Error" in msg

def test_toolmaker_promote(tmp_path):
    tm = ToolMaker(tools_dir=str(tmp_path))
    path = tm.propose_new_tool("test", "code", "desc")
    
    ok, msg = tm.promote_to_live(path)
    assert ok is True
    assert (tmp_path / "test.py").exists()

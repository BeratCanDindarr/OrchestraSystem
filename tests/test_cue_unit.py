import pytest
from unittest.mock import MagicMock, patch
from orchestra.engine.cue import CueEngine

def test_cue_engine_trigger(tmp_path):
    cue_file = tmp_path / "cue.yaml"
    cue_file.write_text("rules:\n  test_rule:\n    on: file.changed\n    do: run_agent\n    with:\n      task: 'hello'", encoding="utf-8")
    
    engine = CueEngine(cue_file=str(cue_file))
    assert len(engine.rules) == 1
    
    with patch("orchestra.engine.cue.run_ask") as mock_run:
        engine.trigger("file.changed", {"file": "test.py"})
        mock_run.assert_called_once()

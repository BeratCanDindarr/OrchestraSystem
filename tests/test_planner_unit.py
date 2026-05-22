import pytest
from unittest.mock import MagicMock, patch
from orchestra.engine.planner import create_graph_plan
from orchestra.models import AgentStatus

def test_planner_basic():
    mock_run = MagicMock()
    mock_run.task = "task"
    mock_run.run_id = "r1"
    
    mock_agent_run = MagicMock()
    mock_agent_run.status = AgentStatus.COMPLETED
    mock_agent_run.stdout_log = '```json\n{"nodes": [{"id": "n1", "task": "t1"}], "edges": []}\n```'
    
    with patch("orchestra.engine.planner.run_parallel", return_value=[mock_agent_run]), \
         patch("orchestra.engine.planner.build_graph_from_plan") as mock_build:
        create_graph_plan(mock_run)
        mock_build.assert_called_once()

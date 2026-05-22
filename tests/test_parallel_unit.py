import pytest
import threading
from unittest.mock import MagicMock, patch
from orchestra.engine.parallel import _run_one_agent, _mark_run_cancelled
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus

def test_mark_run_cancelled():
    run = OrchestraRun(run_id="run1", mode="ask", task="test", created_at="2026-05-21T10:00:00+00:00")
    run.agents = [AgentRun(alias="a1", provider="p1", model="m1", status=AgentStatus.STARTED)]
    lock = threading.Lock()
    with patch("orchestra.engine.artifacts.write_manifest"), \
         patch("orchestra.engine.artifacts.append_event"):
        _mark_run_cancelled(run, lock, "test reason")
        assert run.status == RunStatus.CANCELLED
        assert run.agents[0].status == AgentStatus.CANCELLED

@patch("orchestra.engine.parallel._now", return_value="2026-05-21T10:00:00+00:00")
def test_run_one_agent_success(mock_now):
    run = OrchestraRun(run_id="run1", mode="ask", task="test", created_at="2026-05-21T10:00:00+00:00")
    lock = threading.Lock()
    cancel_event = threading.Event()
    
    mock_provider = MagicMock()
    mock_provider.run.return_value = ("Success", 0)
    mock_provider.model_label.return_value = "p1/m1"
    mock_provider.last_usage.return_value = {"completion_tokens": 10}
    
    with patch("orchestra.engine.parallel.resolve", return_value=(mock_provider, "model1")), \
         patch("orchestra.engine.parallel.run_with_retry", return_value=("Success", 0)), \
         patch("orchestra.engine.artifacts.append_event"), \
         patch("orchestra.engine.artifacts.write_agent_log"), \
         patch("orchestra.engine.artifacts.write_manifest"):
        _run_one_agent(run, "cdx-fast", "the actual prompt text", lock, MagicMock(), cancel_event)
        
        assert len(run.agents) == 1
        assert run.agents[0].status == AgentStatus.COMPLETED
        assert run.agents[0].stdout_log == "Success"

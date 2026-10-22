import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

from orchestra.engine.parallel import _run_one_agent_async
from orchestra.models import AgentStatus, OrchestraRun


@patch("orchestra.engine.parallel._now", return_value="2026-05-21T10:00:00+00:00")
def test_run_one_agent_success(mock_now):
    run = OrchestraRun(run_id="run1", mode="ask", task="test", created_at="2026-05-21T10:00:00+00:00")

    mock_provider = MagicMock()
    mock_provider.model_label.return_value = "p1/m1"
    mock_provider.last_usage.return_value = {"completion_tokens": 10}

    with patch("orchestra.engine.parallel.resolve", return_value=(mock_provider, "model1")), \
         patch("orchestra.engine.parallel.run_with_retry_async", new=AsyncMock(return_value=("Success", 0))), \
         patch("orchestra.engine.artifacts.append_event"), \
         patch("orchestra.engine.artifacts.write_agent_log"), \
         patch("orchestra.engine.artifacts.write_manifest"):
        asyncio.run(_run_one_agent_async(run, "cdx-fast", "the actual prompt text", threading.Lock(),
                                         MagicMock(), asyncio.Event()))

    assert len(run.agents) == 1
    assert run.agents[0].status == AgentStatus.COMPLETED
    assert run.agents[0].stdout_log == "Success"

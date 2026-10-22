"""A finished run is not written to the memory store: memory holds the notes someone added."""
from unittest.mock import patch

from orchestra.engine.runners import _finalize
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus


def test_a_finished_run_computes_no_embedding():
    # Every finished run used to be embedded and stored (333 embeddings, 41 rows); the one
    # reader, the semantic alias router, weighed the agents' self-stated confidence.
    run = OrchestraRun(mode="ask", task="find SmileEventController")
    run.agents = [AgentRun(alias="test-agent", provider="test", model="test-model",
                           status=AgentStatus.COMPLETED, stdout_log="answer")]

    with patch("orchestra.engine.runners.common.get_memory") as memory:
        _finalize(run)

    assert run.status == RunStatus.COMPLETED
    memory.return_value.add.assert_not_called()
    memory.return_value.sweep.assert_called_once()   # expired notes and old log entries still go

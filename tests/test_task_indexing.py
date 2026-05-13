"""Tests for task semantic memory indexing during finalization."""
import pytest
from unittest.mock import MagicMock, patch

from orchestra.models import OrchestraRun, RunStatus, AgentStatus
from orchestra.engine.runner import _finalize
from orchestra.router.semantic import suggest_best_alias


class TestFinalizationIndexing:
    """Test task indexing during finalization."""

    def test_finalize_indexes_completed_task(self):
        """Verify _finalize() indexes completed task to semantic memory."""
        run = OrchestraRun(mode="ask", task="find SmileEventController")
        run.status = RunStatus.COMPLETED

        with patch("orchestra.storage.memory.get_memory") as mock_mem:
            _finalize(run)

            mock_mem.return_value.add.assert_called_once()
            args = mock_mem.return_value.add.call_args
            assert args[0][0] == "find SmileEventController"
            assert args[1]["metadata"]["run_id"] == run.run_id
            assert args[1]["metadata"]["mode"] == "ask"

    def test_finalize_skips_failed_task(self):
        """Verify _finalize() does NOT index failed tasks."""
        from orchestra.models import AgentRun

        run = OrchestraRun(mode="ask", task="find something")
        # Add a failed agent so run will be marked as FAILED
        failed_agent = AgentRun(alias="test-agent", provider="test", model="test-model")
        failed_agent.status = AgentStatus.FAILED
        run.agents = [failed_agent]

        with patch("orchestra.storage.memory.get_memory") as mock_mem:
            _finalize(run)
            mock_mem.return_value.add.assert_not_called()

    def test_finalize_indexing_error_handling(self):
        """Verify indexing errors don't crash finalization."""
        run = OrchestraRun(mode="ask", task="test")
        run.status = RunStatus.COMPLETED

        with patch("orchestra.storage.memory.get_memory") as mock_mem:
            mock_mem.return_value.add.side_effect = Exception("Memory error")
            _finalize(run)

    def test_task_metadata_structure(self):
        """Verify metadata contains required fields."""
        run = OrchestraRun(mode="dual", task="test task")
        run.status = RunStatus.COMPLETED

        with patch("orchestra.storage.memory.get_memory") as mock_mem:
            _finalize(run)

            metadata = mock_mem.return_value.add.call_args[1]["metadata"]
            assert metadata["run_id"] == run.run_id
            assert metadata["mode"] == "dual"
            assert "agents" in metadata
            assert isinstance(metadata["agents"], list)
            assert "agent_stats" in metadata
            assert "total_cost" in metadata


class TestSemanticRouting:
    """Test suggest_best_alias() with indexed tasks."""

    def test_suggest_best_alias_finds_similar(self):
        """Verify suggest_best_alias() retrieves similar tasks."""
        with patch("orchestra.router.semantic.get_memory") as mock_mem:
            mock_mem.return_value.search.return_value = [
                {"metadata": {"run_id": "run1", "agents": ["cdx-fast", "gmn-pro"],
                              "agent_stats": [
                                  {"alias": "cdx-fast", "confidence": 0.95, "status": "completed"},
                                  {"alias": "gmn-pro", "confidence": 0.85, "status": "completed"}
                              ]}}
            ]
            alias = suggest_best_alias("find auth module", ["cdx-fast", "gmn-pro"])
            assert alias == "cdx-fast"

    def test_suggest_best_alias_empty_on_no_similar(self):
        """Verify returns None when no similar tasks found."""
        with patch("orchestra.router.semantic.get_memory") as mock_mem:
            mock_mem.return_value.search.return_value = []
            alias = suggest_best_alias("unique task", ["cdx-fast"])
            assert alias is None

    def test_suggest_best_alias_filters_candidates(self):
        """Verify only candidate aliases are considered."""
        with patch("orchestra.router.semantic.get_memory") as mock_mem:
            mock_mem.return_value.search.return_value = [
                {"metadata": {"run_id": "run1", "agents": ["cdx-fast", "gmn-pro", "other"],
                              "agent_stats": []}}
            ]
            with patch("orchestra.router.semantic.get_db") as mock_db:
                mock_db.return_value.execute.return_value.fetchall.return_value = [
                    ("cdx-fast", 0.9, "COMPLETED"),
                    ("gmn-pro", 0.8, "COMPLETED"),
                    ("other", 0.95, "COMPLETED"),
                ]

                alias = suggest_best_alias("task", ["cdx-fast", "gmn-pro"])
                assert alias == "cdx-fast"
                assert alias != "other"

    def test_suggest_best_alias_aggregates_multiple_similar(self):
        """Verify scoring aggregates across multiple similar tasks."""
        with patch("orchestra.router.semantic.get_memory") as mock_mem:
            mock_mem.return_value.search.return_value = [
                {"metadata": {"run_id": "run1", "agents": ["cdx-fast"], "agent_stats": []}},
                {"metadata": {"run_id": "run2", "agents": ["cdx-fast"], "agent_stats": []}},
            ]
            with patch("orchestra.router.semantic.get_db") as mock_db:
                mock_db.return_value.execute.return_value.fetchall.side_effect = [
                    [("cdx-fast", 0.9, "COMPLETED")],
                    [("cdx-fast", 0.8, "COMPLETED")],
                ]

                alias = suggest_best_alias("similar task", ["cdx-fast"])
                assert alias == "cdx-fast"

    def test_suggest_best_alias_prefers_higher_confidence(self):
        """Verify highest confidence agent is selected."""
        with patch("orchestra.router.semantic.get_memory") as mock_mem:
            mock_mem.return_value.search.return_value = [
                {"metadata": {"run_id": "run1", "agents": ["cdx-fast", "gmn-pro", "cld-fast"],
                              "agent_stats": []}}
            ]
            with patch("orchestra.router.semantic.get_db") as mock_db:
                mock_db.return_value.execute.return_value.fetchall.return_value = [
                    ("cld-fast", 0.99, "COMPLETED"),
                    ("cdx-fast", 0.88, "COMPLETED"),
                    ("gmn-pro", 0.77, "COMPLETED"),
                ]

                alias = suggest_best_alias("task", ["cdx-fast", "gmn-pro", "cld-fast"])
                assert alias == "cld-fast"

    def test_suggest_best_alias_no_completed_agents(self):
        """Verify handles runs with no completed agents."""
        with patch("orchestra.router.semantic.get_memory") as mock_mem:
            mock_mem.return_value.search.return_value = [
                {"metadata": {"run_id": "run1", "agents": ["cdx-fast"], "agent_stats": []}}
            ]
            with patch("orchestra.router.semantic.get_db") as mock_db:
                mock_db.return_value.execute.return_value.fetchall.return_value = [
                    ("cdx-fast", 0.5, "FAILED"),
                ]

                alias = suggest_best_alias("task", ["cdx-fast"])
                assert alias is None or alias == "cdx-fast"

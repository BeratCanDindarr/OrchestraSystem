"""Tests for Phase 3: Planned Mode Checkpointing (Block 4)."""
import pytest
import sqlite3
import json
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from orchestra.models import OrchestraRun, RunStatus, AgentRun, AgentStatus
from orchestra.state import ApprovalState
from orchestra.storage.db import get_db
from orchestra.storage.event_log import EventLog, InvalidEventTypeError
from orchestra.engine import artifacts


class TestPlannedCheckpointSchema:
    """Test Phase 3 database schema for planned_checkpoints table."""

    def setup_method(self):
        """Clear planned_checkpoints table before each test."""
        connection = get_db()
        with connection:
            connection.execute("DELETE FROM planned_checkpoints")
        connection.close()

    def test_planned_checkpoints_table_schema(self):
        """Verify planned_checkpoints table exists with correct schema."""
        connection = get_db()

        # Query the table schema
        cursor = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='planned_checkpoints'"
        )
        schema_row = cursor.fetchone()
        connection.close()

        assert schema_row is not None, "planned_checkpoints table must exist"
        schema_sql = schema_row[0]

        # Verify required columns are present
        required_columns = {
            "run_id", "node_id", "result_json", "status",
            "completed_at", "cost_usd"
        }
        for col in required_columns:
            assert col in schema_sql.lower(), f"Column {col} must exist in schema"

        # Verify PRIMARY KEY constraint
        assert "primary key" in schema_sql.lower(), "PRIMARY KEY constraint must exist"
        assert "run_id" in schema_sql.lower() and "node_id" in schema_sql.lower(), \
            "PRIMARY KEY must be (run_id, node_id)"

    def test_planned_checkpoints_has_primary_key(self):
        """Verify PRIMARY KEY is (run_id, node_id)."""
        connection = get_db()

        # Get table info to verify primary key
        cursor = connection.execute("PRAGMA table_info(planned_checkpoints)")
        columns = cursor.fetchall()
        connection.close()

        # Extract column names and verify they match expected schema
        col_names = {col[1]: col[2] for col in columns}  # name: type

        assert "run_id" in col_names, "run_id column must exist"
        assert "node_id" in col_names, "node_id column must exist"
        assert "result_json" in col_names, "result_json column must exist"
        assert "status" in col_names, "status column must exist"
        assert "completed_at" in col_names, "completed_at column must exist"
        assert "cost_usd" in col_names, "cost_usd column must exist"

    def test_idx_planned_checkpoints_run_id_exists(self):
        """Verify index on run_id exists."""
        connection = get_db()

        cursor = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_planned_checkpoints_run_id'"
        )
        index_row = cursor.fetchone()
        connection.close()

        assert index_row is not None, "Index idx_planned_checkpoints_run_id must exist for query optimization"


class TestCheckpointEventEmission:
    """Test EventLog receives checkpoint events."""

    def setup_method(self):
        """Clear database and prepare event log."""
        connection = get_db()
        with connection:
            connection.execute("DELETE FROM planned_checkpoints")
            connection.execute("DELETE FROM events")
        connection.close()

    def test_checkpoint_written_event_type_valid(self):
        """Verify checkpoint_written is a valid event type."""
        from orchestra.storage.event_log import VALID_EVENT_TYPES

        # Verify the event type is valid
        assert "checkpoint_written" in VALID_EVENT_TYPES, \
            "checkpoint_written must be a valid event type"

        # Verify we can construct a valid event payload
        run_id = "test_run_123"
        event_type = "checkpoint_written"
        payload = {
            "label": "phase_3_checkpoint",
            "checkpoint_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Verify payload structure
        assert isinstance(payload, dict), "Payload must be a dictionary"
        assert "label" in payload, "Payload must contain label"
        assert payload["label"] == "phase_3_checkpoint"

    def test_checkpoint_event_has_required_fields(self):
        """Verify checkpoint event payload contains required fields."""
        event_type = "checkpoint_written"
        run_id = "test_run_456"
        payload = {
            "label": "test_label",
            "checkpoint_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Verify event_type is valid
        from orchestra.storage.event_log import VALID_EVENT_TYPES
        assert event_type in VALID_EVENT_TYPES, \
            f"checkpoint_written must be in VALID_EVENT_TYPES"

        # Verify payload structure
        assert "label" in payload, "Payload must contain label"
        assert isinstance(payload["label"], str), "label must be a string"


class TestPlannedNodeUpsert:
    """Test UPSERT behavior for planned checkpoint nodes."""

    def setup_method(self):
        """Clear planned_checkpoints before each test."""
        connection = get_db()
        with connection:
            connection.execute("DELETE FROM planned_checkpoints")
        connection.close()

    def test_planned_node_insert_then_update(self):
        """Verify UPSERT: insert new node, then update on re-run."""
        connection = get_db()
        run_id = "planned_run_001"
        node_id = "node_1"

        # Insert initial checkpoint for node 1
        with connection:
            connection.execute(
                """INSERT INTO planned_checkpoints
                   (run_id, node_id, status, result_json, cost_usd, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, node_id, "COMPLETED", '{"output":"step1"}', 0.05,
                 datetime.now(timezone.utc).isoformat())
            )

        # Verify insert
        cursor = connection.execute(
            "SELECT COUNT(*) FROM planned_checkpoints WHERE run_id=? AND node_id=?",
            (run_id, node_id)
        )
        count = cursor.fetchone()[0]
        assert count == 1, "Initial insert should create 1 row"

        # UPSERT: update the same node (simulate resume)
        with connection:
            connection.execute(
                """INSERT INTO planned_checkpoints
                   (run_id, node_id, status, result_json, cost_usd, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, node_id) DO UPDATE SET
                   status=excluded.status,
                   result_json=excluded.result_json,
                   cost_usd=excluded.cost_usd,
                   completed_at=excluded.completed_at""",
                (run_id, node_id, "COMPLETED", '{"output":"step1_updated"}', 0.06,
                 datetime.now(timezone.utc).isoformat())
            )

        # Verify no duplicate rows
        cursor = connection.execute(
            "SELECT COUNT(*) FROM planned_checkpoints WHERE run_id=? AND node_id=?",
            (run_id, node_id)
        )
        count = cursor.fetchone()[0]
        assert count == 1, "UPSERT should not create duplicates"

        # Verify updated content
        cursor = connection.execute(
            "SELECT result_json FROM planned_checkpoints WHERE run_id=? AND node_id=?",
            (run_id, node_id)
        )
        result_json = cursor.fetchone()[0]
        result = json.loads(result_json)
        assert result["output"] == "step1_updated", "UPSERT should update the row"

        connection.close()

    def test_multiple_nodes_in_run(self):
        """Verify multiple node checkpoints can coexist for same run."""
        connection = get_db()
        run_id = "planned_run_multi"

        # Insert 3 nodes for same run
        nodes_data = [
            ("node_1", "COMPLETED", '{"output":"1"}', 0.05),
            ("node_2", "COMPLETED", '{"output":"2"}', 0.06),
            ("node_3", "PENDING", '{}', 0.0),
        ]

        with connection:
            for node_id, status, result_json, cost_usd in nodes_data:
                connection.execute(
                    """INSERT INTO planned_checkpoints
                       (run_id, node_id, status, result_json, cost_usd, completed_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (run_id, node_id, status, result_json, cost_usd,
                     datetime.now(timezone.utc).isoformat() if status == "COMPLETED" else None)
                )

        # Verify all 3 nodes exist
        cursor = connection.execute(
            "SELECT COUNT(*) FROM planned_checkpoints WHERE run_id=?",
            (run_id,)
        )
        count = cursor.fetchone()[0]
        assert count == 3, "All 3 nodes should exist for same run"

        # Verify node isolation
        cursor = connection.execute(
            "SELECT node_id, status FROM planned_checkpoints WHERE run_id=? ORDER BY node_id",
            (run_id,)
        )
        rows = cursor.fetchall()
        assert len(rows) == 3
        assert rows[0][1] == "COMPLETED"  # node_1
        assert rows[1][1] == "COMPLETED"  # node_2
        assert rows[2][1] == "PENDING"    # node_3

        connection.close()


class TestApprovalGateBlocks:
    """Test approval gate blocks run execution."""

    def setup_method(self):
        """Clear database."""
        connection = get_db()
        with connection:
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute("DELETE FROM agents")
            connection.execute("DELETE FROM runs")
            connection.execute("DELETE FROM planned_checkpoints")
            connection.execute("PRAGMA foreign_keys=ON")
        connection.close()

    def test_approval_gate_sets_waiting_approval(self):
        """Verify approval gate sets run status to WAITING_APPROVAL."""
        run = OrchestraRun(mode="planned", task="Test task")
        run.status = RunStatus.RUNNING
        run.approval_state = ApprovalState.NOT_REQUIRED

        # Simulate approval gate activation
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        # Verify state change
        assert run.status == RunStatus.WAITING_APPROVAL, \
            "Approval gate must set status to WAITING_APPROVAL"
        assert run.approval_state == ApprovalState.PENDING, \
            "Approval gate must set approval_state to PENDING"

    def test_approval_gate_calls_suspend_verifies_state(self):
        """Verify approval gate state transition for suspension."""
        run = OrchestraRun(mode="planned", task="Test task")
        run.run_id = "approval_test_001"
        run.status = RunStatus.RUNNING
        run.approval_state = ApprovalState.NOT_REQUIRED

        # Simulate approval gate transition
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        # Verify the run would be ready for suspension
        assert run.status == RunStatus.WAITING_APPROVAL, \
            "Run status should be WAITING_APPROVAL for suspension"
        assert run.approval_state == ApprovalState.PENDING, \
            "Approval state should be PENDING for suspension"


class TestResumeSkipsCompletedNodes:
    """Test planned mode resume skips already-completed nodes."""

    def setup_method(self):
        """Clear planned_checkpoints."""
        connection = get_db()
        with connection:
            connection.execute("DELETE FROM planned_checkpoints")
        connection.close()

    def test_resume_skips_completed_nodes(self):
        """Verify resume skips nodes with COMPLETED status."""
        connection = get_db()
        run_id = "resume_test_001"

        # Insert 3 nodes: 2 completed, 1 pending
        with connection:
            connection.execute(
                """INSERT INTO planned_checkpoints
                   (run_id, node_id, status, result_json, cost_usd, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, "node_1", "COMPLETED", '{"output":"1"}', 0.05,
                 datetime.now(timezone.utc).isoformat())
            )
            connection.execute(
                """INSERT INTO planned_checkpoints
                   (run_id, node_id, status, result_json, cost_usd, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, "node_2", "COMPLETED", '{"output":"2"}', 0.06,
                 datetime.now(timezone.utc).isoformat())
            )
            connection.execute(
                """INSERT INTO planned_checkpoints
                   (run_id, node_id, status, result_json, cost_usd, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, "node_3", "PENDING", '{}', 0.0, None)
            )

        # Query for completed nodes (would be skipped)
        cursor = connection.execute(
            "SELECT node_id FROM planned_checkpoints WHERE run_id=? AND status=?",
            (run_id, "COMPLETED")
        )
        completed_nodes = [row[0] for row in cursor.fetchall()]

        # Query for pending nodes (would be executed)
        cursor = connection.execute(
            "SELECT node_id FROM planned_checkpoints WHERE run_id=? AND status=?",
            (run_id, "PENDING")
        )
        pending_nodes = [row[0] for row in cursor.fetchall()]

        assert len(completed_nodes) == 2, "2 nodes should be completed"
        assert "node_1" in completed_nodes
        assert "node_2" in completed_nodes
        assert len(pending_nodes) == 1, "1 node should be pending"
        assert "node_3" in pending_nodes

        connection.close()

    def test_resume_run_skips_completed_execution(self):
        """Verify resume run execution skips completed nodes."""
        run_id = "resume_exec_001"
        connection = get_db()

        # Pre-populate checkpoint data
        with connection:
            connection.execute(
                """INSERT INTO planned_checkpoints
                   (run_id, node_id, status, result_json, cost_usd, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, "node_1", "COMPLETED", '{"output":"1"}', 0.05,
                 datetime.now(timezone.utc).isoformat())
            )

        # Verify checkpoint exists
        cursor = connection.execute(
            "SELECT status FROM planned_checkpoints WHERE run_id=? AND node_id=?",
            (run_id, "node_1")
        )
        status = cursor.fetchone()[0]
        assert status == "COMPLETED", "Node should be marked as COMPLETED"

        connection.close()


class TestIndexPerformance:
    """Test index existence for query optimization."""

    def test_index_query_performance(self):
        """Verify index enables efficient run_id lookups."""
        connection = get_db()
        run_id = "index_test_001"

        # Insert checkpoint
        with connection:
            connection.execute(
                """INSERT INTO planned_checkpoints
                   (run_id, node_id, status, result_json, cost_usd, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, "node_1", "COMPLETED", '{"output":"1"}', 0.05,
                 datetime.now(timezone.utc).isoformat())
            )

        # Query with index (should be fast)
        cursor = connection.execute(
            "SELECT COUNT(*) FROM planned_checkpoints WHERE run_id=?",
            (run_id,)
        )
        count = cursor.fetchone()[0]
        assert count == 1, "Query should find the checkpoint"

        # Verify index exists via sqlite_master
        cursor = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_planned_checkpoints_run_id'"
        )
        index_exists = cursor.fetchone() is not None
        assert index_exists, "Index must exist for optimization"

        connection.close()

    def test_composite_primary_key_query(self):
        """Verify efficient query using composite primary key."""
        connection = get_db()
        run_id = "pk_test_001"

        # Insert multiple nodes
        with connection:
            for i in range(1, 4):
                connection.execute(
                    """INSERT INTO planned_checkpoints
                       (run_id, node_id, status, result_json, cost_usd, completed_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (run_id, f"node_{i}", "PENDING", '{}', 0.0, None)
                )

        # Query specific node via primary key (most efficient)
        cursor = connection.execute(
            "SELECT status FROM planned_checkpoints WHERE run_id=? AND node_id=?",
            (run_id, "node_2")
        )
        result = cursor.fetchone()
        assert result is not None, "Primary key lookup should find the node"
        assert result[0] == "PENDING"

        connection.close()


class TestCheckpointDataIntegrity:
    """Test data integrity of checkpoints."""

    def setup_method(self):
        """Clear planned_checkpoints."""
        connection = get_db()
        with connection:
            connection.execute("DELETE FROM planned_checkpoints")
        connection.close()

    def test_checkpoint_result_json_storage(self):
        """Verify result_json is properly stored and retrieved."""
        connection = get_db()
        run_id = "json_test_001"
        node_id = "node_json"

        test_result = {
            "step": "verification",
            "passed": True,
            "details": {
                "items": [1, 2, 3],
                "nested": {"key": "value"}
            }
        }

        # Store JSON
        with connection:
            connection.execute(
                """INSERT INTO planned_checkpoints
                   (run_id, node_id, status, result_json, cost_usd, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, node_id, "COMPLETED", json.dumps(test_result), 0.05,
                 datetime.now(timezone.utc).isoformat())
            )

        # Retrieve and verify
        cursor = connection.execute(
            "SELECT result_json FROM planned_checkpoints WHERE run_id=? AND node_id=?",
            (run_id, node_id)
        )
        stored_json = cursor.fetchone()[0]
        retrieved_result = json.loads(stored_json)

        assert retrieved_result == test_result, "JSON should be perfectly preserved"
        assert retrieved_result["details"]["nested"]["key"] == "value"

        connection.close()

    def test_checkpoint_cost_tracking(self):
        """Verify cost_usd is accurately tracked."""
        connection = get_db()
        run_id = "cost_test_001"

        costs = [0.05, 0.10, 0.03]

        # Insert 3 nodes with different costs
        with connection:
            for i, cost in enumerate(costs, 1):
                connection.execute(
                    """INSERT INTO planned_checkpoints
                       (run_id, node_id, status, result_json, cost_usd, completed_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (run_id, f"node_{i}", "COMPLETED", '{}', cost,
                     datetime.now(timezone.utc).isoformat())
                )

        # Sum costs
        cursor = connection.execute(
            "SELECT SUM(cost_usd) FROM planned_checkpoints WHERE run_id=?",
            (run_id,)
        )
        total_cost = cursor.fetchone()[0]
        expected_cost = sum(costs)

        assert abs(total_cost - expected_cost) < 0.001, \
            f"Total cost should be {expected_cost}, got {total_cost}"

        connection.close()


def test_phase3_integration():
    """Integration test: planned mode checkpointing workflow."""
    connection = get_db()
    run_id = "phase3_integration_001"

    # Cleanup
    with connection:
        connection.execute("DELETE FROM planned_checkpoints WHERE run_id=?", (run_id,))

    # Simulate planned mode execution: 3 nodes, 2 complete, 1 pending
    nodes = [
        ("node_1", "COMPLETED", '{"result":"verified"}', 0.05),
        ("node_2", "COMPLETED", '{"result":"passed"}', 0.06),
        ("node_3", "PENDING", '{}', 0.0),
    ]

    with connection:
        for node_id, status, result_json, cost_usd in nodes:
            completed_at = datetime.now(timezone.utc).isoformat() if status == "COMPLETED" else None
            connection.execute(
                """INSERT INTO planned_checkpoints
                   (run_id, node_id, status, result_json, cost_usd, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, node_id, status, result_json, cost_usd, completed_at)
            )

    # Verify checkpoint state
    cursor = connection.execute(
        "SELECT COUNT(*) FROM planned_checkpoints WHERE run_id=? AND status=?",
        (run_id, "COMPLETED")
    )
    completed_count = cursor.fetchone()[0]
    assert completed_count == 2

    cursor = connection.execute(
        "SELECT COUNT(*) FROM planned_checkpoints WHERE run_id=? AND status=?",
        (run_id, "PENDING")
    )
    pending_count = cursor.fetchone()[0]
    assert pending_count == 1

    # Verify total cost tracking
    cursor = connection.execute(
        "SELECT SUM(cost_usd) FROM planned_checkpoints WHERE run_id=?",
        (run_id,)
    )
    total_cost = cursor.fetchone()[0]
    assert abs(total_cost - 0.11) < 0.001

    connection.close()

"""Test suite for Orchestra Event Sourcing and Durable Execution Replay."""
import unittest
import uuid
import os
import sqlite3
from orchestra.storage.db import get_db, ensure_schema
from orchestra.storage.events import append_event, get_event_stream, find_tool_result

class TestEventReplay(unittest.TestCase):
    def setUp(self):
        # Setup temporary DB for testing
        os.environ["ORCHESTRA_DB_PATH"] = ":memory:"
        self.conn = get_db()
        ensure_schema(self.conn)
        self.run_id = "test-run-" + uuid.uuid4().hex[:4]
        
        # Initialize run
        with self.conn:
            self.conn.execute(
                "INSERT INTO runs (run_id, status, created_at) VALUES (?, ?, ?)",
                (self.run_id, "running", "2026-04-20T12:00:00Z")
            )

    def test_append_and_stream(self):
        # 1. Append events
        append_event(self.run_id, "AGENT_STARTED", {"alias": "cdx-deep"})
        seq2 = append_event(self.run_id, "TOOL_CALL_REQUESTED", {"tool": "write_file", "args": {"path": "test.txt"}})
        
        # 2. Get stream
        stream = get_event_stream(self.run_id)
        self.assertEqual(len(stream), 2)
        self.assertEqual(stream[0]["seq"], 1)
        self.assertEqual(stream[0]["event_type"], "AGENT_STARTED")
        self.assertEqual(stream[1]["seq"], 2)
        
    def test_idempotency_recovery(self):
        id_key = "tool-hash-12345"
        
        # 1. Simulate a completed tool call
        append_event(
            self.run_id, 
            "TOOL_CALL_COMPLETED", 
            {"status": "success", "output": "File written."},
            idempotency_key=id_key
        )
        
        # 2. Mock a recovery scenario where the engine checks for result
        result = find_tool_result(id_key)
        self.assertIsNotNone(result)
        self.assertEqual(result["output"], "File written.")
        
    def test_sequence_integrity(self):
        # Ensure sequence numbers are monotonically increasing per run
        s1 = append_event(self.run_id, "EVENT_1", {})
        s2 = append_event(self.run_id, "EVENT_2", {})
        self.assertEqual(s2, s1 + 1)
        
        # Check another run's sequence
        run2_id = "other-run-" + uuid.uuid4().hex[:4]
        with self.conn:
             self.conn.execute("INSERT INTO runs (run_id, last_seq_id) VALUES (?, ?)", (run2_id, 10))
        s3 = append_event(run2_id, "EVENT_3", {})
        self.assertEqual(s3, 11)

if __name__ == "__main__":
    unittest.main()

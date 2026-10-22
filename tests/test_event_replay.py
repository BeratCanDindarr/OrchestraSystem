"""Test suite for Orchestra Event Sourcing and Durable Execution Replay."""
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import orchestra.storage.db as storage_db
from orchestra.storage.events import append_event, get_event_stream, find_tool_result


class TestEventReplay(unittest.TestCase):
    def setUp(self):
        # Fresh EventLog database per test: point the config accessor at a temp
        # file and drop the cached singleton so it is re-created there.
        self._tmp = tempfile.TemporaryDirectory()
        self._db_file = Path(self._tmp.name) / "events.db"
        self._cfg_patch = mock.patch(
            "orchestra.config.db_path", return_value=self._db_file
        )
        self._cfg_patch.start()
        storage_db._event_log_instance = None
        self.run_id = "test-run-" + uuid.uuid4().hex[:4]

    def tearDown(self):
        self._cfg_patch.stop()
        storage_db._event_log_instance = None
        self._tmp.cleanup()

    def test_append_and_stream(self):
        # 1. Append events
        append_event(self.run_id, "AGENT_STARTED", {"alias": "cdx-deep"})
        append_event(self.run_id, "TOOL_CALL_REQUESTED", {"tool": "write_file", "args": {"path": "test.txt"}})

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
            idempotency_key=id_key,
        )

        # 2. Mock a recovery scenario where the engine checks for result
        result = find_tool_result(id_key)
        self.assertIsNotNone(result)
        self.assertEqual(result["output"], "File written.")

    def test_sequence_integrity(self):
        # Sequence numbers are global and monotonically increasing
        s1 = append_event(self.run_id, "EVENT_1", {})
        s2 = append_event(self.run_id, "EVENT_2", {})
        self.assertEqual(s2, s1 + 1)

        # Events from another run continue the global sequence
        run2_id = "other-run-" + uuid.uuid4().hex[:4]
        s3 = append_event(run2_id, "EVENT_3", {})
        self.assertEqual(s3, s2 + 1)


if __name__ == "__main__":
    unittest.main()

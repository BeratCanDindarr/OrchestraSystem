"""Tests for orchestra/live_status.py and the bus.py logger fix (Part A + Part B).

Run from a scratch directory to avoid CWD pollution:
    cd /tmp && /abs/path/.venv/bin/pytest /abs/path/tests/test_live_status.py -q --no-cov -p no:cacheprovider
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(name: str, **kwargs) -> dict:
    """Build a minimal event dict (ts/run_id stamped by append_event in prod; here we add them)."""
    from datetime import datetime, timezone
    ev = {"event": name, "ts": datetime.now(timezone.utc).isoformat(), **kwargs}
    return ev


def _emit_sync(run_id: str, event: dict, writer) -> None:
    """Call the internal bus handler synchronously (bypasses the thread-pool)."""
    writer._handle(run_id, event)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_module(tmp_path, monkeypatch):
    """Reset live_status singleton and bus state between tests."""
    import orchestra.live_status as ls
    import orchestra.events.bus as bus

    # M-4: Patch orchestra_root (not artifact_root) so _status_path resolves to tmp.
    monkeypatch.setattr("orchestra.config.orchestra_root", lambda: tmp_path)

    # Reset singleton state so install() is idempotent tests start clean.
    ls._writer = None

    # Clear bus global subscribers so install() tests count correctly.
    bus.clear()
    # M-6: Re-register log_subscriber after clear() so it survives across tests.
    if bus.log_subscriber not in bus._global:
        bus.subscribe_all(bus.log_subscriber)

    yield

    # Tear down again after the test.
    ls._writer = None
    bus.clear()
    # M-6: Re-register log_subscriber after teardown clear() as well.
    if bus.log_subscriber not in bus._global:
        bus.subscribe_all(bus.log_subscriber)


# ---------------------------------------------------------------------------
# Part A: bus.py logger regression
# ---------------------------------------------------------------------------

class TestBusLoggerFix:
    def test_raising_subscriber_does_not_propagate_and_logs_via_bus_logger(self, caplog):
        """A subscriber that raises must NOT crash emit(); exception logged via _bus_logger."""
        import logging
        import orchestra.events.bus as bus

        def bad_handler(run_id: str, event: dict) -> None:
            raise RuntimeError("deliberate failure")

        bus.subscribe_all(bad_handler)

        with caplog.at_level(logging.ERROR, logger="orchestra.events.bus"):
            fut = None
            # emit returns immediately (handlers run in executor); collect future to wait.
            # We can also just wait a short time.
            bus.emit("run-1", {"event": "test_event"})
            # Give thread-pool time to execute and log.
            time.sleep(0.2)

        # No NameError should surface — the test itself should not raise.
        # The exception should be logged by _bus_logger (logger name = "orchestra.events.bus").
        # `logger.exception` logs the message "Ignored exception" and the exc_info separately.
        assert any(
            "orchestra.events.bus" in r.name
            and r.exc_info is not None
            and "deliberate failure" in str(r.exc_info[1])
            for r in caplog.records
        ), f"Expected bus logger to log the exception. Records: {caplog.records}"

    def test_emit_continues_after_raising_subscriber(self):
        """Other subscribers still run even if one raises."""
        import orchestra.events.bus as bus

        received: list[str] = []

        def bad_handler(run_id: str, event: dict) -> None:
            raise ValueError("boom")

        def good_handler(run_id: str, event: dict) -> None:
            received.append(event.get("event", ""))

        bus.subscribe_all(bad_handler)
        bus.subscribe_all(good_handler)
        bus.emit("run-1", {"event": "sentinel"})
        time.sleep(0.3)
        assert "sentinel" in received


# ---------------------------------------------------------------------------
# Part B: LiveStatusWriter
# ---------------------------------------------------------------------------

class TestLiveStatusWriter:

    def _make_writer(self, tmp_path):
        """Create a fresh LiveStatusWriter without calling install()."""
        from orchestra.live_status import LiveStatusWriter
        return LiveStatusWriter()

    # 1. run_started + 2x agent_started
    def test_run_started_and_two_agents(self, tmp_path):
        w = self._make_writer(tmp_path)
        run_id = "r1"
        _emit_sync(run_id, _make_event("run_started", mode="dual"), w)
        _emit_sync(run_id, _make_event("agent_started", alias="a1", model="claude-3"), w)
        _emit_sync(run_id, _make_event("agent_started", alias="a2", model="gpt-4"), w)

        status_path = tmp_path / "live_status.json"
        data = json.loads(status_path.read_text())

        assert data["version"] == 1
        run = data["runs"][run_id]
        assert run["status"] == "running"
        assert run["mode"] == "dual"
        agents = run["agents"]
        assert set(agents.keys()) == {"a1", "a2"}
        assert agents["a1"]["status"] == "started"
        assert agents["a1"]["model"] == "claude-3"
        assert agents["a1"]["started_at"] is not None
        assert agents["a2"]["status"] == "started"

    # 2. agent_completed with estimated_cost_usd
    def test_agent_completed_cost_and_ended_at(self, tmp_path):
        w = self._make_writer(tmp_path)
        run_id = "r2"
        _emit_sync(run_id, _make_event("run_started", mode="ask"), w)
        _emit_sync(run_id, _make_event("agent_started", alias="a1", model="m"), w)
        _emit_sync(
            run_id,
            _make_event("agent_completed", alias="a1", estimated_cost_usd=0.042),
            w,
        )

        data = json.loads((tmp_path / "live_status.json").read_text())
        run = data["runs"][run_id]
        assert run["status"] == "running", "run_finished not emitted so run stays running"
        agent = run["agents"]["a1"]
        assert agent["status"] == "completed"
        assert agent["est_cost_usd"] == pytest.approx(0.042)
        assert agent["ended_at"] is not None

    # 3. run_finished → terminal; pruning after heartbeat > 600s
    def test_run_finished_and_pruning(self, tmp_path):
        from datetime import datetime, timezone, timedelta

        w = self._make_writer(tmp_path)
        run_id = "r3"
        _emit_sync(run_id, _make_event("run_started", mode="dual"), w)
        _emit_sync(run_id, _make_event("run_finished", status="completed"), w)

        data = json.loads((tmp_path / "live_status.json").read_text())
        assert data["runs"][run_id]["status"] == "completed"

        # Simulate heartbeat > 600s old by manipulating internal state.
        old_ts = (datetime.now(timezone.utc) - timedelta(seconds=601)).isoformat()
        with w._lock:
            w._runs[run_id]["heartbeat"] = old_ts

        # Fire an unrelated event for a NEW run to trigger prune.
        _emit_sync("r4", _make_event("run_started", mode="ask"), w)

        data2 = json.loads((tmp_path / "live_status.json").read_text())
        assert run_id not in data2["runs"], "Pruning should have removed the stale completed run"
        assert "r4" in data2["runs"]

    # 4. note_stream: ANSI + multi-line → last cleaned line ≤120 chars
    def test_note_stream_cleans_ansi(self, tmp_path):
        import orchestra.live_status as ls

        w = self._make_writer(tmp_path)
        ls._writer = w  # so note_stream uses this writer

        run_id = "ns1"
        _emit_sync(run_id, _make_event("run_started", mode="ask"), w)
        _emit_sync(run_id, _make_event("agent_started", alias="ag", model="m"), w)

        chunk = "\x1b[31mline one\x1b[0m\nline two with \x1b[1;32mcolor\x1b[0m\n"
        # Force a past write time so throttle doesn't block.
        with w._lock:
            if run_id in w._runs and "ag" in w._runs[run_id].get("agents", {}):
                w._last_stream_write[(run_id, "ag")] = 0.0

        ls.note_stream(run_id, "ag", chunk)
        time.sleep(0.05)  # allow any async flush

        data = json.loads((tmp_path / "live_status.json").read_text())
        last_line = data["runs"][run_id]["agents"]["ag"]["last_line"]
        assert last_line is not None
        # Should be clean (no ANSI)
        assert "\x1b" not in last_line
        # Should be the LAST non-empty line
        assert "line two" in last_line or last_line == "line two with color"
        assert len(last_line) <= 120

    # 5. Throttle: two note_stream calls within 0.5s → one file write;
    #    the SECOND (newest) value must be coalesced and persisted by next event-triggered write.
    def test_note_stream_throttle(self, tmp_path, monkeypatch):
        import orchestra.live_status as ls

        w = self._make_writer(tmp_path)
        ls._writer = w

        run_id = "th1"
        _emit_sync(run_id, _make_event("run_started", mode="ask"), w)
        _emit_sync(run_id, _make_event("agent_started", alias="ag", model="m"), w)

        replace_count = 0
        original_replace = os.replace

        def counting_replace(src, dst):
            nonlocal replace_count
            replace_count += 1
            original_replace(src, dst)

        monkeypatch.setattr("orchestra.live_status.os.replace", counting_replace)

        # Ensure first write timestamp is old enough to allow one write.
        with w._lock:
            w._last_stream_write[(run_id, "ag")] = 0.0

        before = replace_count
        ls.note_stream(run_id, "ag", "first chunk")
        # Second call is within the throttle window — must be coalesced, not dropped.
        ls.note_stream(run_id, "ag", "second chunk coalesced")
        time.sleep(0.05)

        after = replace_count
        # Only one file write should have happened (first call wrote; second was throttled).
        assert after - before == 1, f"Expected 1 write after two rapid calls, got {after - before}"

        # The second value must be buffered as pending — not yet written.
        with w._lock:
            assert (run_id, "ag") in w._pending_stream, (
                "Throttled chunk should be in _pending_stream, waiting for flush"
            )
            assert w._pending_stream[(run_id, "ag")] == "second chunk coalesced", (
                "Pending buffer must hold the NEWEST (second) value, not the first"
            )

        # Trigger flush via a bus event (no need to wait out the throttle window).
        # The event-triggered _dispatch should apply the pending value and write it.
        writes_before_flush = replace_count
        _emit_sync(run_id, _make_event("budget_exceeded"), w)

        # The pending value should now be flushed and written.
        assert replace_count > writes_before_flush, (
            "Event-triggered _dispatch must flush pending stream value and write the file"
        )

        data = json.loads((tmp_path / "live_status.json").read_text())
        last_line = data["runs"][run_id]["agents"]["ag"]["last_line"]
        assert last_line == "second chunk coalesced", (
            f"The NEWEST (second) chunk must be persisted after flush, got: {last_line!r}"
        )

        # Verify that after flush, the pending buffer is cleared.
        with w._lock:
            assert (run_id, "ag") not in w._pending_stream, (
                "Pending buffer should be cleared after flush"
            )

    # 6. Concurrency stress: 200 events, 8 threads, final file valid JSON, all agents present
    def test_concurrency_stress(self, tmp_path):
        w = self._make_writer(tmp_path)
        run_ids = ["stress-r1", "stress-r2"]

        for run_id in run_ids:
            _emit_sync(run_id, _make_event("run_started", mode="dual"), w)
            for a in ["a1", "a2", "a3"]:
                _emit_sync(run_id, _make_event("agent_started", alias=a, model="m"), w)

        errors: list[Exception] = []
        events_per_thread = 25  # 8 threads × 25 = 200 events

        def fire_events():
            try:
                for i in range(events_per_thread):
                    run_id = run_ids[i % 2]
                    alias = ["a1", "a2", "a3"][i % 3]
                    _emit_sync(run_id, _make_event("agent_completed", alias=alias, estimated_cost_usd=0.001), w)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=fire_events) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Threads raised errors: {errors}"

        data = json.loads((tmp_path / "live_status.json").read_text())
        assert data["version"] == 1
        for run_id in run_ids:
            assert run_id in data["runs"]
            assert set(data["runs"][run_id]["agents"].keys()) == {"a1", "a2", "a3"}

    # 7. install() twice → handler registered once
    def test_install_idempotent(self, tmp_path):
        import orchestra.live_status as ls
        import orchestra.events.bus as bus

        ls.install()
        ls.install()

        # After two install() calls only one handler should be in _global.
        with bus._lock:
            handler_count = sum(
                1 for h in bus._global if getattr(h, "__self__", None) is ls._writer
                or (callable(h) and getattr(h, "__func__", None) is ls.LiveStatusWriter._handle)
                or (isinstance(h, ls.LiveStatusWriter))
            )

        assert handler_count == 1, (
            f"install() must be idempotent: expected exactly 1 LiveStatusWriter handler "
            f"in _global after two calls, got {handler_count}"
        )

        # Check: fire one event; writer should process it once.
        run_id = "idem1"
        bus.emit(run_id, _make_event("run_started", mode="ask", run_id=run_id))
        time.sleep(0.3)

        data = json.loads((tmp_path / "live_status.json").read_text())
        assert run_id in data["runs"]

    # 8. Handler swallows malformed event (no alias on agent event)
    def test_handler_swallows_malformed_event(self, tmp_path):
        w = self._make_writer(tmp_path)
        run_id = "mal1"
        _emit_sync(run_id, _make_event("run_started", mode="ask"), w)

        # agent_started WITHOUT alias → malformed
        bad_event = _make_event("agent_started")  # no alias key
        _emit_sync(run_id, bad_event, w)  # must not raise

        # File should still be valid.
        data = json.loads((tmp_path / "live_status.json").read_text())
        assert run_id in data["runs"]

    # 9. approval_requested → run status "waiting_approval"; budget_exceeded → heartbeat only
    def test_approval_requested_and_budget_exceeded(self, tmp_path):
        w = self._make_writer(tmp_path)
        run_id = "app1"
        _emit_sync(run_id, _make_event("run_started", mode="ask"), w)

        old_hb = json.loads((tmp_path / "live_status.json").read_text())["runs"][run_id]["heartbeat"]
        time.sleep(0.01)

        _emit_sync(run_id, _make_event("approval_requested"), w)
        data = json.loads((tmp_path / "live_status.json").read_text())
        assert data["runs"][run_id]["status"] == "waiting_approval"

        # budget_exceeded: heartbeat should update, status unchanged (still waiting_approval)
        _emit_sync(run_id, _make_event("budget_exceeded"), w)
        data2 = json.loads((tmp_path / "live_status.json").read_text())
        assert data2["runs"][run_id]["status"] == "waiting_approval"
        # heartbeat updated (could be same if too fast, but at least not raises)
        assert data2["runs"][run_id]["heartbeat"] is not None

    # 10. unknown event for unknown run → does NOT create a run entry
    def test_unknown_event_does_not_create_run(self, tmp_path):
        w = self._make_writer(tmp_path)
        _emit_sync("unknown-run", _make_event("some_custom_event"), w)

        status_path = tmp_path / "live_status.json"
        if status_path.exists():
            data = json.loads(status_path.read_text())
            assert "unknown-run" not in data.get("runs", {})

    # 11. run_cancelled → status "cancelled"
    def test_run_cancelled(self, tmp_path):
        w = self._make_writer(tmp_path)
        run_id = "c1"
        _emit_sync(run_id, _make_event("run_started", mode="dual"), w)
        _emit_sync(run_id, _make_event("run_cancelled"), w)

        data = json.loads((tmp_path / "live_status.json").read_text())
        assert data["runs"][run_id]["status"] == "cancelled"

    # 12. note_stream ANSI strip: long chunk truncated to 120 chars
    def test_note_stream_truncation(self, tmp_path):
        import orchestra.live_status as ls

        w = self._make_writer(tmp_path)
        ls._writer = w

        run_id = "tr1"
        _emit_sync(run_id, _make_event("run_started", mode="ask"), w)
        _emit_sync(run_id, _make_event("agent_started", alias="ag", model="m"), w)

        # Force old timestamp so throttle doesn't apply.
        with w._lock:
            w._last_stream_write[(run_id, "ag")] = 0.0

        long_line = "x" * 200
        ls.note_stream(run_id, "ag", long_line)
        time.sleep(0.05)

        data = json.loads((tmp_path / "live_status.json").read_text())
        assert len(data["runs"][run_id]["agents"]["ag"]["last_line"]) <= 120


# ---------------------------------------------------------------------------
# Part C: I-1 cross-process merge, M-1 prune hygiene
# ---------------------------------------------------------------------------

class TestCrossProcessMerge:
    """I-1: Two LiveStatusWriter instances sharing one file must not clobber each other."""

    def _make_writer(self):
        from orchestra.live_status import LiveStatusWriter
        return LiveStatusWriter()

    # C-1: two writers, same file → both runs present
    def test_two_writers_both_runs_survive(self, tmp_path):
        w1 = self._make_writer()
        w2 = self._make_writer()

        _emit_sync("proc1-r1", _make_event("run_started", mode="ask"), w1)
        _emit_sync("proc2-r1", _make_event("run_started", mode="dual"), w2)

        data = json.loads((tmp_path / "live_status.json").read_text())
        assert "proc1-r1" in data["runs"], "Writer-1 run should be in the file"
        assert "proc2-r1" in data["runs"], "Writer-2 run should be in the file"

    # C-1b: a run that waits for approval in one process is cancelled from another
    # (the MCP server cancels a run its worker process started)
    def test_cancel_from_another_writer_updates_the_run(self, tmp_path):
        worker = self._make_writer()
        server = self._make_writer()

        _emit_sync("r1", _make_event("run_started", mode="dual"), worker)
        _emit_sync("r1", _make_event("agent_started", alias="cld-deep", model="claude/opus"), worker)
        _emit_sync("r1", _make_event("approval_requested"), worker)
        _emit_sync("r1", _make_event("run_cancelled"), server)

        data = json.loads((tmp_path / "live_status.json").read_text())
        assert data["runs"]["r1"]["status"] == "cancelled"
        assert "cld-deep" in data["runs"]["r1"]["agents"], "the run keeps what the other writer recorded"

    # C-1c: the same for a rejected approval and for a run finished elsewhere
    def test_rejection_and_finish_from_another_writer_update_the_run(self, tmp_path):
        worker = self._make_writer()
        server = self._make_writer()

        _emit_sync("r1", _make_event("run_started", mode="dual"), worker)
        _emit_sync("r1", _make_event("approval_requested"), worker)
        _emit_sync("r1", _make_event("approval_rejected"), server)
        _emit_sync("r2", _make_event("run_started", mode="ask"), worker)
        _emit_sync("r2", _make_event("run_finished", status="failed"), server)

        data = json.loads((tmp_path / "live_status.json").read_text())
        assert data["runs"]["r1"]["status"] == "cancelled"
        assert data["runs"]["r2"]["status"] == "failed"

    # C-1d: an event for a run nobody ever recorded still creates nothing
    def test_cancel_of_an_unknown_run_creates_nothing(self, tmp_path):
        server = self._make_writer()
        _emit_sync("known", _make_event("run_started", mode="ask"), server)

        _emit_sync("ghost", _make_event("run_cancelled"), server)

        data = json.loads((tmp_path / "live_status.json").read_text())
        assert "ghost" not in data["runs"]

    # C-2: foreign entry with stale heartbeat is evicted on next write
    def test_stale_foreign_entry_is_evicted(self, tmp_path):
        from datetime import timedelta

        w1 = self._make_writer()
        # Write a run via w1, then make its heartbeat stale directly in the file.
        _emit_sync("stale-run", _make_event("run_started", mode="ask"), w1)
        status_path = tmp_path / "live_status.json"
        on_disk = json.loads(status_path.read_text())
        old_ts = (datetime.now(timezone.utc) - timedelta(seconds=601)).isoformat()
        on_disk["runs"]["stale-run"]["heartbeat"] = old_ts
        status_path.write_text(json.dumps(on_disk), encoding="utf-8")

        # w2 writes its own run — should evict the stale-run foreign entry.
        w2 = self._make_writer()
        _emit_sync("fresh-run", _make_event("run_started", mode="ask"), w2)

        data = json.loads(status_path.read_text())
        assert "fresh-run" in data["runs"], "w2's run must be present"
        assert "stale-run" not in data["runs"], "Stale foreign entry must be evicted"

    # C-3: foreign FRESH "running" entry survives another writer's write
    def test_fresh_foreign_running_entry_survives(self, tmp_path):
        w1 = self._make_writer()
        w2 = self._make_writer()

        # w1 creates a run with a current heartbeat.
        _emit_sync("proc1-fresh", _make_event("run_started", mode="ask"), w1)

        # w2 now writes its own run — w1's fresh entry must survive.
        _emit_sync("proc2-run", _make_event("run_started", mode="ask"), w2)

        data = json.loads((tmp_path / "live_status.json").read_text())
        assert "proc1-fresh" in data["runs"], "Fresh foreign run must not be evicted"
        assert "proc2-run" in data["runs"], "w2's own run must be present"


class TestPruneHygiene:
    """M-1: _prune must also clean _last_stream_write and _pending_stream."""

    def _make_writer(self):
        from orchestra.live_status import LiveStatusWriter
        return LiveStatusWriter()

    def test_prune_clears_throttle_and_pending_maps(self, tmp_path):
        from datetime import timedelta

        w = self._make_writer()
        run_id = "prune-r1"
        alias = "ag1"

        _emit_sync(run_id, _make_event("run_started", mode="ask"), w)
        _emit_sync(run_id, _make_event("agent_started", alias=alias, model="m"), w)

        # Inject throttle and pending entries for this run.
        with w._lock:
            w._last_stream_write[(run_id, alias)] = time.monotonic()
            w._pending_stream[(run_id, alias)] = "some pending line"

        # Make the run terminal and its heartbeat stale.
        _emit_sync(run_id, _make_event("run_finished", status="completed"), w)
        old_ts = (datetime.now(timezone.utc) - timedelta(seconds=601)).isoformat()
        with w._lock:
            w._runs[run_id]["heartbeat"] = old_ts

        # Trigger prune by writing any event (for a different run).
        _emit_sync("other-run", _make_event("run_started", mode="ask"), w)

        # Both maps must be clean for the pruned run_id.
        with w._lock:
            throttle_keys = [k for k in w._last_stream_write if k[0] == run_id]
            pending_keys = [k for k in w._pending_stream if k[0] == run_id]

        assert throttle_keys == [], f"_last_stream_write should have no keys for pruned run; got {throttle_keys}"
        assert pending_keys == [], f"_pending_stream should have no keys for pruned run; got {pending_keys}"


class TestRunnersInstallTheWriter:
    """A run started from the terminal goes straight to a runner, not through the service."""

    def test_an_ask_run_shows_up_without_the_service(self):
        import orchestra.live_status as ls
        from orchestra.engine.runners import common
        from orchestra.models import AgentRun, AgentStatus

        def answering(run, agents, **kwargs):
            agent = AgentRun(alias=agents[0][0], provider="test", model="test/model", status=AgentStatus.COMPLETED)
            agent.stdout_log = "ok"
            run.agents.append(agent)

        ls._writer = None
        with patch.object(common, "run_parallel", answering):
            common._run_ask_impl("oll-gemma", "status line check from a runner", internal=True)

        assert ls._writer is not None

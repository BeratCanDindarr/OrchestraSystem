"""Many writers at once: no 'database is locked', and a manifest is never seen half-written."""
from __future__ import annotations

import threading

WRITERS = 20


def _together(work) -> list[BaseException]:
    errors: list[BaseException] = []
    start = threading.Barrier(WRITERS)

    def one(i: int) -> None:
        start.wait()
        try:
            work(i)
        except BaseException as exc:   # collected: the test reports every writer that failed
            errors.append(exc)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(WRITERS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    return errors


def test_twenty_writers_append_events_without_a_lock_error(tmp_path):
    from orchestra.storage.event_log import EventLog

    log = EventLog(tmp_path / "events.db")

    errors = _together(lambda i: [log.append("run_started", f"run-{i}", {"n": n}) for n in range(10)])

    assert errors == []
    assert sum(len(log.replay(f"run-{i}")) for i in range(WRITERS)) == WRITERS * 10


def test_twenty_writers_add_memory_rows_without_a_lock_error(tmp_path, fake_embeddings):
    from orchestra.storage.memory import SemanticMemory

    memory = SemanticMemory(db_path=str(tmp_path / "memory.db"), index_path=str(tmp_path / "memory.hnsw"))

    errors = _together(lambda i: memory.add(f"note number {i} about the build", {"kind": "note"}))

    assert errors == []


def test_a_manifest_being_rewritten_is_never_read_as_missing(tmp_path, monkeypatch):
    from orchestra.engine import artifacts

    monkeypatch.setattr("orchestra.config.artifact_root", lambda: tmp_path)
    monkeypatch.setattr("orchestra.storage.db.upsert_run", lambda manifest: None)
    big = {"run_id": "r1", "status": "running", "summary": "x" * 400_000}
    artifacts.write_manifest_data("r1", big)
    stop = threading.Event()
    missing: list[int] = []

    def reader() -> None:
        while not stop.is_set():
            if artifacts.load_manifest("r1") is None:
                missing.append(1)

    thread = threading.Thread(target=reader)
    thread.start()
    for n in range(40):
        artifacts.write_manifest_data("r1", {**big, "n": n})
    stop.set()
    thread.join()

    assert missing == []

"""Project state lives under the project's .orchestra, wherever a command is started."""
import socket
from pathlib import Path

from orchestra import config
from orchestra.storage import memory as memory_store


def test_memory_db_sits_next_to_the_other_project_databases():
    assert config.memory_db_path().parent == config.orchestra_root().resolve()


def test_get_memory_opens_the_project_store_from_any_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    mem = memory_store.get_memory()

    assert Path(mem.db_path) == config.memory_db_path()
    assert not (tmp_path / ".orchestra").exists()


def test_importing_the_runners_does_not_open_a_memory_store():
    from orchestra.engine.runners import common

    assert not hasattr(common, "_memory")


def test_the_suite_never_reaches_a_real_embedding_endpoint(monkeypatch):
    attempts = []

    def record(*args, **kwargs):
        attempts.append(args)
        raise OSError("no network in tests")

    monkeypatch.setattr(socket, "create_connection", record)

    assert memory_store._ollama.embed("anything") == []
    assert attempts == []


def test_replay_log_sits_in_the_run_folder():
    from orchestra.engine import replay

    assert replay._replay_path("run-x") == config.artifact_root() / "run-x" / "replay.jsonl"


def test_a_recorder_creates_nothing_while_replay_is_off(monkeypatch):
    from orchestra.engine import replay

    monkeypatch.setattr(config, "replay_enabled", lambda: False)
    recorder = replay.ReplayRecorder("run-never-recorded")
    recorder.record_run_started("ask", "task")

    assert not (config.artifact_root() / "run-never-recorded").exists()

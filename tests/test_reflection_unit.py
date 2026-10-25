"""Unit tests for orchestra.engine.reflection (self-critique + retry loop)."""
from __future__ import annotations

from unittest.mock import patch

from orchestra import config
from orchestra.engine import reflection
from orchestra.models import AgentRun, AgentStatus, OrchestraRun


class _FakeMemory:
    def __init__(self):
        self.added = []

    def add(self, content, **kwargs):
        self.added.append((content, kwargs))
        return True


def _critique_then_ok():
    return iter([
        _mk_run('{"needs_revision": true, "issues": ["vague"], "guidance": "be specific"}'),
        _mk_run("revised answer"),
        _mk_run('{"needs_revision": false, "issues": [], "guidance": ""}'),
    ])


def _mk_run(output: str) -> OrchestraRun:
    run = OrchestraRun(mode="ask", task="t")
    run.agents = [
        AgentRun(
            alias="a", provider="p", model="m",
            status=AgentStatus.COMPLETED, stdout_log=output,
        )
    ]
    return run


def test_is_reflecting_default_false():
    assert reflection.is_reflecting() is False


def test_reflect_accepts_when_no_revision_needed():
    base = _mk_run("good answer")
    calls = []

    def fake_ask(alias, prompt, **kwargs):
        calls.append(alias)
        return _mk_run('{"needs_revision": false, "issues": [], "guidance": ""}')

    out = reflection.reflect("cdx-deep", "task", base, run_ask_fn=fake_ask, max_rounds=2)
    assert out is base                 # unchanged when critic approves
    assert calls == ["cld-fast"]       # exactly one critique call, no retry
    assert reflection.is_reflecting() is False


def test_reflect_retries_then_accepts():
    base = _mk_run("weak answer")
    seq = iter([
        _mk_run('{"needs_revision": true, "issues": ["vague"], "guidance": "be specific"}'),
        _mk_run("strong revised answer"),
        _mk_run('{"needs_revision": false, "issues": [], "guidance": ""}'),
    ])

    def fake_ask(alias, prompt, **kwargs):
        return next(seq)

    out = reflection.reflect(
        "cdx-deep", "task", base, run_ask_fn=fake_ask, max_rounds=3, critique_alias="cld-fast",
    )
    assert reflection._first_output(out) == "strong revised answer"
    assert reflection.is_reflecting() is False


def test_reflect_stops_at_max_rounds_when_critic_never_satisfied():
    base = _mk_run("answer")
    calls = []

    def fake_ask(alias, prompt, **kwargs):
        calls.append(alias)
        if alias == "cld-fast":   # critique always demands revision
            return _mk_run('{"needs_revision": true, "issues": ["x"], "guidance": "y"}')
        return _mk_run("revised")  # retry always yields output

    out = reflection.reflect(
        "cdx-deep", "task", base, run_ask_fn=fake_ask, max_rounds=2, critique_alias="cld-fast",
    )
    assert reflection._first_output(out) == "revised"
    # 2 rounds => critique, retry, critique, retry
    assert calls == ["cld-fast", "cdx-deep", "cld-fast", "cdx-deep"]


def test_reflect_returns_base_when_already_reflecting():
    base = _mk_run("x")
    token = reflection._in_reflection.set(True)
    try:
        out = reflection.reflect(
            "a", "t", base, run_ask_fn=lambda *a, **k: _mk_run("y"), max_rounds=2,
        )
        assert out is base            # guard blocks re-entry
    finally:
        reflection._in_reflection.reset(token)


def test_reflect_returns_base_when_no_output():
    empty = OrchestraRun(mode="ask", task="t")   # no agents
    out = reflection.reflect(
        "a", "t", empty, run_ask_fn=lambda *a, **k: _mk_run("y"), max_rounds=2,
    )
    assert out is empty


def test_reflection_disabled_by_default():
    assert config.reflection_enabled() is False
    assert config.reflection_max_rounds() == 2
    assert config.reflection_critique_alias() == "cld-fast"
    assert config.reflection_record_lessons() is False


def test_reflect_records_lesson_to_memory_when_enabled():
    base = _mk_run("weak")
    seq = _critique_then_ok()
    mem = _FakeMemory()
    with patch("orchestra.config.reflection_record_lessons", return_value=True):
        reflection.reflect(
            "cdx-deep", "the task", base,
            run_ask_fn=lambda a, p, **k: next(seq), max_rounds=3, memory=mem,
        )
    assert len(mem.added) == 1
    content, kwargs = mem.added[0]
    assert content == "the task"                                  # embed on task, not lesson
    assert kwargs["metadata"]["reflection_lesson"].startswith("avoid: vague")
    assert kwargs["source"] == "reflection_lesson"


def test_reflect_records_no_lesson_when_disabled():
    base = _mk_run("weak")
    seq = _critique_then_ok()
    mem = _FakeMemory()
    # record_lessons is off by default -> no memory writes
    reflection.reflect(
        "cdx-deep", "t", base,
        run_ask_fn=lambda a, p, **k: next(seq), max_rounds=3, memory=mem,
    )
    assert mem.added == []

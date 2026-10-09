"""A cached answer must be the answer to the question that was asked.

Three ways it was not:
- the 48-hour replay store kept every outcome, so a run that stopped for approval or
  failed came back in place of a new attempt;
- there was no way to ask for a new answer;
- the cross-run cache matched on shared words, so a short prompt whose words all occurred
  in an earlier, different prompt got that prompt's answer, whichever model had given it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

from orchestra.engine.idempotency import idempotent
from orchestra.engine.runners import ask_runner, common
from orchestra.models import AgentRun, AgentStatus
from orchestra.storage.fts_cache import FtsSearchCache


@dataclass
class _Log:
    """An event log that keeps its events in a list."""

    events: list = field(default_factory=list)

    def append(self, event_type: str, run_id: str, payload: dict) -> None:
        self.events.append(SimpleNamespace(event_type=event_type, run_id=run_id, payload=payload, ts=9e12))

    def replay(self, run_id: str) -> list:
        return [event for event in self.events if event.run_id == run_id]


def _counting(results: list):
    """An operation that returns the given results in turn and counts its calls."""
    calls = {"count": 0}

    def operation(question: str, **kwargs):
        calls["count"] += 1
        return results[min(calls["count"], len(results)) - 1]

    return operation, calls


def _wrap(operation, log: _Log):
    return idempotent(
        operation="op",
        key_fn=lambda question, **kwargs: question,
        event_log=log,
        run_id="store",
        should_cache=lambda result: str(result).startswith("done"),
        bypass_kwarg="fresh",
    )(operation)


class TestReplayStore:
    def test_a_finished_result_is_replayed(self):
        operation, calls = _counting(["done"])
        wrapped = _wrap(operation, _Log())

        assert wrapped("q") == "done"
        assert wrapped("q") == "done"
        assert calls["count"] == 1

    def test_an_unfinished_result_is_not_kept(self):
        operation, calls = _counting(["waiting", "done"])
        wrapped = _wrap(operation, _Log())

        assert wrapped("q") == "waiting"
        assert wrapped("q") == "done"
        assert calls["count"] == 2

    def test_an_unfinished_result_stored_earlier_is_not_replayed(self):
        log = _Log()
        idempotent(operation="op", key_fn=lambda question, **kwargs: question, event_log=log,
                   run_id="store")(lambda question, **kwargs: "waiting")("q")
        operation, calls = _counting(["done"])

        assert _wrap(operation, log)("q") == "done"
        assert calls["count"] == 1

    def test_fresh_runs_again_and_its_result_is_the_one_replayed_afterwards(self):
        operation, calls = _counting(["done one", "done two"])
        wrapped = _wrap(operation, _Log())

        assert wrapped("q") == "done one"
        assert wrapped("q", fresh=True) == "done two"
        assert wrapped("q") == "done two"
        assert calls["count"] == 2


class TestCrossRunCache:
    def _cache(self) -> FtsSearchCache:
        return FtsSearchCache(db_path=":memory:")

    def test_the_same_question_to_the_same_model_is_a_hit(self):
        cache = self._cache()
        cache.store("r1", "Update the label every frame or once per second?", "Once per second.", alias="kimi-deep")

        hit = cache.lookup("Update the label every frame or once per second?", alias="kimi-deep")

        assert hit is not None and hit.response == "Once per second."

    def test_a_different_question_made_of_the_same_words_is_not(self):
        cache = self._cache()
        cache.store("r1", "Review the pooling code and list risks around the timer and the label update",
                    "The pool leaks on restart.", alias="kimi-deep")

        assert cache.lookup("timer", alias="kimi-deep") is None
        assert cache.lookup("update the label", alias="kimi-deep") is None

    def test_another_model_does_not_get_the_answer(self):
        cache = self._cache()
        cache.store("r1", "Name one risk of pooling.", "Stale state.", alias="kimi-deep")

        assert cache.lookup("Name one risk of pooling.", alias="oll-gemma") is None

    def test_an_expired_answer_is_not_a_hit(self):
        cache = self._cache()
        cache.store("r1", "Name one risk of pooling.", "Stale state.", alias="kimi-deep")
        cache.conn.execute("UPDATE fts_cache_metadata SET created_at = created_at - 10 * 3600")
        cache.conn.commit()

        assert cache.lookup("Name one risk of pooling.", alias="kimi-deep", ttl_hours=1) is None
        assert cache.lookup("Name one risk of pooling.", alias="kimi-deep", ttl_hours=24) is not None

    def test_storing_again_replaces_the_answer(self):
        cache = self._cache()
        cache.store("r1", "Name one risk of pooling.", "Stale state.", alias="kimi-deep")
        cache.store("r2", "Name one risk of pooling.", "Leaked references.", alias="kimi-deep")

        hit = cache.lookup("Name one risk of pooling.", alias="kimi-deep")

        assert hit is not None and hit.response == "Leaked references." and hit.run_id == "r2"
        assert cache.conn.execute("SELECT COUNT(*) FROM fts_cache").fetchone()[0] == 1


def _answering(text: str):
    calls = {"count": 0}

    def fake(run, agents, **kwargs):
        calls["count"] += 1
        agent = AgentRun(alias=agents[0][0], provider="test", model="test/model", status=AgentStatus.COMPLETED)
        agent.stdout_log = f"{text} {calls['count']}"
        run.agents.append(agent)

    return fake, calls


class TestAskUsesTheCache:
    def test_the_second_identical_ask_is_answered_from_the_cache(self):
        fake, calls = _answering("answer")
        with patch.object(common, "run_parallel", fake):
            common._run_ask_impl("oll-gemma", "cache trust: what is pooling for?")
            second = common._run_ask_impl("oll-gemma", "cache trust: what is pooling for?")

        assert calls["count"] == 1
        assert second.agents[0].provider == "cache"

    def test_fresh_asks_the_model_again(self):
        fake, calls = _answering("answer")
        with patch.object(common, "run_parallel", fake):
            common._run_ask_impl("oll-gemma", "cache trust: what is a tween?")
            second = common._run_ask_impl("oll-gemma", "cache trust: what is a tween?", fresh=True)
            third = common._run_ask_impl("oll-gemma", "cache trust: what is a tween?")

        assert calls["count"] == 2
        assert second.agents[0].stdout_log == "answer 2"
        assert third.agents[0].stdout_log == "answer 2"

    def test_another_alias_asks_its_own_model(self):
        fake, calls = _answering("answer")
        with patch.object(common, "run_parallel", fake):
            common._run_ask_impl("oll-gemma", "cache trust: what is a signal?")
            common._run_ask_impl("kimi-deep", "cache trust: what is a signal?")

        assert calls["count"] == 2

    def test_the_cache_can_be_switched_off(self, monkeypatch):
        from orchestra import config

        monkeypatch.setattr(config, "fts_cache_enabled", lambda: False, raising=False)
        fake, calls = _answering("answer")
        with patch.object(common, "run_parallel", fake):
            common._run_ask_impl("oll-gemma", "cache trust: what is an installer?")
            common._run_ask_impl("oll-gemma", "cache trust: what is an installer?")

        assert calls["count"] == 2


class TestRunAskEntry:
    def test_an_internal_step_is_never_replayed(self, monkeypatch):
        from orchestra import config

        monkeypatch.setattr(config, "idempotency_enabled", lambda: True)
        with patch.object(ask_runner, "_run_ask_idempotent", side_effect=AssertionError("replayed")), \
                patch.object(ask_runner, "_run_ask_impl", return_value="ran") as impl, \
                patch.object(config, "retry_enabled", lambda: False), \
                patch.object(config, "reflection_enabled", lambda: False):
            assert ask_runner.run_ask("oll-gemma", "verify", internal=True) == "ran"

        impl.assert_called_once()

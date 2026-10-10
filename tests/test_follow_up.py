"""A run can continue an earlier run and is then given that discussion."""
from __future__ import annotations

from unittest.mock import patch

from orchestra import service
from orchestra.engine import artifacts, follow_up
from orchestra.engine.runners import ask_runner, common
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus


def _finished(task: str, answer: str, aliases=("cld-deep",), follows: str | None = None) -> OrchestraRun:
    run = OrchestraRun(mode="ask" if len(aliases) == 1 else "dual", task=task, follows_run_id=follows)
    for alias in aliases:
        run.agents.append(AgentRun(alias=alias, provider="test", model="test/model", status=AgentStatus.COMPLETED))
    run.summary = answer
    run.status = RunStatus.COMPLETED
    artifacts.write_manifest(run)
    return run


def _answering():
    seen = {"prompts": []}

    def fake(run, agents, **kwargs):
        alias, prompt = agents[0]
        seen["prompts"].append(prompt)
        agent = AgentRun(alias=alias, provider="test", model="test/model", status=AgentStatus.COMPLETED)
        agent.stdout_log = f"answer {len(seen['prompts'])}"
        run.agents.append(agent)

    return fake, seen


class TestHistory:
    def test_a_run_that_follows_nothing_gets_its_prompt_unchanged(self):
        assert follow_up.with_history(None, "What is pooling?") == "What is pooling?"

    def test_the_discussion_is_the_chain_of_runs_oldest_first(self):
        first = _finished("Should the HUD timer update every frame?", "No: once per second.")
        second = _finished("And with a millisecond display?", "Then every frame.", follows=first.run_id)

        text = follow_up.with_history(second.run_id, "What about battery?")

        assert text.index("Should the HUD timer") < text.index("And with a millisecond display?") < text.index("[NEW MESSAGE]")
        assert "No: once per second." in text and "Then every frame." in text
        assert text.endswith("What about battery?")

    def test_only_the_newest_turns_are_kept(self, monkeypatch):
        monkeypatch.setattr(follow_up, "max_turns", lambda: 2)
        run = _finished("turn one", "a1")
        run = _finished("turn two", "a2", follows=run.run_id)
        run = _finished("turn three", "a3", follows=run.run_id)

        text = follow_up.with_history(run.run_id, "next")

        assert "turn one" not in text and "turn two" in text and "turn three" in text

    def test_a_long_answer_is_cut(self, monkeypatch):
        monkeypatch.setattr(follow_up, "answer_chars", lambda: 20)
        run = _finished("q", "x" * 500)

        assert "x" * 21 not in follow_up.with_history(run.run_id, "next")

    def test_the_verifier_is_not_named_as_an_answerer(self):
        run = _finished("q", "a", aliases=("cld-deep", "kimi-deep", "verifier"))

        assert "Answered (cld-deep, kimi-deep)" in follow_up.with_history(run.run_id, "next")


class TestFollowUpRuns:
    def test_the_agent_is_given_the_discussion(self):
        first = _finished("Should the HUD timer update every frame?", "No: once per second.")
        fake, seen = _answering()
        with patch.object(common, "run_parallel", fake):
            run = common._run_ask_impl("cld-deep", "What about battery?", follows_run_id=first.run_id)

        assert "No: once per second." in seen["prompts"][0]
        assert run.task == "What about battery?"
        assert artifacts.load_manifest(run.run_id)["follows_run_id"] == first.run_id

    def test_a_follow_up_is_neither_answered_from_nor_written_to_the_stores(self):
        first = _finished("store check: first question", "first answer")
        fake, seen = _answering()
        with patch.object(common, "run_parallel", fake), \
                patch.object(common, "_try_cache_hit", side_effect=AssertionError("cache read")), \
                patch.object(common, "_store_in_cache", side_effect=AssertionError("cache written")):
            common._run_ask_impl("cld-deep", "store check: explain more", follows_run_id=first.run_id)

        with patch.object(ask_runner, "_run_ask_idempotent", side_effect=AssertionError("replayed")), \
                patch.object(ask_runner, "_run_ask_impl", return_value="ran"), \
                patch("orchestra.config.idempotency_enabled", lambda: True), \
                patch("orchestra.config.retry_enabled", lambda: False), \
                patch("orchestra.config.reflection_enabled", lambda: False):
            assert ask_runner.run_ask("cld-deep", "explain more", follows_run_id=first.run_id) == "ran"

    def test_the_service_continues_with_the_agent_that_answered(self):
        first = _finished("service check: first question", "first answer", aliases=("kimi-deep",))
        fake, seen = _answering()
        with patch.object(common, "run_parallel", fake):
            result = service.run_task(mode="ask", task="service check: and then?", follow_up_to=first.run_id)

        assert result["run"]["agents"][0]["alias"] == "kimi-deep"
        assert result["run"]["follows_run_id"] == first.run_id
        assert "first answer" in seen["prompts"][0]

    def test_following_a_run_that_does_not_exist_is_an_error(self):
        import pytest

        with pytest.raises(ValueError, match="not found"):
            service.run_task(mode="ask", alias="cld-deep", task="and then?", follow_up_to="nope0000")

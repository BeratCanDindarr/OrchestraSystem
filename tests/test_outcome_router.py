"""Unit tests for outcome-weighted routing (Phase 1 + Phase 2)."""
import pytest
from orchestra.router.outcome_router import OutcomeRouter
from orchestra.utils.outcome_recorder import OutcomeRecorder


class TestTokenization:
    def test_tokenize_basic(self):
        tokens = OutcomeRecorder.tokenize("Fix the routing bug in production")
        assert "fix" in tokens
        assert "routing" in tokens
        assert "bug" in tokens
        assert "production" in tokens
        # Stop words filtered out
        assert "the" not in tokens
        assert "in" not in tokens

    def test_tokenize_empty(self):
        tokens = OutcomeRecorder.tokenize("")
        assert tokens == set()

    def test_tokenize_nonstring(self):
        with pytest.raises(ValueError):
            OutcomeRecorder.tokenize(123)


class TestJaccardSimilarity:
    def test_identical_sets(self):
        sim = OutcomeRouter.jaccard_similarity({"fix", "bug"}, {"fix", "bug"})
        assert sim == 1.0

    def test_disjoint_sets(self):
        sim = OutcomeRouter.jaccard_similarity({"fix"}, {"routing"})
        assert sim == 0.0

    def test_partial_overlap(self):
        sim = OutcomeRouter.jaccard_similarity({"fix", "bug"}, {"fix", "routing"})
        assert 0.25 < sim < 0.75

    def test_none_inputs(self):
        with pytest.raises(ValueError):
            OutcomeRouter.jaccard_similarity(None, {"fix"})


class TestFindTopSimilar:
    def test_find_top_k(self):
        history = [
            {"task_tokens": ["fix", "bug"], "mode": "ask", "confidence": 0.5, "cost_usd": 0.1},
            {"task_tokens": ["fix", "bug", "routing"], "mode": "dual", "confidence": 0.9, "cost_usd": 0.2},
            {"task_tokens": ["implement", "feature"], "mode": "planned", "confidence": 0.8, "cost_usd": 0.3},
        ]
        similar = OutcomeRouter.find_top_similar({"fix", "bug"}, history, k=2)
        assert len(similar) <= 2
        assert similar[0]["mode"] in ("ask", "dual")

    def test_invalid_tokens_type(self):
        with pytest.raises(ValueError):
            OutcomeRouter.find_top_similar("not_a_set", [], k=10)


def _records(mode: str, status: str, tokens: list[str], count: int, cost: float = 0.01) -> list[dict]:
    return [{"mode": mode, "status": status, "task_tokens": tokens, "cost_usd": cost} for _ in range(count)]


_UNRELATED = _records("ask", "completed", ["weather", "tomorrow"], 60)
_SHADER = ["shader", "compile", "error", "unity"]


class TestSuggest:
    """The mode that worked on similar tasks; no opinion when the history does not say."""

    def test_no_opinion_before_there_is_enough_history(self):
        history = _records("dual", "completed", _SHADER, 10)
        assert OutcomeRouter.suggest("unity shader compile error", history, min_history=50) is None

    def test_the_mode_that_succeeded_on_similar_tasks_wins(self):
        history = _UNRELATED + _records("ask", "failed", _SHADER, 5) + _records("dual", "completed", _SHADER, 5)
        assert OutcomeRouter.suggest("unity shader compile error", history, min_history=50) == "dual"

    def test_history_about_other_things_gives_no_opinion(self):
        assert OutcomeRouter.suggest("unity shader compile error", _UNRELATED, min_history=50) is None

    def test_two_similar_tasks_are_not_evidence(self):
        history = _UNRELATED + _records("dual", "completed", _SHADER, 2)
        assert OutcomeRouter.suggest("unity shader compile error", history, min_history=50) is None

    def test_when_two_modes_do_equally_well_the_cheaper_one_wins(self):
        history = (_UNRELATED + _records("dual", "completed", _SHADER, 4, cost=0.50)
                   + _records("ask", "completed", _SHADER, 4, cost=0.02))
        assert OutcomeRouter.suggest("unity shader compile error", history, min_history=50) == "ask"

    def test_invalid_task_description(self):
        with pytest.raises(ValueError):
            OutcomeRouter.suggest("", [], min_history=1)


@pytest.fixture
def outcome_db(tmp_path, monkeypatch):
    monkeypatch.setattr("orchestra.config.db_path", lambda: tmp_path / "orchestra.db")
    return tmp_path


class TestHistory:
    """What finalize records is what the router reads."""

    @staticmethod
    def _run(task: str, status: str, **fields):
        from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus

        run = OrchestraRun(mode=fields.pop("mode", "dual"), task=task, status=RunStatus(status), **fields)
        run.agents = [AgentRun(alias="cdx-deep", provider="codex", model="m", status=AgentStatus.COMPLETED)]
        run.total_cost_usd = 0.2
        return run

    def test_a_recorded_outcome_can_be_read_back_with_its_tokens(self, outcome_db):
        from orchestra.router.outcome_router import append_outcome, load_history

        append_outcome(self._run("unity shader compile error", "completed"))

        (record,) = load_history()
        assert record["mode"] == "dual" and record["status"] == "completed"
        assert set(record["task_tokens"]) == OutcomeRecorder.tokenize("unity shader compile error")
        assert record["cost_usd"] == 0.2

    def test_steps_and_cancelled_runs_are_not_outcomes(self, outcome_db):
        from orchestra.router.outcome_router import append_outcome, load_history

        append_outcome(self._run("verify this", "completed", mode="ask", parent_run_id="parent01"))
        append_outcome(self._run("stopped by the user", "cancelled"))

        assert load_history() == []

    def test_an_outcomes_table_from_before_tokens_were_stored_still_works(self, outcome_db):
        from orchestra.router.outcome_router import append_outcome, load_history
        from orchestra.storage.db import get_db

        connection = get_db()
        with connection:
            connection.execute("CREATE TABLE outcomes (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, mode TEXT, "
                               "task_hash TEXT, aliases TEXT, avg_confidence REAL, total_cost_usd REAL, status TEXT, created_at REAL)")
            connection.execute("INSERT INTO outcomes (run_id, mode, status, created_at) VALUES ('old', 'ask', 'completed', 1.0)")
        connection.close()

        append_outcome(self._run("unity shader compile error", "completed"))

        assert [record["mode"] for record in load_history()] == ["dual"]     # the old row has no tokens to compare


class TestAutoRouting:
    """`auto` asks the history only when the owner switched that on."""

    @staticmethod
    def _auto(monkeypatch, *, enabled: bool) -> str:
        from unittest.mock import MagicMock, patch

        from orchestra import config, service

        history = _UNRELATED + _records("dual", "completed", _SHADER, 5)
        monkeypatch.setitem(config._cfg, "outcome_routing", {"enabled": enabled})
        monkeypatch.setitem(config._cfg, "cascade", {"enabled": False})
        fake_run = MagicMock()
        fake_run.to_manifest.return_value = {}
        decision = MagicMock(mode="ask", alias="cdx-fast")
        with patch("orchestra.router.outcome_router.load_history", return_value=history), \
             patch.object(service, "route_task", return_value=decision), \
             patch.object(service, "classify", return_value="cheap"), \
             patch.object(service, "run_ask", return_value=fake_run), \
             patch.object(service, "run_dual", return_value=fake_run), \
             patch("orchestra.live_status.install"):
            return service.run_task(mode="auto", task="unity shader compile error")["routed_mode"]

    def test_off_by_default_the_heuristic_decides(self, monkeypatch):
        assert self._auto(monkeypatch, enabled=False) == "ask"

    def test_switched_on_the_history_decides(self, monkeypatch):
        assert self._auto(monkeypatch, enabled=True) == "dual"


class TestOutcomeRecorder:
    def test_record_outcome_validation(self):
        with pytest.raises(ValueError):
            OutcomeRecorder.record_outcome("", "ask", 0.5, 0.1)

        with pytest.raises(ValueError):
            OutcomeRecorder.record_outcome("task", "invalid_mode", 0.5, 0.1)

        with pytest.raises(ValueError):
            OutcomeRecorder.record_outcome("task", "ask", 1.5, 0.1)  # confidence > 1.0

        with pytest.raises(ValueError):
            OutcomeRecorder.record_outcome("task", "ask", 0.5, -0.1)  # negative cost


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

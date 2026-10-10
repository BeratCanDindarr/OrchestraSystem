"""Structured code review: only findings whose quote is in the diff count."""
from __future__ import annotations

import json
from unittest.mock import patch

from orchestra.engine import code_review
from orchestra.models import AgentRun, AgentStatus

DIFF = """diff --git a/Assets/Timer.cs b/Assets/Timer.cs
--- a/Assets/Timer.cs
+++ b/Assets/Timer.cs
@@ -1,4 +1,6 @@
 public class Timer
 {
+    private float _remaining;
+    public void Tick(float dt) { _remaining -= dt; label.text = _remaining.ToString(); }
     public void Reset() { }
 }
"""
BAD_LINE = "public void Tick(float dt) { _remaining -= dt; label.text = _remaining.ToString(); }"


def _project(tmp_path):
    source = tmp_path / "Assets" / "Timer.cs"
    source.parent.mkdir(parents=True)
    source.write_text("public class Timer\n{\n    private float _remaining;\n    " + BAD_LINE + "\n    public void Reset() { }\n}\n")
    return tmp_path


def _finding(**over) -> dict:
    return {"severity": "high", "file": "Assets/Timer.cs", "line": 99, "quote": BAD_LINE,
            "problem": "allocates a string every frame", "fix": "update once per second", "confidence": 0.8, **over}


class TestDiffLines:
    def test_added_and_context_lines_are_kept_per_file(self):
        files = code_review.diff_lines(DIFF)

        assert list(files) == ["Assets/Timer.cs"]
        assert "private float _remaining;" in files["Assets/Timer.cs"]
        assert "public class Timer" in files["Assets/Timer.cs"]


class TestCheck:
    def test_a_finding_whose_quote_is_in_the_diff_is_kept_with_the_real_line(self, tmp_path):
        kept, discarded = code_review.check([_finding()], "cld-deep", code_review.diff_lines(DIFF), _project(tmp_path))

        assert discarded == []
        assert (kept[0].file, kept[0].line, kept[0].found_by) == ("Assets/Timer.cs", 4, ["cld-deep"])

    def test_an_invented_quote_is_discarded(self, tmp_path):
        made_up = _finding(quote="private static Timer _instance = null;")
        kept, discarded = code_review.check([made_up], "kimi-deep", code_review.diff_lines(DIFF), _project(tmp_path))

        assert kept == [] and discarded[0]["reason"] == "quote is not in the diff"

    def test_a_file_outside_the_diff_is_discarded(self, tmp_path):
        elsewhere = _finding(file="Assets/Other.cs")
        kept, discarded = code_review.check([elsewhere], "kimi-deep", code_review.diff_lines(DIFF), _project(tmp_path))

        assert kept == [] and discarded[0]["reason"] == "file is not in the diff"

    def test_a_bare_file_name_and_odd_spacing_still_match(self, tmp_path):
        loose = _finding(file="Timer.cs", quote="  public void Tick(float dt)   { _remaining -= dt; label.text = _remaining.ToString(); }")
        kept, _ = code_review.check([loose], "cld-deep", code_review.diff_lines(DIFF), _project(tmp_path))

        assert kept and kept[0].file == "Assets/Timer.cs"


class TestMerge:
    def test_two_reviewers_on_the_same_line_are_one_finding(self, tmp_path):
        files, root = code_review.diff_lines(DIFF), _project(tmp_path)
        first, _ = code_review.check([_finding(severity="medium", confidence=0.6)], "cld-deep", files, root)
        second, _ = code_review.check([_finding(severity="high", confidence=0.9)], "kimi-deep", files, root)

        merged = code_review.merge(first + second)

        assert len(merged) == 1
        assert merged[0].severity == "high" and merged[0].found_by == ["cld-deep", "kimi-deep"]
        assert merged[0].confidence == 0.9

    def test_a_repeated_statement_reported_at_either_copy_is_one_finding(self):
        def at(line, alias):
            return code_review.Finding("critical", "Pool.cs", line, "_free.Push(tile);", "pushed twice", found_by=[alias])

        merged = code_review.merge([at(22, "kimi-deep"), at(23, "cld-deep"), at(60, "cld-deep")])

        assert [(f.line, f.found_by) for f in merged] == [(22, ["cld-deep", "kimi-deep"]), (60, ["cld-deep"])]

    def test_the_most_severe_comes_first(self, tmp_path):
        files, root = code_review.diff_lines(DIFF), _project(tmp_path)
        low, _ = code_review.check([_finding(severity="low", quote="private float _remaining;")], "a", files, root)
        critical, _ = code_review.check([_finding(severity="critical")], "b", files, root)

        assert [f.severity for f in code_review.merge(low + critical)] == ["critical", "low"]


class TestParse:
    def test_json_inside_prose_or_fences_is_read(self):
        text = "Here you go:\n```json\n" + json.dumps({"findings": [_finding()]}) + "\n```"
        assert len(code_review.parse_findings(text)) == 1

    def test_an_answer_without_json_has_no_findings(self):
        assert code_review.parse_findings("Looks good to me.") == []


class TestRunReview:
    def test_a_review_reports_checked_findings_and_counts_the_discarded(self, tmp_path):
        answers = {
            "cld-deep": json.dumps({"findings": [_finding()]}),
            "kimi-deep": json.dumps({"findings": [_finding(severity="critical"), _finding(quote="int invented = 0; // not there")]}),
        }

        def fake(run, agents, **kwargs):
            for alias, prompt in agents:
                assert "[DIFF]" in prompt and "Pay particular attention to: allocations" in prompt
                agent = AgentRun(alias=alias, provider="test", model="test/model", status=AgentStatus.COMPLETED)
                agent.stdout_log = answers[alias]
                run.agents.append(agent)

        with patch("orchestra.engine.parallel.run_parallel", fake):
            run, findings, discarded = code_review.run_review(
                DIFF, focus="allocations", aliases=["cld-deep", "kimi-deep"], root=_project(tmp_path))

        assert len(findings) == 1 and findings[0].severity == "critical" and len(findings[0].found_by) == 2
        assert len(discarded) == 1
        assert "[CRITICAL] Assets/Timer.cs:4" in run.summary and "both reviewers" in run.summary
        assert "1 reported finding(s) were discarded" in run.summary
        assert run.mode == "review"

    def test_a_sound_change_says_so(self, tmp_path):
        def fake(run, agents, **kwargs):
            for alias, _ in agents:
                agent = AgentRun(alias=alias, provider="test", model="test/model", status=AgentStatus.COMPLETED)
                agent.stdout_log = '{"findings": []}'
                run.agents.append(agent)

        with patch("orchestra.engine.parallel.run_parallel", fake):
            run, findings, _ = code_review.run_review(DIFF, aliases=["cld-deep", "kimi-deep"], root=_project(tmp_path))

        assert findings == [] and run.summary.startswith("No defects found")


class TestService:
    def test_no_changes_is_an_error_not_an_empty_review(self):
        import pytest

        from orchestra import service

        with patch.object(code_review, "collect_diff", return_value=""), pytest.raises(ValueError, match="Nothing to review"):
            service.review_changes()

    def test_a_given_diff_is_reviewed_and_findings_come_back_as_data(self, tmp_path, monkeypatch):
        from orchestra import config, service

        monkeypatch.setattr(config, "project_root", lambda: _project(tmp_path))

        def fake(run, agents, **kwargs):
            for alias, _ in agents:
                agent = AgentRun(alias=alias, provider="test", model="test/model", status=AgentStatus.COMPLETED)
                agent.stdout_log = json.dumps({"findings": [_finding()]})
                run.agents.append(agent)

        with patch("orchestra.engine.parallel.run_parallel", fake):
            result = service.review_changes(diff=DIFF)

        assert result["discarded"] == 0
        assert result["findings"][0]["file"] == "Assets/Timer.cs" and result["findings"][0]["line"] == 4
        assert result["run"]["mode"] == "review"

    def test_the_tool_is_registered(self):
        import orchestra.mcp_server  # noqa: F401
        from orchestra.engine.mcp_registry import REGISTRY

        tool = {t["name"]: t for t in REGISTRY.list_tools()}["orchestra_review_changes"]
        assert tool["inputSchema"]["properties"]["paths"]["type"] == "array"


class TestWhoReallyReviewed:
    def test_a_fallback_is_named_and_agreement_between_the_same_model_is_flagged(self, tmp_path):
        def fake(run, agents, **kwargs):
            for alias, _ in agents:
                # kimi-deep could not answer: Claude answered under its alias
                agent = AgentRun(alias=alias, provider="claude", model="claude/opus", status=AgentStatus.COMPLETED)
                agent.stdout_log = json.dumps({"findings": [_finding()]})
                run.agents.append(agent)

        with patch("orchestra.engine.parallel.run_parallel", fake):
            run, findings, _ = code_review.run_review(DIFF, aliases=["cld-deep", "kimi-deep"], root=_project(tmp_path))

        assert findings[0].found_by == ["claude/opus (in place of kimi-deep)", "cld-deep"]
        assert "Both reviews came from the same model (claude/opus)" in run.summary

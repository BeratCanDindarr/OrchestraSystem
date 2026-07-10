"""Tests for shared MCP helpers (previously in mcp_server, now in mcp_registry).

Removed:
- test_resource_contents_*: _resource_contents was removed along with the
  fake resources capability.
"""
from orchestra.engine.mcp_registry import _structured_run_result, _tool_success, _tool_success_run


# Restored from commit 3bea394, renamed because _structured_run_result no
# longer exposes primary_source/answer (those fields were removed when the
# function was refactored to return a flat summary shape).  The original
# intent — verifying that the helper correctly handles a multi-agent run
# payload — is preserved by asserting on every field of the current shape.
def test_structured_run_result_returns_run_fields():
    payload = {
        "run": {
            "run_id": "run-1",
            "mode": "dual",
            "status": "completed",
            "task": "do something",
            "summary": "all good",
            "avg_confidence": 0.5,
            "total_cost_usd": 0.0,
            "agents": [
                {"alias": "verifier", "status": "completed", "stdout_log": "Verifier output"},
                {"alias": "cdx-deep", "status": "completed", "stdout_log": "Candidate output"},
            ],
        }
    }

    result = _structured_run_result(payload)

    assert result["run_id"] == "run-1"
    assert result["status"] == "completed"
    assert result["mode"] == "dual"
    assert result["task"] == "do something"
    assert result["summary"] == "all good"
    assert result["agent_count"] == 2


def test_tool_success_scrubs_pii_in_payload():
    payload = {"logs": {"agent": "Contact me at alice@example.com"}}
    result = _tool_success(payload)
    text = result["content"][0]["text"]
    assert "alice@example.com" not in text
    assert "[REDACTED:email]" in text
    assert "[REDACTED:email]" in result["structuredContent"]["logs"]["agent"]
    assert "alice@example.com" not in result["structuredContent"]["logs"]["agent"]


def test_tool_success_run_scrubs_pii_in_run():
    payload = {
        "run": {
            "run_id": "run-1",
            "mode": "dual",
            "status": "completed",
            "summary": "",
            "avg_confidence": 0.5,
            "total_cost_usd": 0.0,
            "agents": [
                {
                    "alias": "verifier",
                    "status": "completed",
                    "stdout_log": "Verifier output with sk_test_1234567890abcdefghij",
                },
                {
                    "alias": "cdx-deep",
                    "status": "completed",
                    "stdout_log": "Candidate output with bob@example.com",
                },
            ],
        }
    }
    result = _tool_success_run(payload)
    text = result["content"][0]["text"]
    assert "sk_test_1234567890abcdefghij" not in text
    assert "bob@example.com" not in text
    assert "[REDACTED:api_key_sk]" in text
    assert "[REDACTED:email]" in text


# ---------------------------------------------------------------------------
# Task 27 (Phase 3a): _structured_run_result approval state fields
# ---------------------------------------------------------------------------

def test_structured_run_result_waiting_approval():
    """Run with status 'waiting_approval' exposes approval_state and waiting_approval=True."""
    payload = {
        "run": {
            "run_id": "run-2",
            "mode": "critical",
            "status": "waiting_approval",
            "task": "analyze something",
            "summary": "round 1 done",
            "agents": [{"alias": "cdx-fast", "status": "completed"}],
            "approval_state": "waiting_approval",
        }
    }
    result = _structured_run_result(payload)

    assert result["waiting_approval"] is True
    assert result["approval_state"] == "waiting_approval"
    # Existing fields must still be present
    assert result["run_id"] == "run-2"
    assert result["status"] == "waiting_approval"
    assert result["mode"] == "critical"
    assert result["task"] == "analyze something"
    assert result["summary"] == "round 1 done"
    assert result["agent_count"] == 1


def test_structured_run_result_completed_no_approval():
    """A completed run has waiting_approval=False and approval_state defaults to 'not_required'."""
    payload = {
        "run": {
            "run_id": "run-3",
            "mode": "dual",
            "status": "completed",
            "task": "do stuff",
            "summary": "done",
            "agents": [],
        }
    }
    result = _structured_run_result(payload)

    assert result["waiting_approval"] is False
    assert result["approval_state"] == "not_required"

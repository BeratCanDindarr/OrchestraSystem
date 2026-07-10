"""Tests for shared MCP helpers (previously in mcp_server, now in mcp_registry).

Removed:
- test_structured_run_result_prefers_candidate_over_verifier: the
  _structured_run_result helper no longer exposes primary_source/answer fields.
- test_resource_contents_*: _resource_contents was removed along with the
  fake resources capability.
"""
from orchestra.engine.mcp_registry import _structured_run_result, _tool_success, _tool_success_run


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

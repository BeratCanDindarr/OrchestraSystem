from orchestra.mcp_server import _structured_run_result, _tool_success, _tool_success_run, _resource_contents


def test_structured_run_result_prefers_candidate_over_verifier():
    payload = {
        "run": {
            "run_id": "run-1",
            "mode": "dual",
            "status": "completed",
            "summary": "",
            "avg_confidence": 0.5,
            "total_cost_usd": 0.0,
            "agents": [
                {"alias": "verifier", "status": "completed", "stdout_log": "Verifier output"},
                {"alias": "cdx-deep", "status": "completed", "stdout_log": "Candidate output"},
            ],
        }
    }

    result = _structured_run_result(payload)

    assert result["primary_source"] == "cdx-deep"
    assert result["answer"] == "Candidate output"


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
                {"alias": "verifier", "status": "completed", "stdout_log": "Verifier output with sk_test_1234567890abcdefghij"},
                {"alias": "cdx-deep", "status": "completed", "stdout_log": "Candidate output with bob@example.com"},
            ],
        }
    }
    result = _tool_success_run(payload)
    text = result["content"][0]["text"]
    assert "sk_test_1234567890abcdefghij" not in text
    assert "bob@example.com" not in text
    assert "[REDACTED:api_key_sk]" in text
    assert "[REDACTED:email]" in text


def test_resource_contents_scrubs_pii_in_text():
    result = _resource_contents("orchestra://test", "text/plain", "Key: sk-abcdefghij1234567890xyz")
    assert "sk-abcdefghij1234567890xyz" not in result["contents"][0]["text"]
    assert "[REDACTED:api_key_sk]" in result["contents"][0]["text"]


def test_resource_contents_scrubs_pii_in_dict():
    result = _resource_contents("orchestra://test", "application/json", {"key": "sk-abcdefghij1234567890xyz"})
    assert "sk-abcdefghij1234567890xyz" not in result["contents"][0]["text"]
    assert "[REDACTED:api_key_sk]" in result["contents"][0]["text"]

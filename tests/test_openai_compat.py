import json
from unittest.mock import patch
from orchestra.openai_compat import (
    messages_to_prompt, to_response, to_sse_chunk, sse_done, list_models,
)
import orchestra.openai_compat as oc


def test_messages_to_prompt_system_prefix_and_history():
    msgs = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again?"},
    ]
    out = messages_to_prompt(msgs)
    assert out == (
        "System: be terse\n\n"
        "User: hi\n\n"
        "Assistant: hello\n\n"
        "User: again?\n\n"
        "Assistant:"
    )


def test_messages_to_prompt_no_system():
    out = messages_to_prompt([{"role": "user", "content": "hi"}])
    assert out == "User: hi\n\nAssistant:"


def test_to_response_shape():
    r = to_response("oll-fast", "hello", prompt_tokens=3, completion_tokens=1, _id="chatcmpl-x", _created=111)
    assert r["object"] == "chat.completion"
    assert r["id"] == "chatcmpl-x"
    assert r["created"] == 111
    assert r["model"] == "oll-fast"
    assert r["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
    assert r["choices"][0]["finish_reason"] == "stop"
    assert r["usage"] == {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}


def test_to_sse_chunk_delta():
    line = to_sse_chunk("oll-fast", "hel", _id="chatcmpl-x", _created=111)
    assert line.startswith("data: ")
    assert line.endswith("\n\n")
    obj = json.loads(line[len("data: "):].strip())
    assert obj["object"] == "chat.completion.chunk"
    assert obj["choices"][0]["delta"] == {"content": "hel"}
    assert obj["choices"][0]["finish_reason"] is None


def test_to_sse_chunk_final():
    line = to_sse_chunk("oll-fast", "", finish_reason="stop", _id="chatcmpl-x", _created=111)
    obj = json.loads(line[len("data: "):].strip())
    assert obj["choices"][0]["finish_reason"] == "stop"
    assert obj["choices"][0]["delta"] == {}


def test_sse_done():
    assert sse_done() == "data: [DONE]\n\n"


def test_list_models_includes_aliases_and_modes():
    with patch("orchestra.openai_compat.available_alias_ids", return_value=["oll-fast"]):
        models = list_models()
    ids = [m["id"] for m in models]
    assert "oll-fast" in ids            # available single alias
    assert "gmn-pro" not in ids         # unavailable, skipped
    assert "orchestra-dual" in ids
    assert "orchestra-critical" in ids
    assert "orchestra-auto" in ids
    assert all(m["object"] == "model" for m in models)

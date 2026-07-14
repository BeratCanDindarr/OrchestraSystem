from orchestra.anthropic_compat import request_to_prompt, resolve_anthropic_model, approx_tokens


def test_request_to_prompt_system_str_and_history():
    system = "be terse"
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
    ]
    assert request_to_prompt(system, messages) == (
        "System: be terse\n\nUser: hi\n\nAssistant: hello\n\nUser: bye\n\nAssistant:"
    )


def test_request_to_prompt_system_blocks_and_content_blocks():
    system = [{"type": "text", "text": "sys one"}, {"type": "text", "text": "sys two"}]
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "look "},
        {"type": "image", "source": {}},          # ignored
        {"type": "text", "text": "here"},
    ]}]
    assert request_to_prompt(system, messages) == (
        "System: sys one\n\nsys two\n\nUser: look here\n\nAssistant:"
    )


def test_request_to_prompt_no_system():
    assert request_to_prompt(None, [{"role": "user", "content": "yo"}]) == "User: yo\n\nAssistant:"


def test_resolve_anthropic_model_known_alias_and_mode():
    assert resolve_anthropic_model("cdx-deep") == "cdx-deep"
    assert resolve_anthropic_model("cdx-deep@windows") == "cdx-deep@windows"
    assert resolve_anthropic_model("orchestra-dual") == "orchestra-dual"


def test_resolve_anthropic_model_unknown_falls_back_to_default():
    assert resolve_anthropic_model("claude-3-5-haiku") == "oll-fast"
    assert resolve_anthropic_model("orchestra-bogus") == "oll-fast"


def test_approx_tokens():
    assert approx_tokens("") == 0
    assert approx_tokens("abcdefgh") == 2


import json as _json
from orchestra.anthropic_compat import (
    to_message_response, new_message_id,
    sse_message_start, sse_content_block_start, sse_text_delta,
    sse_content_block_stop, sse_message_delta, sse_message_stop,
)


def _parse_sse(line):
    assert line.endswith("\n\n")
    event_line, data_line = line.rstrip("\n").split("\n")
    assert event_line.startswith("event: ")
    assert data_line.startswith("data: ")
    return event_line[len("event: "):], _json.loads(data_line[len("data: "):])


def test_to_message_response_shape():
    r = to_message_response("oll-fast", "hi there", input_tokens=3, output_tokens=2, _id="msg_x")
    assert r["id"] == "msg_x" and r["type"] == "message" and r["role"] == "assistant"
    assert r["model"] == "oll-fast"
    assert r["content"] == [{"type": "text", "text": "hi there"}]
    assert r["stop_reason"] == "end_turn" and r["stop_sequence"] is None
    assert r["usage"] == {"input_tokens": 3, "output_tokens": 2}


def test_new_message_id_prefix():
    assert new_message_id().startswith("msg_")


def test_sse_event_sequence_shapes():
    ev, data = _parse_sse(sse_message_start("oll-fast", "msg_1", input_tokens=5))
    assert ev == "message_start"
    assert data["message"]["id"] == "msg_1" and data["message"]["role"] == "assistant"
    assert data["message"]["usage"] == {"input_tokens": 5, "output_tokens": 0}

    ev, data = _parse_sse(sse_content_block_start())
    assert ev == "content_block_start" and data["content_block"] == {"type": "text", "text": ""}

    ev, data = _parse_sse(sse_text_delta("hello"))
    assert ev == "content_block_delta"
    assert data["delta"] == {"type": "text_delta", "text": "hello"}

    ev, data = _parse_sse(sse_content_block_stop())
    assert ev == "content_block_stop" and data["index"] == 0

    ev, data = _parse_sse(sse_message_delta(output_tokens=7))
    assert ev == "message_delta"
    assert data["delta"]["stop_reason"] == "end_turn"
    assert data["usage"] == {"output_tokens": 7}

    ev, data = _parse_sse(sse_message_stop())
    assert ev == "message_stop"

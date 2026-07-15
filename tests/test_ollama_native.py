from orchestra.providers.ollama_native import (
    to_ollama_messages, to_ollama_tools, from_ollama_response,
)


def test_to_ollama_messages_system_and_text():
    msgs = to_ollama_messages("be terse", [{"role": "user", "content": "hi"}])
    assert msgs == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]


def test_to_ollama_messages_tool_use_becomes_tool_calls():
    messages = [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "ok"},
            {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"cmd": "ls"}},
        ]},
    ]
    out = to_ollama_messages(None, messages)
    assert out[0] == {"role": "user", "content": "list files"}
    assert out[1]["role"] == "assistant" and out[1]["content"] == "ok"
    assert out[1]["tool_calls"] == [{"function": {"name": "bash", "arguments": {"cmd": "ls"}}}]


def test_to_ollama_messages_tool_result_becomes_role_tool():
    messages = [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file1\nfile2"},
            {"type": "text", "text": "now summarize"},
        ]},
    ]
    out = to_ollama_messages(None, messages)
    # tool results come FIRST (they answer the previous assistant turn), then user text
    assert out[0] == {"role": "tool", "content": "file1\nfile2"}
    assert out[1] == {"role": "user", "content": "now summarize"}


def test_to_ollama_messages_tool_result_block_content():
    messages = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t", "content": [{"type": "text", "text": "out"}]},
    ]}]
    assert to_ollama_messages(None, messages) == [{"role": "tool", "content": "out"}]


def test_to_ollama_tools_schema_rename():
    tools = [{"name": "bash", "description": "run", "input_schema": {"type": "object"}}]
    assert to_ollama_tools(tools) == [{
        "type": "function",
        "function": {"name": "bash", "description": "run", "parameters": {"type": "object"}},
    }]


def test_from_ollama_response_text_only():
    blocks, stop = from_ollama_response({"content": "hello"})
    assert blocks == [{"type": "text", "text": "hello"}]
    assert stop == "end_turn"


def test_from_ollama_response_tool_calls():
    blocks, stop = from_ollama_response({
        "content": "",
        "tool_calls": [{"function": {"name": "bash", "arguments": {"cmd": "ls"}}}],
    })
    assert stop == "tool_use"
    assert len(blocks) == 1
    b = blocks[0]
    assert b["type"] == "tool_use" and b["name"] == "bash" and b["input"] == {"cmd": "ls"}
    assert b["id"].startswith("toolu_")


def test_from_ollama_response_string_arguments_parsed():
    blocks, _ = from_ollama_response({
        "content": "", "tool_calls": [{"function": {"name": "f", "arguments": "{\"a\": 1}"}}],
    })
    assert blocks[0]["input"] == {"a": 1}

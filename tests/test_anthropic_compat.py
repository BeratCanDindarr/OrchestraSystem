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

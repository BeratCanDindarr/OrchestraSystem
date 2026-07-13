from orchestra.chat.history import render_history


def test_single_user_turn():
    h = [{"role": "user", "content": "hello"}]
    assert render_history(h) == "User: hello\n\nAssistant:"


def test_multi_turn_prompt_ends_with_last_user_and_assistant_cue():
    h = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
        {"role": "user", "content": "how are you?"},
    ]
    out = render_history(h)
    assert out == (
        "User: hi\n\n"
        "Assistant: hello there\n\n"
        "User: how are you?\n\n"
        "Assistant:"
    )


def test_empty_history_is_empty_string():
    assert render_history([]) == ""

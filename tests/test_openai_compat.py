import pytest
from orchestra.openai_compat import messages_to_prompt, resolve_target


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


def test_resolve_target_alias():
    assert resolve_target("oll-fast") == ("alias", "oll-fast")
    assert resolve_target("cdx-deep@windows") == ("alias", "cdx-deep@windows")


def test_resolve_target_modes():
    assert resolve_target("orchestra-dual") == ("mode", "dual")
    assert resolve_target("orchestra-critical") == ("mode", "critical")
    assert resolve_target("orchestra-auto") == ("mode", "auto")


def test_resolve_target_unknown_mode_raises():
    with pytest.raises(ValueError):
        resolve_target("orchestra-bogus")

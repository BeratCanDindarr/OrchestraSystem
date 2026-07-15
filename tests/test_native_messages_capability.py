import pytest


def _concrete_base_provider():
    from orchestra.providers.base import BaseProvider

    class _P(BaseProvider):
        name = "stub"
        def is_available(self): return True
        def model_label(self, e): return f"stub/{e}"
        def build_command(self, p, e): return ["echo", p]
    return _P()


def test_base_provider_not_native_capable_by_default():
    p = _concrete_base_provider()
    assert p.supports_native_messages is False


def test_base_native_messages_raises_not_implemented():
    p = _concrete_base_provider()
    with pytest.raises(NotImplementedError):
        p.native_messages({"messages": [{"role": "user", "content": "hi"}]})

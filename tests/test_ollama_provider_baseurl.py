from unittest.mock import patch, MagicMock
from orchestra.providers.ollama import OllamaProvider


def test_default_base_url_uses_module_default():
    p = OllamaProvider()
    with patch("orchestra.providers.ollama._ollama_base", return_value="http://localhost:11434"):
        assert p._base_url() == "http://localhost:11434"


def test_instance_base_url_overrides_default():
    p = OllamaProvider("http://192.168.1.42:11434")
    assert p._base_url() == "http://192.168.1.42:11434"


def test_is_available_pings_instance_base_url():
    p = OllamaProvider("http://192.168.1.42:11434")
    captured = {}

    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _Resp()

    with patch("orchestra.providers.ollama.urllib.request.urlopen", _fake_urlopen):
        assert p.is_available() is True
    assert captured["url"] == "http://192.168.1.42:11434/api/tags"

import json
from unittest.mock import patch
import pytest
from orchestra.providers import ollama_backends as ob
from orchestra.providers.ollama_agent import OllamaAgentProvider


def _fake_tags_response(models):
    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({"models": models}).encode()
    return _Resp()


def setup_function(_):
    ob.reset_cache()


def test_list_backends_defaults_to_macbook_localhost():
    with patch("orchestra.config.ollama_backends_config", return_value={}):
        assert ob.list_backends() == {"macbook": "http://localhost:11434"}


def test_list_backends_reads_configured():
    cfg = {
        "macbook": {"base_url": "http://localhost:11434"},
        "windows-gpu": {"base_url": "http://192.168.1.42:11434"},
    }
    with patch("orchestra.config.ollama_backends_config", return_value=cfg):
        assert ob.list_backends() == {
            "macbook": "http://localhost:11434",
            "windows-gpu": "http://192.168.1.42:11434",
        }


def test_get_backend_returns_provider_with_right_base_url():
    cfg = {"windows-gpu": {"base_url": "http://192.168.1.42:11434"}}
    with patch("orchestra.config.ollama_backends_config", return_value=cfg):
        p = ob.get_backend("windows-gpu")
        assert isinstance(p, OllamaAgentProvider)
        assert p._http._base_url() == "http://192.168.1.42:11434"
        assert ob.get_backend("windows-gpu") is p  # cached, same instance


def test_get_backend_unknown_raises_valueerror_listing_known():
    cfg = {"windows-gpu": {"base_url": "http://192.168.1.42:11434"}}
    with patch("orchestra.config.ollama_backends_config", return_value=cfg):
        with pytest.raises(ValueError) as exc:
            ob.get_backend("nope")
        assert "windows-gpu" in str(exc.value)


def test_discover_all_lists_models_per_backend():
    ob.reset_cache()
    ob.reset_discovery_cache()
    cfg = {
        "macbook": {"base_url": "http://localhost:11434"},
        "windows-gpu": {"base_url": "http://192.168.1.42:11434"},
    }
    def _fake_urlopen(req, timeout=None):
        if "192.168.1.42" in req.full_url:
            return _fake_tags_response([{"name": "llama3:70b", "size": 40_000_000_000}])
        return _fake_tags_response([{"name": "qwen2.5:7b", "size": 4_700_000_000}])

    with patch("orchestra.config.ollama_backends_config", return_value=cfg), \
         patch("orchestra.providers.ollama_backends.urllib.request.urlopen", _fake_urlopen):
        result = ob.discover_all(refresh=True)

    assert result["windows-gpu"]["online"] is True
    assert result["windows-gpu"]["models"][0]["name"] == "llama3:70b"
    assert result["macbook"]["models"][0]["name"] == "qwen2.5:7b"


def test_discover_all_marks_offline_backend_without_failing_others():
    ob.reset_cache()
    ob.reset_discovery_cache()
    cfg = {
        "macbook": {"base_url": "http://localhost:11434"},
        "windows-gpu": {"base_url": "http://192.168.1.42:11434"},
    }
    def _fake_urlopen(req, timeout=None):
        if "192.168.1.42" in req.full_url:
            raise OSError("connection refused")
        return _fake_tags_response([{"name": "qwen2.5:7b", "size": 4_700_000_000}])

    with patch("orchestra.config.ollama_backends_config", return_value=cfg), \
         patch("orchestra.providers.ollama_backends.urllib.request.urlopen", _fake_urlopen):
        result = ob.discover_all(refresh=True)

    assert result["windows-gpu"]["online"] is False
    assert result["windows-gpu"]["models"] == []
    assert result["macbook"]["online"] is True
    assert result["macbook"]["models"][0]["name"] == "qwen2.5:7b"


def test_discover_all_uses_ttl_cache_second_call_skips_network():
    """A second discover_all(refresh=False) within TTL must NOT hit the network."""
    from unittest.mock import MagicMock
    ob.reset_cache()
    ob.reset_discovery_cache()
    cfg = {"macbook": {"base_url": "http://localhost:11434"}}

    mock_urlopen = MagicMock(
        return_value=_fake_tags_response([{"name": "qwen2.5:7b", "size": 4_700_000_000}])
    )
    with patch("orchestra.config.ollama_backends_config", return_value=cfg), \
         patch("orchestra.providers.ollama_backends.urllib.request.urlopen", mock_urlopen):
        first = ob.discover_all(refresh=True)      # populates cache, 1 network call
        second = ob.discover_all(refresh=False)     # served from cache, no new call

    assert mock_urlopen.call_count == 1, "second call within TTL should be cache-served"
    assert second is first  # same cached object returned

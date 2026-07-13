from unittest.mock import patch
import pytest
from orchestra.providers import ollama_backends as ob
from orchestra.providers.ollama import OllamaProvider


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
        assert isinstance(p, OllamaProvider)
        assert p._base_url() == "http://192.168.1.42:11434"
        assert ob.get_backend("windows-gpu") is p  # cached, same instance


def test_get_backend_unknown_raises_valueerror_listing_known():
    cfg = {"windows-gpu": {"base_url": "http://192.168.1.42:11434"}}
    with patch("orchestra.config.ollama_backends_config", return_value=cfg):
        with pytest.raises(ValueError) as exc:
            ob.get_backend("nope")
        assert "windows-gpu" in str(exc.value)

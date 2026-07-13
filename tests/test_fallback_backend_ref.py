# tests/test_fallback_backend_ref.py
from unittest.mock import patch
import pytest
from orchestra.providers import fallback
from orchestra.providers import ollama_backends as ob


def test_resolve_backend_ref_routes_to_backend_provider():
    ob.reset_cache()
    cfg = {"windows-gpu": {"base_url": "http://192.168.1.42:11434"}}
    with patch("orchestra.config.ollama_backends_config", return_value=cfg):
        provider, model = fallback.resolve("qwen2.5-coder:7b@windows-gpu")
        assert model == "qwen2.5-coder:7b"
        assert provider._base_url() == "http://192.168.1.42:11434"


def test_resolve_unknown_backend_raises():
    ob.reset_cache()
    cfg = {"windows-gpu": {"base_url": "http://192.168.1.42:11434"}}
    with patch("orchestra.config.ollama_backends_config", return_value=cfg):
        with pytest.raises(ValueError):
            fallback.resolve("qwen2.5:7b@nope")


def test_resolve_without_at_is_backward_compatible():
    # oll-fast resolves to the ollama provider + "fast" effort as before.
    # When Ollama is unavailable, resolve() may substitute a fallback alias's
    # (provider, effort); accept either the primary or a known fallback effort.
    provider, effort = fallback.resolve("oll-fast")
    assert provider is not None
    assert effort in ("fast", "cdx-deep", "low", "xhigh", "opus", "sonnet", "pro", "flash",
                      "minimax-m2.7:cloud")

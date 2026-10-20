# tests/test_fallback_node_ref.py
from unittest.mock import patch
import pytest
from orchestra.providers import fallback
from orchestra import remote_nodes as rn
from orchestra.providers import ollama_backends as ob
from orchestra.providers.remote_node import RemoteNodeProvider


def test_alias_at_remote_node_routes_to_remote_provider():
    rn.reset_cache()
    node_cfg = {"windows": {"url": "http://192.168.1.42:7070", "token": "s"}}
    with patch("orchestra.config.remote_nodes_config", return_value=node_cfg):
        provider, alias = fallback.resolve("cdx-deep@windows")
        assert isinstance(provider, RemoteNodeProvider)
        assert alias == "cdx-deep"
        assert provider.model_label(alias) == "remote/windows/cdx-deep"


def test_remote_node_wins_over_ollama_backend_same_name():
    rn.reset_cache(); ob.reset_cache()
    node_cfg = {"windows": {"url": "http://192.168.1.42:7070"}}
    backend_cfg = {"windows": {"base_url": "http://192.168.1.42:11434"}}
    with patch("orchestra.config.remote_nodes_config", return_value=node_cfg), \
         patch("orchestra.config.ollama_backends_config", return_value=backend_cfg):
        provider, alias = fallback.resolve("cdx-deep@windows")
        assert isinstance(provider, RemoteNodeProvider)  # remote wins


def test_at_backend_still_routes_to_ollama_when_not_a_node():
    rn.reset_cache(); ob.reset_cache()
    backend_cfg = {"windows-gpu": {"base_url": "http://192.168.1.42:11434"}}
    with patch("orchestra.config.remote_nodes_config", return_value={}), \
         patch("orchestra.config.ollama_backends_config", return_value=backend_cfg):
        from orchestra.providers.ollama_agent import OllamaAgentProvider
        provider, model = fallback.resolve("qwen2.5:7b@windows-gpu")
        assert isinstance(provider, OllamaAgentProvider)
        assert model == "qwen2.5:7b"


def test_no_at_is_backward_compatible():
    provider, effort = fallback.resolve("oll-fast")
    assert provider is not None

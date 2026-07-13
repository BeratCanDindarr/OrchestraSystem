import json
from unittest.mock import patch
import pytest
from orchestra import remote_nodes as rn
from orchestra.providers.remote_node import RemoteNodeProvider


def setup_function(_):
    rn.reset_cache()


def test_list_nodes_empty_when_unconfigured():
    with patch("orchestra.config.remote_nodes_config", return_value={}):
        assert rn.list_nodes() == {}


def test_list_nodes_parses_url_and_token():
    cfg = {"windows": {"url": "http://192.168.1.42:7070", "token": "secret"}}
    with patch("orchestra.config.remote_nodes_config", return_value=cfg):
        assert rn.list_nodes() == {"windows": {"url": "http://192.168.1.42:7070", "token": "secret"}}


def test_list_nodes_skips_entry_without_url():
    cfg = {"bad": {"token": "x"}, "windows": {"url": "http://x:7070"}}
    with patch("orchestra.config.remote_nodes_config", return_value=cfg):
        assert list(rn.list_nodes()) == ["windows"]


def test_get_node_returns_cached_provider():
    cfg = {"windows": {"url": "http://192.168.1.42:7070", "token": "secret"}}
    with patch("orchestra.config.remote_nodes_config", return_value=cfg):
        p = rn.get_node("windows")
        assert isinstance(p, RemoteNodeProvider)
        assert p.model_label("cdx-deep") == "remote/windows/cdx-deep"
        assert rn.get_node("windows") is p


def test_get_node_unknown_raises_listing_known():
    cfg = {"windows": {"url": "http://x:7070"}}
    with patch("orchestra.config.remote_nodes_config", return_value=cfg):
        with pytest.raises(ValueError) as exc:
            rn.get_node("nope")
        assert "windows" in str(exc.value)


class _HealthResp:
    def __init__(self, host):
        self.status = 200
        self._b = json.dumps({"status": "ok", "host": host}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._b


def test_scan_subnet_finds_only_responders():
    def _fake_urlopen(req, timeout=None):
        if "192.168.1.42" in req.full_url:
            return _HealthResp("winbox")
        raise OSError("no route")

    with patch("orchestra.remote_nodes.urllib.request.urlopen", _fake_urlopen):
        found = rn.scan_subnet("192.168.1", token="s", port=7070)

    ips = [f["ip"] for f in found]
    assert "192.168.1.42" in ips
    assert len(found) == 1
    assert found[0]["host"] == "winbox"


def test_default_subnet_prefix_shape():
    with patch("orchestra.remote_nodes._primary_ip", return_value="192.168.1.7"):
        assert rn.default_subnet_prefix() == "192.168.1"

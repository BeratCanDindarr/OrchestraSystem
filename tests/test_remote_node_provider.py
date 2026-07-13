import json
from unittest.mock import patch
from orchestra.providers.remote_node import RemoteNodeProvider


class _Resp:
    def __init__(self, status, body):
        self.status = status
        self._body = json.dumps(body).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return self._body


def test_native_run_posts_alias_and_task_with_token():
    p = RemoteNodeProvider("windows", "http://192.168.1.42:7070", token="secret")
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["auth"] = req.headers.get("Authorization")
        captured["body"] = json.loads(req.data)
        return _Resp(200, {"stdout": "hello from windows", "status": "completed"})

    with patch("orchestra.providers.remote_node.urllib.request.urlopen", _fake_urlopen):
        out, rc = p.native_run("do it", "cdx-deep")

    assert out == "hello from windows"
    assert rc == 0
    assert captured["url"] == "http://192.168.1.42:7070/run"
    assert captured["auth"] == "Bearer secret"
    assert captured["body"] == {"alias": "cdx-deep", "task": "do it"}


def test_native_run_non_200_is_error():
    p = RemoteNodeProvider("windows", "http://192.168.1.42:7070", token="secret")

    def _fake_urlopen(req, timeout=None):
        return _Resp(401, {"error": "Unauthorized"})

    with patch("orchestra.providers.remote_node.urllib.request.urlopen", _fake_urlopen):
        out, rc = p.native_run("do it", "cdx-deep")
    assert rc == 1
    assert "Unauthorized" in out or "401" in out


def test_model_label_shows_node_and_alias():
    p = RemoteNodeProvider("windows", "http://x:7070")
    assert p.model_label("cdx-deep") == "remote/windows/cdx-deep"


def test_is_available_hits_health():
    p = RemoteNodeProvider("windows", "http://192.168.1.42:7070")
    seen = {}

    def _fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        return _Resp(200, {"status": "ok"})

    with patch("orchestra.providers.remote_node.urllib.request.urlopen", _fake_urlopen):
        assert p.is_available() is True
    assert seen["url"] == "http://192.168.1.42:7070/health"


def test_async_run_delegates_to_native_run():
    """The async path (parallel/dual executor) must use native_run, not build_command."""
    import asyncio
    p = RemoteNodeProvider("windows", "http://192.168.1.42:7070", token="secret")

    def _fake_urlopen(req, timeout=None):
        return _Resp(200, {"stdout": "async ok", "status": "completed"})

    with patch("orchestra.providers.remote_node.urllib.request.urlopen", _fake_urlopen):
        out, rc = asyncio.run(p.async_run("do it", "cdx-deep"))
    assert out == "async ok"
    assert rc == 0

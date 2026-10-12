"""A web page the user happens to visit must not be able to drive the local servers.

The gateway, the dashboard and the WebSocket daemon all run agents. A browser can reach
127.0.0.1 from any site, so each of them refuses requests that come from another origin
(or, when bound to loopback, that name another host: DNS rebinding).
"""
from __future__ import annotations

import io
import json
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets
from aiohttp.test_utils import TestClient, TestServer
from typer.testing import CliRunner

from orchestra import claude_launcher, config, gateway
from orchestra import remote_nodes as rn
from orchestra import server as srv
from orchestra.monitoring import dashboard_app
from orchestra.utils import browser_guard

LOOPBACK = browser_guard.LOOPBACK_HOSTS
EVIL = "https://evil.example"


# ── the rule itself ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("host, origin, allowed_hosts, expected", [
    ("127.0.0.1:7070", None, LOOPBACK, True),                        # curl, an SDK
    ("localhost:8080", "http://localhost:8080", LOOPBACK, True),     # the dashboard's own page
    ("[::1]:7070", None, LOOPBACK, True),
    ("127.0.0.1:7070", EVIL, LOOPBACK, False),                       # another site
    ("127.0.0.1:7070", "null", LOOPBACK, False),                     # sandboxed frame, file://
    ("evil.example:7070", "http://evil.example:7070", LOOPBACK, False),  # DNS rebinding
    ("evil.example:7070", None, LOOPBACK, False),
    ("192.168.1.5:7070", None, None, True),                          # LAN bind: the token decides
    ("192.168.1.5:7070", EVIL, None, False),
])
def test_request_allowed(host, origin, allowed_hosts, expected):
    assert browser_guard.request_allowed(host, origin, allowed_hosts) is expected


# ── gateway ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_gateway_refuses_a_run_posted_from_another_site():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway.run_ask_async", new=AsyncMock()) as run_ask:
            resp = await client.post(
                "/run", data='{"alias": "cld-write", "task": "x"}',
                headers={"Origin": EVIL, "Content-Type": "text/plain"},
            )
    assert resp.status == 403
    run_ask.assert_not_awaited()


@pytest.mark.asyncio
async def test_loopback_gateway_refuses_a_request_addressed_to_another_host():
    app = gateway.make_app(token="", allowed_hosts=LOOPBACK)
    async with TestClient(TestServer(app)) as client:
        rebound = await client.get("/aliases", headers={"Host": "evil.example:7070"})
        local = await client.get("/health")
    assert rebound.status == 403
    assert local.status == 200


# ── dashboard ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dashboard_gates_cannot_be_opened_from_another_site(monkeypatch):
    monkeypatch.delenv("ORCHESTRA_DASHBOARD__DISPATCH_ENABLED", raising=False)
    monkeypatch.delenv("ORCHESTRA_DASHBOARD__APPROVALS_ENABLED", raising=False)
    try:
        async with TestClient(TestServer(dashboard_app.make_app())) as client:
            resp = await client.post(
                "/api/settings",
                data={"dispatch_enabled": "1", "approvals_enabled": "1"},
                headers={"Origin": EVIL},
            )
        assert resp.status == 403
        assert config.dashboard_dispatch_enabled() is False
        assert config.dashboard_approvals_enabled() is False
    finally:
        config.reload_config()


@pytest.mark.asyncio
async def test_dashboard_page_can_still_post_to_itself(monkeypatch):
    for key in ("DASHBOARD__DISPATCH_ENABLED", "DASHBOARD__APPROVALS_ENABLED",
                "FLEET__INCLUDE_REMOTE_NODES", "DASHBOARD__BUDGET_USD"):
        monkeypatch.setenv(f"ORCHESTRA_{key}", "false" if "BUDGET" not in key else "0")
    try:
        async with TestClient(TestServer(dashboard_app.make_app())) as client:
            own_origin = str(client.make_url("/")).rstrip("/")
            resp = await client.post(
                "/api/settings", data={"budget_usd": "1"}, headers={"Origin": own_origin},
            )
        assert resp.status == 200
    finally:
        monkeypatch.undo()
        config.reload_config()


def test_dashboard_refuses_a_public_host_without_a_password(monkeypatch):
    monkeypatch.setattr(config, "dashboard_auth_token", lambda: "")
    with patch.object(dashboard_app.web, "run_app") as run_app:
        with pytest.raises(ValueError, match="auth"):
            dashboard_app.run_app(host="0.0.0.0", port=8080)
    run_app.assert_not_called()


# ── WebSocket daemon ────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _connects(port: int, headers: dict | None = None) -> bool:
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}", additional_headers=headers):
            return True
    except websockets.exceptions.InvalidStatus:
        return False


@pytest.mark.asyncio
async def test_daemon_refuses_a_connection_opened_by_a_web_page():
    port = _free_port()
    async with websockets.serve(srv.server_loop, "127.0.0.1", port, **srv.serve_options("")):
        assert await _connects(port, {"Origin": EVIL}) is False
        assert await _connects(port) is True


@pytest.mark.asyncio
async def test_daemon_with_a_token_refuses_a_client_without_it():
    port = _free_port()
    async with websockets.serve(srv.server_loop, "127.0.0.1", port, **srv.serve_options("s3cret")):
        assert await _connects(port) is False
        assert await _connects(port, {"Authorization": "Bearer wrong"}) is False
        assert await _connects(port, {"Authorization": "Bearer s3cret"}) is True


@pytest.mark.asyncio
async def test_daemon_refuses_a_public_host_without_a_token():
    with pytest.raises(ValueError, match="token"):
        await srv.start_server("0.0.0.0", _free_port())


# ── `orchestra claude` ──────────────────────────────────────────────────────

def test_claude_is_never_handed_the_users_own_api_key():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "real-key"}, clear=True):
        env = claude_launcher.build_claude_env("127.0.0.1", 7070, "oll-fast", token="tok")
    assert env["ANTHROPIC_API_KEY"] == "tok"


def test_a_gateway_started_for_claude_has_its_own_token():
    with patch("orchestra.claude_launcher.gateway_settings", return_value=("127.0.0.1", 7070, "oll-fast")), \
         patch("orchestra.claude_launcher.gateway_healthy", return_value=False), \
         patch("orchestra.claude_launcher.wait_for_gateway", return_value=True), \
         patch("orchestra.claude_launcher.subprocess.Popen") as popen, \
         patch("orchestra.claude_launcher.subprocess.run", return_value=MagicMock(returncode=0)) as run:
        claude_launcher.run_claude()

    argv = popen.call_args.args[0]
    gateway_token = popen.call_args.kwargs["env"]["ORCHESTRA_GATEWAY__TOKEN"]
    assert len(gateway_token) >= 32
    assert gateway_token not in argv                      # argv is visible in `ps`
    assert run.call_args.kwargs["env"]["ANTHROPIC_API_KEY"] == gateway_token


def test_serve_reads_its_token_from_the_environment():
    from orchestra import cli

    with patch("orchestra.gateway.serve") as serve:
        result = CliRunner().invoke(cli.app, ["serve"], env={"ORCHESTRA_GATEWAY__TOKEN": "from-env"})
    assert result.exit_code == 0
    assert serve.call_args.kwargs["token"] == "from-env"


# ── `orchestra nodes-scan` ──────────────────────────────────────────────────

class _Health(io.BytesIO):
    status = 200

    def __init__(self, body: dict):
        super().__init__(json.dumps(body).encode())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _scan_one(body: dict) -> tuple[list[dict], list[dict]]:
    sent_headers: list[dict] = []

    def fake_urlopen(req, timeout=None):
        if "192.168.1.9:" not in req.full_url:
            raise OSError("no route")
        sent_headers.append(dict(req.header_items()))
        return _Health(body)

    with patch("orchestra.remote_nodes.urllib.request.urlopen", fake_urlopen):
        return rn.scan_subnet("192.168.1", port=7070), sent_headers


def test_scan_sends_no_secret_to_the_hosts_it_probes():
    found, sent_headers = _scan_one({"service": "orchestra-gateway", "host": "winbox"})
    assert [f["host"] for f in found] == ["winbox"]
    assert all("Authorization" not in headers for headers in sent_headers)


def test_scan_ignores_a_host_that_is_not_a_gateway():
    found, _ = _scan_one({"status": "ok", "host": "printer"})
    assert found == []


def test_scan_does_not_take_a_config_key_from_the_responder():
    found, _ = _scan_one({"service": "orchestra-gateway", "host": 'pc]\n[dashboard]\nx = "'})
    assert [f["host"] for f in found] == ["node-192-168-1-9"]


def test_nodes_scan_keeps_a_node_that_already_has_that_name():
    from orchestra import cli

    found = [{"ip": "192.168.1.9", "host": "windows", "port": 7070}]
    known = {"windows": {"url": "http://192.168.1.42:7070", "token": "t"}}
    with patch("orchestra.remote_nodes.scan_subnet", return_value=found), \
         patch("orchestra.remote_nodes.verify_gateway", return_value=True), \
         patch("orchestra.remote_nodes.list_nodes", return_value=known), \
         patch("orchestra.cli._set_node_config") as set_node:
        result = CliRunner().invoke(cli.app, ["nodes-scan", "--subnet", "192.168.1", "--token", "t"])

    assert result.exit_code == 0
    set_node.assert_not_called()
    assert "192.168.1.42" in result.output


# ── what the first review of this boundary found ────────────────────────────

@pytest.mark.parametrize("fetch_site, expected", [
    ("cross-site", False),      # <img>, <script>, a link on another site: no Origin is sent
    ("same-site", False),       # another port on localhost
    ("same-origin", True),
    ("none", True),             # typed into the address bar
    (None, True),               # not a browser
])
def test_a_browser_request_started_by_another_site_is_refused(fetch_site, expected):
    assert browser_guard.request_allowed("127.0.0.1:8080", None, LOOPBACK, fetch_site) is expected


@pytest.mark.asyncio
async def test_dashboard_refuses_a_model_run_requested_through_an_image_tag():
    with patch.object(dashboard_app.config, "dashboard_dispatch_enabled", return_value=True), \
         patch("orchestra.gateway_dispatch.stream_target") as stream_target:
        async with TestClient(TestServer(dashboard_app.make_app(LOOPBACK))) as client:
            resp = await client.get(
                "/api/character/cld-write/chat/stream", params={"message": "x"},
                headers={"Sec-Fetch-Site": "cross-site"},
            )
    assert resp.status == 403
    stream_target.assert_not_called()


@pytest.mark.asyncio
async def test_a_second_origin_header_does_not_hide_the_first():
    from multidict import CIMultiDict

    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        own = str(client.make_url("/")).rstrip("/")
        resp = await client.get("/health", headers=CIMultiDict([("Origin", own), ("Origin", EVIL)]))
    assert resp.status == 403


@pytest.mark.asyncio
async def test_a_client_that_presents_the_token_is_not_asked_for_its_origin():
    async with TestClient(TestServer(gateway.make_app(token="tok"))) as client:
        with patch("orchestra.gateway.available_aliases", return_value=["x"]):
            with_token = await client.get(
                "/aliases", headers={"Origin": "http://localhost:3000", "Authorization": "Bearer tok"})
            without = await client.get("/aliases", headers={"Origin": "http://localhost:3000"})
    assert with_token.status == 200
    assert without.status == 403


@pytest.mark.asyncio
async def test_federation_server_refuses_a_request_from_another_site():
    from orchestra.federation_server import federation_http

    async with TestClient(TestServer(federation_http.create_app())) as client:
        resp = await client.get("/federation/health", headers={"Origin": EVIL})
    assert resp.status == 403


@pytest.mark.asyncio
async def test_start_server_applies_the_connection_rules():
    seen: dict = {}

    class _Stop(Exception):
        pass

    def fake_serve(handler, **kwargs):
        seen.update(kwargs)
        raise _Stop

    with patch.object(srv.websockets, "serve", fake_serve), patch("orchestra.live_status.install"):
        with pytest.raises(_Stop):
            await srv.start_server("127.0.0.1", _free_port(), "tok")
    seen.pop("sock").close()

    assert seen["origins"] == [None]
    assert callable(seen["process_request"])


@pytest.mark.asyncio
async def test_daemon_answers_a_malformed_credential_with_401():
    port = _free_port()
    async with websockets.serve(srv.server_loop, "127.0.0.1", port, **srv.serve_options("s3cret")):
        try:
            async with websockets.connect(
                f"ws://127.0.0.1:{port}",
                additional_headers=[("Authorization", "Bearer a"), ("Authorization", "Bearer s3cret")],
            ):
                status = 101
        except websockets.exceptions.InvalidStatus as exc:
            status = exc.response.status_code
    assert status == 401


# ── a gateway proves it knows the token; the token is not sent to find out ───

@pytest.mark.asyncio
async def test_gateway_proves_its_token_for_a_nonce():
    nonce = "n" * 32
    async with TestClient(TestServer(gateway.make_app(token="tok"))) as client:
        body = await (await client.get("/identify", params={"nonce": nonce})).json()
        bad = await client.get("/identify", params={"nonce": "short"})

    assert body["token_required"] is True
    assert rn.proof_matches(body["proof"], "tok", nonce, "127.0.0.1")
    assert not rn.proof_matches(body["proof"], "other", nonce, "127.0.0.1")
    assert not rn.proof_matches(body["proof"], "tok", nonce, "192.168.1.66")   # a relayed answer
    assert bad.status == 400


def _scan_cli(found, *, verified=True, token="t", nodes=None, backends=None):
    from orchestra import cli

    with patch("orchestra.remote_nodes.scan_subnet", return_value=found), \
         patch("orchestra.remote_nodes.verify_gateway", return_value=verified), \
         patch("orchestra.remote_nodes.list_nodes", return_value=nodes or {}), \
         patch("orchestra.providers.ollama_backends.list_backends", return_value=backends or {}), \
         patch("orchestra.cli._set_node_config") as set_node:
        args = ["nodes-scan", "--subnet", "192.168.1"] + (["--token", token] if token else [])
        result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    return [call.args[0] for call in set_node.call_args_list]


_WINBOX = {"ip": "192.168.1.9", "host": "winbox", "port": 7070}


def test_nodes_scan_adds_a_gateway_that_proved_the_token():
    assert _scan_cli([_WINBOX]) == ["winbox"]


def test_nodes_scan_skips_a_host_that_cannot_prove_the_token():
    assert _scan_cli([_WINBOX], verified=False) == []


def test_nodes_scan_adds_nothing_without_a_token_to_check():
    assert _scan_cli([_WINBOX], token="") == []


def test_nodes_scan_does_not_shadow_an_ollama_backend():
    assert _scan_cli([_WINBOX], backends={"WinBox": "http://192.168.1.42:11434"}) == []


def test_nodes_scan_adds_a_name_once_per_scan():
    twin = {"ip": "192.168.1.10", "host": "Winbox", "port": 7070}
    assert _scan_cli([_WINBOX, twin]) == ["winbox"]


# ── `orchestra claude` and a gateway that is already running ────────────────

def test_claude_is_not_started_against_a_gateway_whose_token_it_lacks():
    with patch("orchestra.claude_launcher.gateway_settings", return_value=("127.0.0.1", 7070, "oll-fast")), \
         patch("orchestra.claude_launcher.gateway_healthy", return_value=True), \
         patch("orchestra.remote_nodes.verify_gateway", return_value=False), \
         patch("orchestra.claude_launcher.subprocess.run") as run:
        with pytest.raises(RuntimeError, match="token"):
            claude_launcher.run_claude()
    run.assert_not_called()


def test_serve_falls_back_to_the_token_in_the_config_file(monkeypatch):
    from orchestra import cli

    monkeypatch.delenv("ORCHESTRA_GATEWAY__TOKEN", raising=False)
    monkeypatch.setitem(config._cfg, "gateway", {"token": "from-file"})
    with patch("orchestra.gateway.serve") as serve:
        result = CliRunner().invoke(cli.app, ["serve"])
    assert result.exit_code == 0
    assert serve.call_args.kwargs["token"] == "from-file"


def test_a_token_from_the_environment_is_used_as_written(monkeypatch):
    monkeypatch.setenv("ORCHESTRA_GATEWAY__TOKEN", "0123456")
    assert claude_launcher.configured_token() == "0123456"

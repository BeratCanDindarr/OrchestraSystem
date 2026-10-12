"""Launch Claude Code (or any Anthropic-API tool) pointed at this machine's Orchestra gateway.

`orchestra claude [args]` ensures a local gateway is up on 127.0.0.1 and runs `claude` with
ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY / ANTHROPIC_MODEL set, so it drives Orchestra's models —
including agentic tool use on tool-capable Ollama models. If this command started the gateway,
it stops it again when claude exits."""
from __future__ import annotations

import contextlib
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request

from orchestra import config

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 7070
_DEFAULT_MODEL = "oll-fast"
# claude requires a non-empty key. A gateway this command starts gets a fresh token; one
# that was already running is given the configured token, or this placeholder, which a
# token-less gateway accepts.
_LOCAL_KEY = "orchestra-local"
_TOKEN_ENV = "ORCHESTRA_GATEWAY__TOKEN"  # read by `orchestra serve`; argv would show in `ps`


def gateway_settings() -> tuple[str, int, str]:
    """(host, port, default_model) from the optional [gateway] config section."""
    cfg = config.section("gateway")
    host = str(cfg.get("host", _DEFAULT_HOST))
    port = int(cfg.get("port", _DEFAULT_PORT))
    model = str(cfg.get("default_model", _DEFAULT_MODEL))
    return host, port, model


def gateway_healthy(host: str, port: int, timeout: float = 2.0) -> bool:
    """True if an Orchestra gateway answers /health on host:port."""
    req = urllib.request.Request(f"http://{host}:{port}/health", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, TimeoutError):
        return False


def wait_for_gateway(host: str, port: int, attempts: int = 30, delay: float = 0.5) -> bool:
    """Poll /health until the gateway comes up (or give up)."""
    for _ in range(attempts):
        if gateway_healthy(host, port):
            return True
        time.sleep(delay)
    return False


def configured_token() -> str:
    """The token of a gateway the user runs themselves ([gateway] token), if any."""
    return str(config.section("gateway").get("token", "") or "")


def build_claude_env(host: str, port: int, model: str, token: str = _LOCAL_KEY) -> dict:
    """A copy of the environment with the Anthropic-API vars pointed at the gateway.

    The key is the gateway's token: the user's own ANTHROPIC_API_KEY is never sent to it.
    """
    env = dict(os.environ)
    env["ANTHROPIC_BASE_URL"] = f"http://{host}:{port}"
    env["ANTHROPIC_API_KEY"] = token
    env["ANTHROPIC_MODEL"] = model
    return env


def _spawn_gateway(host: str, port: int, token: str) -> subprocess.Popen:
    """Start a local gateway in the background (detached, silent) that requires *token*."""
    return subprocess.Popen(
        [sys.executable, "-m", "orchestra", "serve", "--host", host, "--port", str(port)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
        env={**os.environ, _TOKEN_ENV: token},
    )


def run_claude(
    model: str | None = None, port: int | None = None, extra_args: list[str] | None = None
) -> int:
    """Ensure a local gateway is up, then run `claude` against it in the foreground.

    Returns claude's exit code. If this call started the gateway, it is stopped on exit.
    Raises RuntimeError if the gateway can't be started or `claude` isn't installed.
    """
    host, cfg_port, cfg_model = gateway_settings()
    port = port or cfg_port
    model = model or cfg_model
    started: subprocess.Popen | None = None
    token = configured_token() or _LOCAL_KEY
    if not gateway_healthy(host, port):
        token = secrets.token_urlsafe(32)
        started = _spawn_gateway(host, port, token)
        if not wait_for_gateway(host, port):
            started.terminate()
            raise RuntimeError(f"Orchestra gateway did not come up on {host}:{port}")
    try:
        env = build_claude_env(host, port, model, token)
        return subprocess.run(["claude", *(extra_args or [])], env=env).returncode
    except FileNotFoundError as exc:
        raise RuntimeError("`claude` not found on PATH — install Claude Code first") from exc
    finally:
        if started is not None:
            started.terminate()
            try:
                started.wait(timeout=5)
            except subprocess.TimeoutExpired:
                started.kill()  # escalate if graceful shutdown stalls
                with contextlib.suppress(Exception):
                    started.wait(timeout=2)

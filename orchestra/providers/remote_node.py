"""Provider that delegates a task to a remote Orchestra node's agent gateway."""
from __future__ import annotations

import asyncio
import json as _json
import urllib.error
import urllib.request
from typing import Any

from orchestra.providers.base import BaseProvider


class RemoteNodeProvider(BaseProvider):
    name = "remote"

    def __init__(self, name: str, url: str, token: str = "") -> None:
        self._node = name
        self._url = url.rstrip("/")
        self._token = token

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def is_available(self) -> bool:
        req = urllib.request.Request(f"{self._url}/health", headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def model_label(self, effort_or_model: str) -> str:
        return f"remote/{self._node}/{effort_or_model}"

    def build_command(self, prompt: str, effort_or_model: str) -> list[str]:
        raise NotImplementedError("RemoteNodeProvider runs over HTTP, not a subprocess")

    def native_run(self, prompt: str, effort_or_model: str, timeout: int = 180, **_kwargs) -> tuple[str, int]:
        payload = _json.dumps({"alias": effort_or_model, "task": prompt}).encode()
        req = urllib.request.Request(
            f"{self._url}/run", data=payload, headers=self._headers(), method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body: dict[str, Any] = _json.loads(resp.read())
                if resp.status != 200:
                    return f"[ERROR] node {self._node}: {body.get('error', resp.status)}", 1
                return str(body.get("stdout", "")), 0
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            return f"[ERROR] node {self._node} HTTP {exc.code}: {detail}", 1
        except Exception as exc:
            return f"[ERROR] cannot reach node {self._node!r} at {self._url}: {exc}", 1

    async def async_run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        """Async path (parallel/dual executor) — run the blocking HTTP call in a thread.

        Without this, BaseProvider.async_run falls back to self.run → build_command,
        which raises NotImplementedError for this HTTP-only provider.
        """
        return await asyncio.to_thread(
            self.native_run, prompt, effort_or_model, timeout=timeout, **kwargs
        )

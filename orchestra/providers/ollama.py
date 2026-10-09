"""Ollama local provider adapter — HTTP streaming."""
from __future__ import annotations

import json as _json
import time
import urllib.error
import urllib.request
from typing import Iterator

from orchestra import config
from orchestra.providers.base import BaseProvider
from orchestra.engine.tracer import TimingContext


MODELS = {
    # Cloud tags: local chat models were removed because they made the machine thrash.
    # Only the embedding model below still runs locally.
    "coder":    "minimax-m3:cloud",
    "analyst":  "glm-5.3:cloud",
    "fast":     "deepseek-v4.1-flash:cloud",
    "mini":     "deepseek-v4.1-flash:cloud",
    "llama3":   "llama3:latest",
    "mistral":  "mistral:latest",
    "embed":    "nomic-embed-text-v2-moe:latest",
}

_THINK_LEVELS = ("low", "medium", "high")


def think_level(model: str) -> bool | str | None:
    """How much the model should think: False, True, "low"/"medium"/"high", or None to not say.

    Read from `[providers.ollama]`: `think_by_model."<tag>"` for one model, else
    `think` for all of them. Left unsaid, a thinking model reasons as long as it
    likes before its first answer token, and thinking is billed as output.
    """
    section = config.providers_config().get("ollama", {})
    if not isinstance(section, dict):
        return None
    by_model = section.get("think_by_model", {})
    value = by_model[model] if isinstance(by_model, dict) and model in by_model else section.get("think")
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value in _THINK_LEVELS:
        return value
    return None


def _generate_body(model: str, prompt: str, temperature: float | None, seed: int | None) -> dict:
    """The /api/generate request for one streamed prompt."""
    body: dict = {"model": model, "prompt": prompt, "stream": True}
    if temperature is not None:
        body["temperature"] = temperature
    if seed is not None:
        body["seed"] = seed
    think = think_level(model)
    if think is not None:
        body["think"] = think
    return body


def _deadline_error(timeout: int) -> str:
    return f"[ERROR] Ollama: no complete answer within {timeout}s (the model may still be thinking)"


def _ollama_base() -> str:
    return config.providers_config().get("ollama", {}).get("base_url", "http://localhost:11434")


class OllamaProvider(BaseProvider):
    name = "ollama"
    supports_native_messages = True

    def __init__(self, base_url: str | None = None) -> None:
        self._base = base_url

    def _base_url(self) -> str:
        return self._base or _ollama_base()

    def is_available(self) -> bool:
        req = urllib.request.Request(f"{self._base_url()}/api/tags", method="GET")
        healthcheck_timeout = int(config.availability_config().get("ollama_healthcheck_timeout_seconds", 2))
        try:
            with urllib.request.urlopen(req, timeout=healthcheck_timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def model_label(self, effort_or_model: str) -> str:
        model = MODELS.get(effort_or_model, effort_or_model)
        return f"ollama/{model}"

    def build_command(self, prompt: str, effort_or_model: str) -> list[str]:
        # Fallback for pipeline compatibility — native_run() is preferred
        model = MODELS.get(effort_or_model, effort_or_model)
        return ["ollama", "run", model, prompt]

    def native_run(
        self,
        prompt: str,
        effort_or_model: str,
        timeout: int = 120,
        stream_callback=None,
        temperature: float | None = None,
        seed: int | None = None,
        **_kwargs,
    ) -> tuple[str, int]:
        """HTTP streaming run — bypasses subprocess, keeps model warm in VRAM."""
        model = MODELS.get(effort_or_model, effort_or_model)
        payload = _json.dumps(_generate_body(model, prompt, temperature, seed)).encode()
        req = urllib.request.Request(
            f"{self._base_url()}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        full_output: list[str] = []
        # The read timeout only catches a silent socket; a model that keeps thinking
        # keeps sending chunks, so the whole call gets a deadline of its own.
        deadline = time.monotonic() + timeout
        try:
            with TimingContext() as timer:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    for raw_line in resp:
                        if time.monotonic() > deadline:
                            return _deadline_error(timeout), 124
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            data = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue
                        token = data.get("response", "")
                        if token:
                            timer.mark_first_chunk()
                            full_output.append(token)
                            if stream_callback:
                                stream_callback(token)
                        if data.get("done"):
                            break
            return "".join(full_output).strip(), 0
        except Exception as exc:
            return f"[ERROR] Ollama: {exc}", 1

    async def async_run(self, prompt: str, effort_or_model: str, timeout: int = 120, **kwargs) -> tuple[str, int]:
        """Async HTTP streaming run using aiohttp."""
        import aiohttp
        model = MODELS.get(effort_or_model, effort_or_model)
        body = _generate_body(model, prompt, kwargs.get("temperature"), kwargs.get("seed"))

        full_output: list[str] = []
        try:
            with TimingContext() as timer:
                timeout_obj = aiohttp.ClientTimeout(total=timeout)
                async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                    async with session.post(
                        f"{self._base_url()}/api/generate",
                        json=body,
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        async for raw_line in resp.content:
                            line = raw_line.strip()
                            if not line:
                                continue
                            try:
                                data = _json.loads(line)
                            except _json.JSONDecodeError:
                                continue
                            token = data.get("response", "")
                            if token:
                                timer.mark_first_chunk()
                                full_output.append(token)
                                stream_callback = kwargs.get("stream_callback")
                                if stream_callback:
                                    stream_callback(token)
                            if data.get("done"):
                                break
            return "".join(full_output).strip(), 0
        except Exception as exc:
            return f"[ERROR] Ollama: {exc}", 1

    def stream(self, prompt: str, effort_or_model: str, timeout: int = 120, **kwargs) -> Iterator[str]:
        """Stream Ollama output as a generator of text chunks."""
        model = MODELS.get(effort_or_model, effort_or_model)
        body = _generate_body(model, prompt, kwargs.get("temperature"), kwargs.get("seed"))
        payload = _json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self._base_url()}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        deadline = time.monotonic() + timeout
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw_line in resp:
                    if time.monotonic() > deadline:
                        yield _deadline_error(timeout)
                        return
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        data = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
        except Exception as exc:
            yield f"[ERROR] Ollama: {exc}"

    def native_messages(self, payload: dict, timeout: int = 180, **kwargs) -> dict:
        """Anthropic-shaped request -> POST /api/chat -> Anthropic-shaped response.

        Errors RAISE (no silent text fallback): the gateway maps them to a generic
        Anthropic 500 and logs the detail server-side.
        """
        from orchestra.providers import ollama_native as _native
        model = MODELS.get(payload.get("model", ""), payload.get("model", ""))
        num_ctx = int(config.providers_config().get("ollama", {}).get("num_ctx", 32768))
        body: dict = {
            "model": model,
            "messages": _native.to_ollama_messages(payload.get("system"), payload.get("messages", [])),
            "stream": False,
            "options": {"num_ctx": num_ctx},
        }
        if payload.get("tools"):
            body["tools"] = _native.to_ollama_tools(payload["tools"])
        think = think_level(model)
        if think is not None:
            body["think"] = think
        req = urllib.request.Request(
            f"{self._base_url()}/api/chat",
            data=_json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = _json.loads(resp.read())
        except urllib.error.HTTPError as exc:  # surface body detail to SERVER logs only
            detail = exc.read().decode(errors="replace")[:500]
            raise RuntimeError(f"ollama /api/chat {exc.code}: {detail}") from exc
        blocks, stop_reason = _native.from_ollama_response(data.get("message", {}) or {})
        return {
            "content": blocks,
            "stop_reason": stop_reason,
            "usage": {
                "input_tokens": int(data.get("prompt_eval_count") or 0),
                "output_tokens": int(data.get("eval_count") or 0),
            },
        }

    @staticmethod
    def warmup(model: str = "qwen2.5-coder:7b", keep_alive: str = "60m") -> bool:
        """Load model into VRAM without generating — call once at daemon start."""
        warmup_timeout = int(config.availability_config().get("ollama_warmup_timeout_seconds", 60))
        payload = _json.dumps({"model": model, "keep_alive": keep_alive}).encode()
        req = urllib.request.Request(
            f"{_ollama_base()}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=warmup_timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def embed(self, text: str, model: str = "nomic-embed-text-v2-moe:latest") -> list[float]:
        embed_timeout = int(config.availability_config().get("ollama_embed_timeout_seconds", 30))
        payload = _json.dumps({"model": model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{self._base_url()}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=embed_timeout) as resp:
                return _json.loads(resp.read()).get("embedding", [])
        except Exception:
            return []

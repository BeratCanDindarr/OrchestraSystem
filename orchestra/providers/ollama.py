"""Ollama over HTTP: health checks, embeddings, the router's short classification call.

Agents run Ollama models through the Claude CLI instead (ollama_agent.py)."""
from __future__ import annotations

import contextvars
import json as _json
import time
import urllib.error
import urllib.request

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


# Token counts of the call made in the current thread or task. The provider object is shared
# by every Ollama agent of a run, so one attribute on it would mix up concurrent calls.
_usage: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar("ollama_usage", default=None)


def _note_usage(final_chunk: dict) -> None:
    """Record what Ollama billed, from the last chunk of a response (thinking tokens included)."""
    prompt_tokens = final_chunk.get("prompt_eval_count")
    completion_tokens = final_chunk.get("eval_count")
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        _usage.set({"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens})


class OllamaProvider(BaseProvider):
    name = "ollama"

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

    def last_usage(self) -> dict[str, int] | None:
        """Token counts of the last call made in this thread or task, or None."""
        return _usage.get()

    def reset_usage(self) -> None:
        _usage.set(None)

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
        _usage.set(None)
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
                            _note_usage(data)
                            break
            return "".join(full_output).strip(), 0
        except Exception as exc:
            return f"[ERROR] Ollama: {exc}", 1





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

from __future__ import annotations
"""OpenAI SDK provider — uses OpenAI Python SDK with streaming.

Requires: OPENAI_API_KEY env var + `pip install openai`.
Supports: gpt-4o, gpt-4o-mini, o3-mini.
"""

import os

from orchestra import config
from orchestra.providers.base import BaseProvider
from orchestra.engine.tracer import TimingContext

_DEFAULT_SYSTEM = "You are a helpful AI coding assistant."

_DEFAULT_MODELS: dict[str, str] = {
    "fast": "gpt-4o-mini",
    "pro": "gpt-4o",
    "reasoning": "o3-mini",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "o3-mini": "o3-mini",
}


class OpenAISDKProvider(BaseProvider):
    """OpenAI SDK-based provider with streaming support."""

    name = "openai_sdk"

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("OPENAI_API_KEY"))

    # ── Interface ─────────────────────────────────────────────────────────────

    def model_label(self, effort_or_model: str) -> str:
        cfg_models = config.caching_config().get("models", {})
        model = cfg_models.get(effort_or_model) or _DEFAULT_MODELS.get(effort_or_model, effort_or_model)
        return f"openai_sdk/{model}"

    def build_command(self, prompt: str, effort_or_model: str) -> list[str]:
        raise NotImplementedError("OpenAISDKProvider uses run() directly — build_command not applicable")

    # ── Execution ─────────────────────────────────────────────────────────────

    async def async_run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        """Async version using OpenAI's async client if available, otherwise thread fallback."""
        try:
            import openai
        except ImportError:
            return "[ERROR] openai package not installed — run: pip install openai", 1

        if not hasattr(openai, "AsyncOpenAI"):
            return await super().async_run(prompt, effort_or_model, timeout=timeout, **kwargs)

        cfg = config.caching_config()
        cfg_models = cfg.get("models", {})
        model = cfg_models.get(effort_or_model) or _DEFAULT_MODELS.get(effort_or_model, effort_or_model)

        client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        messages: list[dict] = [
            {"role": "system", "content": _DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        stream_kwargs: dict = dict(
            model=model,
            messages=messages,
            stream=True,
            timeout=timeout,
        )
        temperature = kwargs.get("temperature")
        seed = kwargs.get("seed")
        if temperature is not None:
            stream_kwargs["temperature"] = temperature
        if seed is not None:
            stream_kwargs["seed"] = seed

        stream_kwargs["stream_options"] = {"include_usage": True}

        output = ""
        prompt_tokens = 0
        completion_tokens = 0
        try:
            with TimingContext() as timer:
                response = await client.chat.completions.create(**stream_kwargs)
                async for chunk in response:
                    timer.mark_first_chunk()
                    if chunk.usage:
                        prompt_tokens = chunk.usage.prompt_tokens or 0
                        completion_tokens = chunk.usage.completion_tokens or 0
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        output += delta.content
            self._last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            return output, 0
        except openai.APIStatusError as exc:
            return f"[ERROR] OpenAI API {exc.status_code}: {exc.message}", 1
        except openai.APIConnectionError as exc:
            return f"[ERROR] OpenAI connection: {exc}", 1
        except Exception as exc:
            return f"[ERROR] {exc}", 1

    def run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        """Call OpenAI API with streaming and usage extraction."""
        try:
            import openai
        except ImportError:
            return "[ERROR] openai package not installed — run: pip install openai", 1

        cfg = config.caching_config()
        cfg_models = cfg.get("models", {})
        model = cfg_models.get(effort_or_model) or _DEFAULT_MODELS.get(effort_or_model, effort_or_model)

        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        messages: list[dict] = [
            {"role": "system", "content": _DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        stream_kwargs: dict = dict(
            model=model,
            messages=messages,
            stream=True,
            timeout=timeout,
        )
        temperature = kwargs.get("temperature")
        seed = kwargs.get("seed")
        if temperature is not None:
            stream_kwargs["temperature"] = temperature
        if seed is not None:
            stream_kwargs["seed"] = seed

        # Request usage to be included in the final stream chunk when supported
        stream_kwargs["stream_options"] = {"include_usage": True}

        output = ""
        prompt_tokens = 0
        completion_tokens = 0
        try:
            with TimingContext() as timer:
                response = client.chat.completions.create(**stream_kwargs)
                for chunk in response:
                    timer.mark_first_chunk()
                    if chunk.usage:
                        prompt_tokens = chunk.usage.prompt_tokens or 0
                        completion_tokens = chunk.usage.completion_tokens or 0
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        output += delta.content
            self._last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            return output, 0
        except openai.APIStatusError as exc:
            return f"[ERROR] OpenAI API {exc.status_code}: {exc.message}", 1
        except openai.APIConnectionError as exc:
            return f"[ERROR] OpenAI connection: {exc}", 1
        except Exception as exc:  # noqa: BLE001
            return f"[ERROR] {exc}", 1

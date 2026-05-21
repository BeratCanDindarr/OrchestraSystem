from __future__ import annotations
"""Azure OpenAI provider — uses OpenAI Python SDK with Azure configuration.

Requires:
  - AZURE_OPENAI_API_KEY
  - AZURE_OPENAI_ENDPOINT
  - AZURE_OPENAI_DEPLOYMENT_NAME (or model name as deployment)
  - OPENAI_API_VERSION (optional, defaults to 2024-02-01)
"""

import os
import logging

from orchestra import config
from orchestra.providers.base import BaseProvider
from orchestra.engine.tracer import TimingContext

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM = "You are a helpful AI coding assistant."

_DEFAULT_MODELS: dict[str, str] = {
    "fast": "gpt-4o-mini",
    "pro": "gpt-4o",
    "reasoning": "o3-mini",
}


class AzureOpenAIProvider(BaseProvider):
    """Azure OpenAI SDK-based provider."""

    name = "azure_openai"

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("AZURE_OPENAI_API_KEY")) and bool(os.environ.get("AZURE_OPENAI_ENDPOINT"))

    # ── Interface ─────────────────────────────────────────────────────────────

    def model_label(self, effort_or_model: str) -> str:
        cfg_models = config.caching_config().get("models", {})
        model = cfg_models.get(effort_or_model) or _DEFAULT_MODELS.get(effort_or_model, effort_or_model)
        return f"azure_openai/{model}"

    def build_command(self, prompt: str, effort_or_model: str) -> list[str]:
        raise NotImplementedError("AzureOpenAIProvider uses run() directly — build_command not applicable")

    # ── Execution ─────────────────────────────────────────────────────────────

    def _get_client_params(self) -> dict:
        return {
            "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
            "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
            "api_version": os.environ.get("OPENAI_API_VERSION", "2024-02-01"),
        }

    async def async_run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        """Async version using AzureOpenAI async client."""
        try:
            import openai
        except ImportError:
            return "[ERROR] openai package not installed — run: pip install openai", 1

        if not hasattr(openai, "AsyncAzureOpenAI"):
            return await super().async_run(prompt, effort_or_model, timeout=timeout, **kwargs)

        cfg = config.caching_config()
        cfg_models = cfg.get("models", {})
        deployment = cfg_models.get(effort_or_model) or _DEFAULT_MODELS.get(effort_or_model, effort_or_model)
        # Use env override for deployment if specific deployment name is set
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", deployment)

        client = openai.AsyncAzureOpenAI(**self._get_client_params())
        messages: list[dict] = [
            {"role": "system", "content": _DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        stream_kwargs: dict = dict(
            model=deployment,
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
        except Exception as exc:
            return f"[ERROR] Azure OpenAI: {exc}", 1

    def run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        """Call Azure OpenAI API with streaming."""
        try:
            import openai
        except ImportError:
            return "[ERROR] openai package not installed — run: pip install openai", 1

        cfg = config.caching_config()
        cfg_models = cfg.get("models", {})
        deployment = cfg_models.get(effort_or_model) or _DEFAULT_MODELS.get(effort_or_model, effort_or_model)
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", deployment)

        client = openai.AzureOpenAI(**self._get_client_params())
        messages: list[dict] = [
            {"role": "system", "content": _DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        stream_kwargs: dict = dict(
            model=deployment,
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
        except Exception as exc:
            return f"[ERROR] Azure OpenAI: {exc}", 1

from __future__ import annotations
"""DeepSeek provider — uses OpenAI Python SDK with DeepSeek base URL.

Requires: DEEPSEEK_API_KEY env var + `pip install openai`.
Supports: deepseek-chat, deepseek-reasoner.
"""

import os

from orchestra import config
from orchestra.providers.base import BaseProvider
from orchestra.engine.tracer import TimingContext

_DEFAULT_SYSTEM = "You are a helpful AI coding assistant."

_DEFAULT_MODELS: dict[str, str] = {
    "chat": "deepseek-chat",
    "reasoner": "deepseek-reasoner",
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-reasoner",
}


class DeepSeekProvider(BaseProvider):
    """DeepSeek provider using OpenAI-compatible API."""

    name = "deepseek"

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("DEEPSEEK_API_KEY"))

    # ── Interface ─────────────────────────────────────────────────────────────

    def model_label(self, effort_or_model: str) -> str:
        cfg_models = config.caching_config().get("models", {})
        model = cfg_models.get(effort_or_model) or _DEFAULT_MODELS.get(effort_or_model, effort_or_model)
        return f"deepseek/{model}"

    def build_command(self, prompt: str, effort_or_model: str) -> list[str]:
        raise NotImplementedError("DeepSeekProvider uses run() directly — build_command not applicable")

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        """Call DeepSeek API with streaming and usage extraction."""
        try:
            import openai
        except ImportError:
            return "[ERROR] openai package not installed — run: pip install openai", 1

        cfg = config.caching_config()
        cfg_models = cfg.get("models", {})
        model = cfg_models.get(effort_or_model) or _DEFAULT_MODELS.get(effort_or_model, effort_or_model)

        client = openai.OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
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
            self.last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            return output, 0
        except openai.APIStatusError as exc:
            return f"[ERROR] DeepSeek API {exc.status_code}: {exc.message}", 1
        except openai.APIConnectionError as exc:
            return f"[ERROR] DeepSeek connection: {exc}", 1
        except Exception as exc:  # noqa: BLE001
            return f"[ERROR] {exc}", 1

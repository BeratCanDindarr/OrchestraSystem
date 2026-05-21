from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
"""Claude SDK provider — uses Anthropic Python SDK with cache_control ephemeral.

Activated when `[caching] enabled = true` and `provider = "sdk"` in config.toml.
Requires: ANTHROPIC_API_KEY env var + `pip install anthropic`.
Expected benefit: ~90% token savings on repeated system prompts via server-side caching.
"""

import os
from typing import Iterator

from orchestra import config
from orchestra.providers.base import BaseProvider
from orchestra.engine.tracer import TimingContext

_DEFAULT_SYSTEM = "You are a helpful AI coding assistant."

_DEFAULT_MODELS: dict[str, str] = {
    "light":  "claude-3-5-sonnet-20241022",
    "medium": "claude-3-5-sonnet-20241022",
    "heavy":  "claude-3-opus-20240229",
    "sonnet": "claude-3-5-sonnet-20241022",
    "opus":   "claude-3-opus-20240229",
}


class ClaudeSDKProvider(BaseProvider):
    """Anthropic SDK-based provider with cache_control: ephemeral on system prompt."""

    name = "claude_sdk"

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    # ── Interface ─────────────────────────────────────────────────────────────

    def model_label(self, effort_or_model: str) -> str:
        cfg_models = config.caching_config().get("models", {})
        model = cfg_models.get(effort_or_model) or _DEFAULT_MODELS.get(effort_or_model, effort_or_model)
        return f"claude_sdk/{model}"

    def build_command(self, prompt: str, effort_or_model: str) -> list[str]:
        raise NotImplementedError("ClaudeSDKProvider uses run() directly — build_command not applicable")

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        """Call Anthropic API with streaming and 3-zone cache_control.

        Splits the incoming prompt into static / frozen / volatile zones
        so that the stable parts (role, policies, relevant context) are
        cached via ``cache_control: ephemeral``, while the volatile task
        prompt is sent as a fresh user message.
        """
        try:
            import anthropic
        except ImportError:
            return "[ERROR] anthropic package not installed — run: pip install anthropic", 1

        cfg = config.caching_config()
        cfg_models = cfg.get("models", {})
        model = cfg_models.get(effort_or_model) or _DEFAULT_MODELS.get(effort_or_model, effort_or_model)

        # Split prompt into cache zones. The prompt assembled by _prefixed()
        # follows the format:  role\n\npriority_context\n\ntask_prompt
        parts = prompt.split("\n\n", 2)
        if len(parts) == 3:
            static_zone, frozen_zone, volatile_zone = parts
        else:
            static_zone, frozen_zone, volatile_zone = _DEFAULT_SYSTEM, "", prompt

        response_format = kwargs.get("response_format", "text")
        if response_format == "json":
            volatile_zone += "\n\nPlease respond with valid JSON only."

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        try:
            with TimingContext() as timer:
                output = ""
                system_blocks: list[dict] = [
                    {
                        "type": "text",
                        "text": static_zone,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                if frozen_zone:
                    system_blocks.append(
                        {
                            "type": "text",
                            "text": frozen_zone,
                            "cache_control": {"type": "ephemeral"},
                        }
                    )
                stream_kwargs: dict = dict(
                    model=model,
                    max_tokens=8096,
                    system=system_blocks,
                    messages=[{"role": "user", "content": volatile_zone}],
                    timeout=timeout,
                )
                temperature = kwargs.get("temperature")
                if response_format == "json":
                    temperature = 0
                if temperature is not None:
                    stream_kwargs["temperature"] = temperature
                seed = kwargs.get("seed")
                if seed is not None:
                    stream_kwargs["seed"] = seed
                with client.messages.stream(**stream_kwargs) as stream:
                    for text in stream.text_stream:
                        timer.mark_first_chunk()
                        output += text
                try:
                    final = stream.get_final_message()
                    self._last_usage = {
                        "prompt_tokens": final.usage.input_tokens,
                        "completion_tokens": final.usage.output_tokens,
                    }
                except Exception:
                    logger.exception("Ignored exception")

            return output, 0
        except anthropic.APIStatusError as exc:
            return f"[ERROR] Anthropic API {exc.status_code}: {exc.message}", 1
        except anthropic.APIConnectionError as exc:
            return f"[ERROR] Anthropic connection: {exc}", 1
        except Exception as exc:  # noqa: BLE001
            return f"[ERROR] {exc}", 1

    async def async_run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        """Async version using Anthropic's async client."""
        try:
            import anthropic
        except ImportError:
            return "[ERROR] anthropic package not installed — run: pip install anthropic", 1

        cfg = config.caching_config()
        cfg_models = cfg.get("models", {})
        model = cfg_models.get(effort_or_model) or _DEFAULT_MODELS.get(effort_or_model, effort_or_model)

        parts = prompt.split("\n\n", 2)
        if len(parts) == 3:
            static_zone, frozen_zone, volatile_zone = parts
        else:
            static_zone, frozen_zone, volatile_zone = _DEFAULT_SYSTEM, "", prompt

        response_format = kwargs.get("response_format", "text")
        if response_format == "json":
            volatile_zone += "\n\nPlease respond with valid JSON only."

        client = anthropic.AsyncAnthropic()
        try:
            with TimingContext() as timer:
                output = ""
                system_blocks: list[dict] = [
                    {
                        "type": "text",
                        "text": static_zone,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                if frozen_zone:
                    system_blocks.append(
                        {
                            "type": "text",
                            "text": frozen_zone,
                            "cache_control": {"type": "ephemeral"},
                        }
                    )
                stream_kwargs: dict = dict(
                    model=model,
                    max_tokens=8096,
                    system=system_blocks,
                    messages=[{"role": "user", "content": volatile_zone}],
                    timeout=timeout,
                )
                temperature = kwargs.get("temperature")
                if response_format == "json":
                    temperature = 0
                if temperature is not None:
                    stream_kwargs["temperature"] = temperature
                seed = kwargs.get("seed")
                if seed is not None:
                    stream_kwargs["seed"] = seed
                async with client.messages.stream(**stream_kwargs) as stream:
                    async for text in stream.text_stream:
                        timer.mark_first_chunk()
                        output += text
                try:
                    final = await stream.get_final_message()
                    self._last_usage = {
                        "prompt_tokens": final.usage.input_tokens,
                        "completion_tokens": final.usage.output_tokens,
                    }
                except Exception:
                    logger.exception("Ignored exception")

            return output, 0
        except anthropic.APIStatusError as exc:
            return f"[ERROR] Anthropic API {exc.status_code}: {exc.message}", 1
        except anthropic.APIConnectionError as exc:
            return f"[ERROR] Anthropic connection: {exc}", 1
        except Exception as exc:
            return f"[ERROR] {exc}", 1

    def stream(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> Iterator[str]:
        """Stream Anthropic API output as a generator of text chunks."""
        try:
            import anthropic
        except ImportError:
            yield "[ERROR] anthropic package not installed — run: pip install anthropic"
            return

        cfg = config.caching_config()
        cfg_models = cfg.get("models", {})
        model = cfg_models.get(effort_or_model) or _DEFAULT_MODELS.get(effort_or_model, effort_or_model)

        parts = prompt.split("\n\n", 2)
        if len(parts) == 3:
            static_zone, frozen_zone, volatile_zone = parts
        else:
            static_zone, frozen_zone, volatile_zone = _DEFAULT_SYSTEM, "", prompt

        response_format = kwargs.get("response_format", "text")
        if response_format == "json":
            volatile_zone += "\n\nPlease respond with valid JSON only."

        client = anthropic.Anthropic()
        system_blocks: list[dict] = [
            {
                "type": "text",
                "text": static_zone,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if frozen_zone:
            system_blocks.append(
                {
                    "type": "text",
                    "text": frozen_zone,
                    "cache_control": {"type": "ephemeral"},
                }
            )
        stream_kwargs: dict = dict(
            model=model,
            max_tokens=8096,
            system=system_blocks,
            messages=[{"role": "user", "content": volatile_zone}],
            timeout=timeout,
        )
        temperature = kwargs.get("temperature")
        if response_format == "json":
            temperature = 0
        if temperature is not None:
            stream_kwargs["temperature"] = temperature
        seed = kwargs.get("seed")
        if seed is not None:
            stream_kwargs["seed"] = seed

        with client.messages.stream(**stream_kwargs) as stream:
            for text in stream.text_stream:
                yield text

        try:
            final = stream.get_final_message()
            self._last_usage = {
                "prompt_tokens": final.usage.input_tokens,
                "completion_tokens": final.usage.output_tokens,
            }
        except Exception:
            logger.exception("Ignored exception")


"""Claude SDK provider — uses Anthropic Python SDK with cache_control ephemeral.

Activated when `[caching] enabled = true` and `provider = "sdk"` in config.toml.
Requires: ANTHROPIC_API_KEY env var + `pip install anthropic`.
Expected benefit: ~90% token savings on repeated system prompts via server-side caching.
"""
from __future__ import annotations

import os

from orchestra import config
from orchestra.providers.base import BaseProvider
from orchestra.engine.tracer import TimingContext

_DEFAULT_SYSTEM = "You are a helpful AI coding assistant."

_DEFAULT_MODELS: dict[str, str] = {
    "light":  "claude-sonnet-4-5-20251001",
    "medium": "claude-sonnet-4-5-20251001",
    "heavy":  "claude-opus-4-5",
    "sonnet": "claude-sonnet-4-5-20251001",
    "opus":   "claude-opus-4-5",
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

    def run(self, prompt: str, effort_or_model: str, timeout: int = 180) -> tuple[str, int]:
        """Call Anthropic API with streaming and cache_control: ephemeral on the system prompt."""
        try:
            import anthropic
        except ImportError:
            return "[ERROR] anthropic package not installed — run: pip install anthropic", 1

        cfg = config.caching_config()
        cfg_models = cfg.get("models", {})
        model = cfg_models.get(effort_or_model) or _DEFAULT_MODELS.get(effort_or_model, effort_or_model)

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        try:
            with TimingContext() as timer:
                output = ""
                with client.messages.stream(
                    model=model,
                    max_tokens=8096,
                    system=[
                        {
                            "type": "text",
                            "text": _DEFAULT_SYSTEM,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": prompt}],
                    timeout=timeout,
                ) as stream:
                    for text in stream.text_stream:
                        timer.mark_first_chunk()
                        output += text
            return output, 0
        except anthropic.APIStatusError as exc:
            return f"[ERROR] Anthropic API {exc.status_code}: {exc.message}", 1
        except anthropic.APIConnectionError as exc:
            return f"[ERROR] Anthropic connection: {exc}", 1
        except Exception as exc:  # noqa: BLE001
            return f"[ERROR] {exc}", 1

"""What a model call used and what it cost.

One place for the arithmetic: the provider's own cost when it reports one, else the tokens
it reports at the model's prices, else an estimate from the text (marked as one).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from orchestra import config

REPORTED = "reported"      # the provider gave token counts (or the cost itself)
ESTIMATED = "estimated"    # counted from the prompt and the answer text


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0            # the part of prompt_tokens read from a cache
    cost_usd: Optional[float] = None  # what the provider says the call cost, when it says
    source: str = ESTIMATED


def _estimate_tokens(text: str) -> int:
    from orchestra.engine.context_compression import estimate_tokens
    return estimate_tokens(text or "")


def _count(reported: dict, key: str) -> Optional[int]:
    value = reported.get(key)
    return int(value) if isinstance(value, (int, float)) else None


def usage_of(reported: Optional[dict], *, prompt: str, output: str) -> Usage:
    """The usage of one call: what the provider reported, the rest estimated from the text.

    A CLI that reports nothing leaves both sides estimated. The prompt estimate counts the
    text that was sent, not the files an agent went on to read, so it is a lower bound.
    """
    if not reported:
        return Usage(prompt_tokens=_estimate_tokens(prompt), completion_tokens=_estimate_tokens(output))
    completion = _count(reported, "completion_tokens")
    prompt_tokens = _count(reported, "prompt_tokens")
    cost = reported.get("cost_usd")
    return Usage(
        prompt_tokens=prompt_tokens if prompt_tokens is not None else _estimate_tokens(prompt),
        completion_tokens=completion if completion is not None else _estimate_tokens(output),
        cached_tokens=_count(reported, "cached_tokens") or 0,
        cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
        source=REPORTED,
    )


def cost_of(model_label: str, usage: Usage) -> float:
    """The cost of a call in USD."""
    if usage.cost_usd is not None:
        return round(usage.cost_usd, 6)
    price = config.price_of(model_label)
    cached = min(usage.cached_tokens, usage.prompt_tokens)
    cost = ((usage.prompt_tokens - cached) * price["input"]
            + cached * price["cached"]
            + usage.completion_tokens * price["output"]) / 1000.0
    return round(float(cost), 6)


def charge(agent, model_label: str, reported: Optional[dict], *, prompt: str, output: str) -> None:
    """Write what a call used and cost onto its agent record."""
    usage = usage_of(reported, prompt=prompt, output=output)
    agent.prompt_tokens = usage.prompt_tokens
    agent.cached_tokens = usage.cached_tokens
    agent.estimated_completion_tokens = usage.completion_tokens
    agent.estimated_cost_usd = cost_of(model_label, usage)
    agent.cost_source = usage.source

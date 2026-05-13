"""Auto-synthesis: combine agent outputs with adaptive model selection and dissent tracking."""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from orchestra import config
from orchestra.protocol import parse_envelope
from orchestra.models import AgentRun, OrchestraRun

_TEMPLATE = """Aşağıda aynı konuda iki farklı AI'ın bağımsız analizi var.
Bu iki perspektifi confidence ve kalite sinyaline göre sentezle.

Kurallar:
- Confidence >= 0.80 ise primary signal olarak değerlendir.
- Çatışmalarda daha yüksek confidence ve daha somut olanı tercih et.
- Sonuç kısa, net ve actionable olsun.

Çıktı formatı:
1. ## Answer
2. ## Key Signals
3. ## Dissent

--- AGENT 1 ({label_a}) | confidence={confidence_a:.2f} ---
{output_a}

--- AGENT 2 ({label_b}) | confidence={confidence_b:.2f} ---
{output_b}
"""

def _output_similarity(a: str, b: str) -> float:
    """Jaccard word-overlap similarity — quick proxy for output agreement."""
    if not a or not b: return 0.0
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b: return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def select_synthesis_alias(agent_a: AgentRun, agent_b: AgentRun, task_length: int) -> str:
    """Adaptive tier: prefer gmn-fast for simple/agreeing outputs, cld-deep for uncertain ones."""
    synthesis = config.synthesis_config()
    avg_confidence = (agent_a.confidence + agent_b.confidence) / 2
    total_words    = len((agent_a.stdout_log or "").split()) + len((agent_b.stdout_log or "").split())
    similarity     = _output_similarity(agent_a.stdout_log or "", agent_b.stdout_log or "")

    # Low confidence → deep synthesis
    if avg_confidence < float(synthesis.get("low_confidence_threshold", 0.60)):
        return "cld-deep"

    # High agreement → cheap synthesis sufficient
    if similarity > float(synthesis.get("high_similarity_threshold", 0.70)):
        return "gmn-fast"

    # Short + confident + simple task → fast
    if avg_confidence >= float(synthesis.get("fast_confidence_threshold", 0.80)) and total_words < int(synthesis.get("fast_total_words_threshold", 150)):
        return "gmn-fast"
    if task_length < int(synthesis.get("short_task_chars", 200)) and avg_confidence > float(synthesis.get("short_task_confidence_threshold", 0.75)):
        return "gmn-fast"

    # Long output or long prompt → analytical model
    if task_length > int(synthesis.get("long_task_chars", 1000)) or total_words > int(synthesis.get("long_output_words", 800)):
        return "gmn-pro"

    return "gmn-pro"

def build_synthesis_prompt(agent_a: AgentRun, agent_b: AgentRun) -> str:
    return _TEMPLATE.format(
        label_a=agent_a.model, confidence_a=agent_a.confidence, output_a=(agent_a.stdout_log or "").strip(),
        label_b=agent_b.model, confidence_b=agent_b.confidence, output_b=(agent_b.stdout_log or "").strip(),
    )

def parse_dissent(synthesis_output: str) -> Optional[str]:
    """Extract dissent section from synthesizer output."""
    match = re.search(r"## Dissent(.*?)(?:$|##)", synthesis_output, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None

def outputs_sufficient(agent_a: AgentRun, agent_b: AgentRun, min_chars: int = 100) -> bool:
    min_chars = int(config.synthesis_config().get("min_output_chars", min_chars))
    def _is_valid(agent: AgentRun) -> bool:
        body = (agent.stdout_log or "").strip()
        return len(body) >= min_chars and not agent.soft_failed
    return _is_valid(agent_a) and _is_valid(agent_b)

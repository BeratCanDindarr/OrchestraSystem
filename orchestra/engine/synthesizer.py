"""Auto-synthesis: combine agent outputs with adaptive model selection, dissent tracking, and provenance."""
from __future__ import annotations

import re
import math
from typing import List, Dict, Any, Optional
from orchestra import config
from orchestra.protocol import parse_envelope
from orchestra.models import AgentRun, AgentStatus, OrchestraRun

_TEMPLATE = """Below are independent analyses from multiple AI agents on the same topic.
Your task is to synthesize these perspectives into a coherent, actionable answer.

Rules:
- Treat confidence >= 0.80 as a strong primary signal.
- In conflicts, prefer higher confidence and more concrete evidence.
- If an agent FAILED or was CANCELLED, note the gap in the Failure Report.
- If agents disagree, preserve the disagreement honestly — do not average it away.
- Keep the final answer concise, clear, and actionable.

Output format:
1. ## Answer
2. ## Key Signals
3. ## Dissent
4. ## Failure Report
5. ## Provenance

{agent_blocks}
"""

_AGENT_BLOCK_OK = """--- AGENT {idx} ({alias}) | model={model} | confidence={confidence:.2f} | status=COMPLETED ---
{output}
"""

_AGENT_BLOCK_FAILED = """--- AGENT {idx} ({alias}) | model={model} | status={status} | failure={failure_kind} ---
Error: {error}
Partial output (if any): {output}
"""

_MAX_OUTPUT_CHARS = 4000  # Prevent context window blowout from huge partial logs


def _truncate(text: str | None, max_chars: int = _MAX_OUTPUT_CHARS) -> str:
    """Truncate long output to protect the synthesizer context window."""
    if not text:
        return ""
    text = text.strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def _output_similarity(a: str, b: str) -> float:
    """Jaccard word-overlap similarity — quick proxy for output agreement."""
    if not a or not b:
        return 0.0
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_divergence_index(agents: List[AgentRun]) -> float:
    """Compute a normalized divergence score (0.0 = full agreement, 1.0 = max divergence).

    Factors:
    - Pairwise Jaccard dissimilarity between completed agent outputs.
    - Failure ratio (failed / total agents).
    - Soft-failure ratio.
    """
    if not agents:
        return 0.0

    completed = [a for a in agents if a.status == AgentStatus.COMPLETED and not a.soft_failed and a.stdout_log]
    n_total = len(agents)
    n_completed = len(completed)
    n_failed = sum(1 for a in agents if a.status == AgentStatus.FAILED)
    n_soft = sum(1 for a in agents if a.soft_failed)

    # Textual dissimilarity among completed outputs
    text_dissimilarity = 0.0
    if n_completed >= 2:
        sims = []
        for i in range(n_completed):
            for j in range(i + 1, n_completed):
                sims.append(_output_similarity(completed[i].stdout_log, completed[j].stdout_log))
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        text_dissimilarity = 1.0 - avg_sim

    failure_penalty = (n_failed + n_soft * 0.5) / n_total if n_total > 0 else 0.0
    # Weighted combination: when failures exist, they should dominate the score.
    # If no completed outputs, divergence is purely driven by failures.
    if n_completed == 0:
        return round(min(1.0, failure_penalty), 3)
    if n_failed > 0 and n_completed < 2:
        # Single completed + failures: boost failure weight
        return round(min(1.0, text_dissimilarity * 0.4 + failure_penalty * 0.6), 3)
    return round(min(1.0, text_dissimilarity * 0.7 + failure_penalty * 0.3), 3)


def build_synthesis_prompt(agents: List[AgentRun]) -> str:
    """Build a synthesis prompt that includes ALL agents — completed and failed."""
    blocks: List[str] = []
    for idx, agent in enumerate(agents, start=1):
        if agent.status == AgentStatus.COMPLETED and not agent.soft_failed:
            blocks.append(
                _AGENT_BLOCK_OK.format(
                    idx=idx,
                    alias=agent.alias,
                    model=agent.model,
                    confidence=agent.confidence,
                    output=_truncate(agent.stdout_log),
                )
            )
        else:
            blocks.append(
                _AGENT_BLOCK_FAILED.format(
                    idx=idx,
                    alias=agent.alias,
                    model=agent.model,
                    status=agent.status.value,
                    failure_kind=getattr(agent, "failure_kind", None) or "unknown",
                    error=agent.error or "unknown error",
                    output=_truncate(agent.stdout_log) if agent.stdout_log else "<none>",
                )
            )
    return _TEMPLATE.format(agent_blocks="\n".join(blocks))


def select_synthesis_alias(agents: List[AgentRun], task_length: int) -> str:
    """Adaptive tier: prefer gmn-fast for simple/agreeing outputs, cld-deep for uncertain ones.

    In local-first mode, always use the configured local synthesis alias.
    """
    if config.local_first_enabled():
        return config.local_first_config().get("synthesis_alias", "oll-analyst")

    synthesis = config.synthesis_config()
    fast_alias = str(synthesis.get("fast_alias") or "gmn-fast")
    default_alias = str(synthesis.get("default_alias") or "gmn-pro")
    completed = [a for a in agents if a.status == AgentStatus.COMPLETED and not a.soft_failed and a.stdout_log]

    if not completed:
        # All failed — use deep reasoning to salvage whatever we can
        return "cld-deep"

    avg_confidence = sum(a.confidence for a in completed) / len(completed)
    total_words = sum(len((a.stdout_log or "").split()) for a in completed)
    similarity_matrix = []
    for i in range(len(completed)):
        for j in range(i + 1, len(completed)):
            similarity_matrix.append(_output_similarity(completed[i].stdout_log, completed[j].stdout_log))
    avg_similarity = sum(similarity_matrix) / len(similarity_matrix) if similarity_matrix else 0.0

    # Low confidence or failures present → deep synthesis
    failure_ratio = sum(1 for a in agents if a.status != AgentStatus.COMPLETED or a.soft_failed) / len(agents)
    if avg_confidence < float(synthesis.get("low_confidence_threshold", 0.60)) or failure_ratio > 0.3:
        return "cld-deep"

    # High agreement → cheap synthesis sufficient
    if avg_similarity > float(synthesis.get("high_similarity_threshold", 0.70)):
        return fast_alias

    # Short + confident + simple task → fast
    if (
        avg_confidence >= float(synthesis.get("fast_confidence_threshold", 0.80))
        and total_words < int(synthesis.get("fast_total_words_threshold", 150))
    ):
        return fast_alias
    if (
        task_length < int(synthesis.get("short_task_chars", 200))
        and avg_confidence > float(synthesis.get("short_task_confidence_threshold", 0.75))
    ):
        return fast_alias

    # Long output or long prompt → analytical model
    if task_length > int(synthesis.get("long_task_chars", 1000)) or total_words > int(synthesis.get("long_output_words", 800)):
        return default_alias

    return default_alias


def parse_dissent(synthesis_output: str) -> Optional[str]:
    """Extract dissent section from synthesizer output."""
    match = re.search(r"## Dissent(.*?)(?:$|##)", synthesis_output, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def parse_failure_report(synthesis_output: str) -> Optional[str]:
    """Extract failure report section from synthesizer output."""
    match = re.search(r"## Failure Report(.*?)(?:$|##)", synthesis_output, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def parse_provenance(synthesis_output: str) -> Optional[str]:
    """Extract provenance section from synthesizer output."""
    match = re.search(r"## Provenance(.*?)(?:$|##)", synthesis_output, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def measure_synthesis_quality(synthesis_output: str, divergence_index: float) -> dict:
    """Measure structural quality of synthesizer output.

    Returns a dict with boolean flags and a composite score (0.0-1.0).
    """
    output = (synthesis_output or "").strip()
    has_answer = bool(re.search(r"## Answer", output, re.IGNORECASE))
    has_key_signals = bool(re.search(r"## Key Signals", output, re.IGNORECASE))
    has_dissent = bool(re.search(r"## Dissent", output, re.IGNORECASE))
    has_failure_report = bool(re.search(r"## Failure Report", output, re.IGNORECASE))
    has_provenance = bool(re.search(r"## Provenance", output, re.IGNORECASE))
    length = len(output)

    # Composite score: each required section contributes; length bonus capped
    score = 0.0
    if has_answer:
        score += 0.20
    if has_key_signals:
        score += 0.15
    if has_dissent:
        score += 0.15
    if has_failure_report:
        score += 0.15
    if has_provenance:
        score += 0.15
    # Length bonus: 0.20 max for outputs > 500 chars
    score += min(0.20, length / 2500.0)

    # Penalize if divergence is high but dissent/provenance are missing
    if divergence_index > 0.5 and not (has_dissent and has_provenance):
        score -= 0.15

    return {
        "score": round(max(0.0, min(1.0, score)), 3),
        "length": length,
        "has_answer": has_answer,
        "has_key_signals": has_key_signals,
        "has_dissent": has_dissent,
        "has_failure_report": has_failure_report,
        "has_provenance": has_provenance,
        "divergence_index": divergence_index,
    }


def outputs_sufficient(agents: List[AgentRun], min_chars: int = 100) -> bool:
    """Check whether there are enough viable outputs to attempt synthesis.

    Synthesis proceeds if EITHER:
    - At least 2 agents have completed, non-soft-failed outputs meeting min_chars, OR
    - At least 2 agents exist (any status) so a failure-report synthesis can be attempted.
    """
    min_chars = int(config.synthesis_config().get("min_output_chars", min_chars))
    valid_completed = [
        a
        for a in agents
        if a.status == AgentStatus.COMPLETED
        and not a.soft_failed
        and len((a.stdout_log or "").strip()) >= min_chars
    ]
    if len(valid_completed) >= 2:
        return True
    # Allow synthesis when all agents failed so we can generate a failure report
    if len(agents) >= 2:
        return True
    return False


def format_provenance(run: OrchestraRun, divergence_index: float) -> str:
    """Format a machine-readable provenance graph for the run."""
    lines = [
        "### Provenance Graph",
        f"- divergence_index: {divergence_index}",
        f"- total_agents: {len(run.agents)}",
        f"- completed: {sum(1 for a in run.agents if a.status == AgentStatus.COMPLETED)}",
        f"- failed: {sum(1 for a in run.agents if a.status == AgentStatus.FAILED)}",
        f"- cancelled: {sum(1 for a in run.agents if a.status == AgentStatus.CANCELLED)}",
        f"- soft_failed: {sum(1 for a in run.agents if a.soft_failed)}",
        "",
        "| Agent | Model | Status | Confidence | Failure Kind |",
        "|-------|-------|--------|------------|--------------|",
    ]
    for a in run.agents:
        lines.append(
            f"| {a.alias} | {a.model} | {a.status.value} | {a.confidence:.2f} | {getattr(a, 'failure_kind', None) or '-'} |"
        )
    return "\n".join(lines)

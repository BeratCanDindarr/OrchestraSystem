"""Normalize raw agent output into standard Orchestra body sections."""
from __future__ import annotations

import re
from orchestra.protocol import OrchestraMeta, wrap_output


# Headers we look for in raw output to extract confidence signal
_CONFIDENCE_KEYWORDS = {
    "high": 0.85,
    "medium": 0.60,
    "low": 0.35,
}

_SOFT_FAILURE_MARKERS = (
    "i cannot help",
    "i can't help",
    "i am unable to help",
    "i'm unable to help",
    "i am unable to comply",
    "i'm unable to comply",
    "i do not have access",
    "i don't have access",
    "as an ai",
    "not enough context",
    "need more context",
    "need more information",
    "cannot comply with this request",
    "unable to comply with this request",
)

# Section headers to look for in brain-storming outputs
_SECTION_PATTERNS = [
    ("ÖNCE YAPILACAKLAR", "## Önce Yapılacaklar"),
    ("ÖNCELIK 1", "## Öncelik 1"),
    ("GÖZDEN KAÇANLAR", "## Gözden Kaçanlar"),
    ("RİSKLER", "## Riskler"),
    ("DIAGNOSIS", "## Diagnosis"),
    ("BUGFIX RESULT", "## Bugfix Result"),
    ("UNITY VERIFY", "## Unity Verify"),
]


def infer_confidence(text: str) -> float:
    """Heuristic: look for explicit confidence keywords."""
    text_lower = text.lower()
    for kw, score in _CONFIDENCE_KEYWORDS.items():
        if f"confidence: {kw}" in text_lower or f"confidence:{kw}" in text_lower:
            return score
    if is_soft_failure(text):
        return 0.2
    return 0.5  # default neutral


def is_soft_failure(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(r"\b" + re.escape(marker) + r"\b", text_lower) for marker in _SOFT_FAILURE_MARKERS)


def infer_summary(text: str, max_chars: int = 120) -> str:
    """Extract first meaningful sentence as summary."""
    # Skip blank lines and section markers
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("="):
            continue
        if len(line) >= 20:
            return line[:max_chars]
    return text[:max_chars].strip()


def normalize(raw: str, provider: str = "") -> str:
    """
    Normalize raw agent output.
    - Adds ORCH_META + ORCH_BODY envelope
    - Infers confidence and summary
    """
    soft_failed = is_soft_failure(raw)
    confidence = infer_confidence(raw)
    summary = infer_summary(raw)

    # Count evidence markers (bullet points, numbered lists)
    evidence_count = len(re.findall(r"^\s*[-*•]|\d+\.", raw, re.MULTILINE))

    meta = OrchestraMeta(
        status="partial" if soft_failed else "ok",
        summary=summary,
        confidence=confidence,
        evidence_count=min(evidence_count, 99),
        verifier_result="soft_fail" if soft_failed else "not_run",
    )

    return wrap_output(meta, raw)

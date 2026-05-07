"""Orchestra standard output envelope: ORCH_META + ORCH_BODY."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional


META_BEGIN = "=== ORCH_META_BEGIN ==="
META_END = "=== ORCH_META_END ==="
BODY_BEGIN = "=== ORCH_BODY_BEGIN ==="
BODY_END = "=== ORCH_BODY_END ==="
HANDOFF_BEGIN = "=== ORCH_HANDOFF_BEGIN ==="
HANDOFF_END = "=== ORCH_HANDOFF_END ==="

SCHEMA_VERSION = "orchestra.v1"


@dataclass
class OrchestraMeta:
    status: str = "ok"          # "ok" | "error" | "partial"
    summary: str = ""
    confidence: float = 0.0     # 0.0–1.0
    evidence_count: int = 0
    verifier_result: str = "not_run"   # "passed" | "failed" | "soft_fail" | "not_run"
    schema: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "status": self.status,
            "summary": self.summary,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "verifier_result": self.verifier_result,
        }

    def to_envelope_header(self) -> str:
        return f"{META_BEGIN}\n{json.dumps(self.to_dict())}\n{META_END}"


@dataclass
class HandoffEnvelope:
    stage: str = "analysis"
    summary: str = ""
    next_action: str = ""
    needs_approval: bool = False
    risks: list[str] = field(default_factory=list)
    owner: str = "orchestra"

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "summary": self.summary,
            "next_action": self.next_action,
            "needs_approval": self.needs_approval,
            "risks": self.risks,
            "owner": self.owner,
        }

    def to_block(self) -> str:
        return f"{HANDOFF_BEGIN}\n{json.dumps(self.to_dict(), ensure_ascii=False)}\n{HANDOFF_END}"


def wrap_output(meta: OrchestraMeta, body: str) -> str:
    """Wrap agent output in standard Orchestra envelope."""
    return (
        f"{meta.to_envelope_header()}\n"
        f"{BODY_BEGIN}\n{body.strip()}\n{BODY_END}"
    )


def attach_handoff(raw: str, handoff: HandoffEnvelope) -> str:
    """Append a structured handoff block to an existing Orchestra payload."""
    return f"{raw.rstrip()}\n{handoff.to_block()}\n"


def parse_envelope(raw: str) -> tuple[Optional[OrchestraMeta], str]:
    """
    Parse raw string into (OrchestraMeta | None, body_str).
    If envelope markers are missing, body = raw, meta = None.
    """
    meta = None
    body = raw

    meta_match = re.search(
        re.escape(META_BEGIN) + r"\s*(.*?)\s*" + re.escape(META_END),
        raw, re.DOTALL
    )
    if meta_match:
        try:
            data = json.loads(meta_match.group(1))
            meta = OrchestraMeta(**{k: v for k, v in data.items() if k in OrchestraMeta.__dataclass_fields__})
        except (json.JSONDecodeError, TypeError):
            pass

    body_match = re.search(
        re.escape(BODY_BEGIN) + r"\s*(.*?)\s*" + re.escape(BODY_END),
        raw, re.DOTALL
    )
    if body_match:
        body = body_match.group(1)

    return meta, body


def parse_handoff(raw: str) -> Optional[HandoffEnvelope]:
    handoff_match = re.search(
        re.escape(HANDOFF_BEGIN) + r"\s*(.*?)\s*" + re.escape(HANDOFF_END),
        raw,
        re.DOTALL,
    )
    if not handoff_match:
        return None
    try:
        data = json.loads(handoff_match.group(1))
        return HandoffEnvelope(
            stage=data.get("stage", "analysis"),
            summary=data.get("summary", ""),
            next_action=data.get("next_action", ""),
            needs_approval=bool(data.get("needs_approval", False)),
            risks=list(data.get("risks", [])),
            owner=data.get("owner", "orchestra"),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None

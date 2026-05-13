"""Typed runtime state helpers for Orchestra."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Optional


class FailureKind(str, Enum):
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_SCHEMA_INVALID = "MODEL_SCHEMA_INVALID"
    MODEL_SOFT_FAILURE = "MODEL_SOFT_FAILURE"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    TOOL_POLICY_DENIED = "TOOL_POLICY_DENIED"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    HANDOFF_INVALID = "HANDOFF_INVALID"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    APPROVAL_TIMEOUT = "APPROVAL_TIMEOUT"
    CONSENSUS_FAILED = "CONSENSUS_FAILED"
    UNKNOWN_RUNTIME_ERROR = "UNKNOWN_RUNTIME_ERROR"


class ApprovalState(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    RESUMED = "resumed"


class InterruptState(str, Enum):
    IDLE = "idle"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"


@dataclass
class HandoffEnvelope:
    stage: str = "analysis"
    summary: str = ""
    next_action: str = ""
    needs_approval: bool = False
    risks: list[str] = field(default_factory=list)
    owner: str = "orchestra"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["risks"] = list(self.risks or [])
        return payload

    @classmethod
    def from_dict(cls, data: dict | None) -> "HandoffEnvelope | None":
        if not data:
            return None
        return cls(
            stage=data.get("stage", "analysis"),
            summary=data.get("summary", ""),
            next_action=data.get("next_action", ""),
            needs_approval=bool(data.get("needs_approval", False)),
            risks=list(data.get("risks", []) or []),
            owner=data.get("owner", "orchestra"),
        )


@dataclass
class FailureState:
    kind: str
    message: str
    retryable: bool = False
    source: str = "runtime"
    agent_alias: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "FailureState | None":
        if not data:
            return None
        return cls(
            kind=data.get("kind", FailureKind.UNKNOWN_RUNTIME_ERROR.value),
            message=data.get("message", ""),
            retryable=bool(data.get("retryable", False)),
            source=data.get("source", "runtime"),
            agent_alias=data.get("agent_alias"),
        )


@dataclass
class RunStateSnapshot:
    schema_version: int
    checkpoint_version: int
    label: str
    run_id: str
    mode: str
    status: str
    turns: int
    total_cost_usd: float
    avg_confidence: float
    approval_state: ApprovalState
    interrupt_state: InterruptState
    failure: dict | None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["approval_state"] = self.approval_state.value
        payload["interrupt_state"] = self.interrupt_state.value
        return payload

"""Typed runtime state helpers for Orchestra."""
from __future__ import annotations

from dataclasses import dataclass, asdict
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
    approval_state: str
    interrupt_state: str
    failure: dict | None

    def to_dict(self) -> dict:
        return asdict(self)

"""Data models for Orchestra runs and artifacts."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from orchestra.state import ApprovalState, FailureState, HandoffEnvelope, InterruptState


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_APPROVAL = "waiting_approval"


class AgentStatus(str, Enum):
    QUEUED = "queued"
    STARTED = "started"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentRun:
    alias: str              # e.g. "cdx-deep"
    provider: str           # e.g. "codex"
    model: str              # e.g. "gpt-5.4/xhigh"
    status: AgentStatus = AgentStatus.QUEUED
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    stdout_log: str = ""
    error: Optional[str] = None
    pid: Optional[int] = None
    estimated_completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    confidence: float = 0.5
    soft_failed: bool = False
    validation_status: str = "not_run"
    validation_reason: str = ""

    @property
    def elapsed(self) -> str:
        if not self.start_time:
            return "--:--"
        start = datetime.fromisoformat(self.start_time)
        end = datetime.fromisoformat(self.end_time) if self.end_time else datetime.now(timezone.utc)
        diff = int((end - start).total_seconds())
        return f"{diff // 60:02d}:{diff % 60:02d}"

    @property
    def last_line(self) -> str:
        lines = [l.strip() for l in self.stdout_log.splitlines() if l.strip()]
        return lines[-1][:60] if lines else ""

    @classmethod
    def from_manifest(cls, data: dict) -> "AgentRun":
        return cls(
            alias=data.get("alias", ""),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            status=AgentStatus(data.get("status", AgentStatus.QUEUED.value)),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            error=data.get("error"),
            pid=data.get("pid"),
            estimated_completion_tokens=data.get("estimated_completion_tokens", 0),
            estimated_cost_usd=data.get("estimated_cost_usd", 0.0),
            confidence=data.get("confidence", 0.5),
            soft_failed=data.get("soft_failed", False),
            validation_status=data.get("validation_status", "not_run"),
            validation_reason=data.get("validation_reason", ""),
        )


@dataclass
class OrchestraRun:
    mode: str               # "ask", "dual", "critical"
    task: str               # user prompt
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: RunStatus = RunStatus.QUEUED
    agents: list[AgentRun] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: Optional[str] = None
    summary: Optional[str] = None
    prompt_id: Optional[str] = None
    prompt_version: Optional[str] = None
    # Circuit breaker fields
    turns: int = 0          # how many run_parallel calls have been made
    max_turns: Optional[int] = None  # None = use config default
    total_cost_usd: float = 0.0
    avg_confidence: float = 0.0
    reviews: list[dict] = field(default_factory=list)
    latest_review_stage: str = ""
    latest_review_status: str = "not_run"
    latest_review_winner: str = ""
    latest_review_reason: str = ""
    latest_handoff: Optional[HandoffEnvelope] = None
    approval_state: ApprovalState = ApprovalState.NOT_REQUIRED
    interrupt_state: InterruptState = InterruptState.IDLE
    schema_version: int = 2
    checkpoint_version: int = 0
    last_checkpoint_at: Optional[str] = None
    failure: Optional[FailureState] = None

    def to_manifest(self) -> dict:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "task": self.task,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "agents": [
                {
                    "alias": a.alias,
                    "provider": a.provider,
                    "model": a.model,
                    "status": a.status.value,
                    "start_time": a.start_time,
                    "end_time": a.end_time,
                    "elapsed": a.elapsed,
                    "error": a.error,
                    "pid": a.pid,
                    "estimated_completion_tokens": a.estimated_completion_tokens,
                    "estimated_cost_usd": a.estimated_cost_usd,
                    "confidence": a.confidence,
                    "soft_failed": a.soft_failed,
                    "validation_status": a.validation_status,
                    "validation_reason": a.validation_reason,
                }
                for a in self.agents
            ],
            "summary": self.summary,
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "turns": self.turns,
            "max_turns": self.max_turns,
            "total_cost_usd": self.total_cost_usd,
            "avg_confidence": self.avg_confidence,
            "reviews": self.reviews,
            "latest_review_stage": self.latest_review_stage,
            "latest_review_status": self.latest_review_status,
            "latest_review_winner": self.latest_review_winner,
            "latest_review_reason": self.latest_review_reason,
            "latest_handoff": self.latest_handoff.to_dict() if self.latest_handoff else None,
            "approval_state": self.approval_state.value,
            "interrupt_state": self.interrupt_state.value,
            "schema_version": self.schema_version,
            "checkpoint_version": self.checkpoint_version,
            "last_checkpoint_at": self.last_checkpoint_at,
            "failure": self.failure.to_dict() if self.failure else None,
        }

    @classmethod
    def from_manifest(cls, data: dict) -> "OrchestraRun":
        return cls(
            mode=data.get("mode", "ask"),
            task=data.get("task", ""),
            run_id=data.get("run_id", uuid.uuid4().hex[:8]),
            status=RunStatus(data.get("status", RunStatus.QUEUED.value)),
            agents=[AgentRun.from_manifest(agent) for agent in data.get("agents", [])],
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at"),
            summary=data.get("summary"),
            prompt_id=data.get("prompt_id"),
            prompt_version=data.get("prompt_version"),
            turns=data.get("turns", 0),
            max_turns=data.get("max_turns"),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            avg_confidence=data.get("avg_confidence", 0.0),
            reviews=data.get("reviews", []),
            latest_review_stage=data.get("latest_review_stage", ""),
            latest_review_status=data.get("latest_review_status", "not_run"),
            latest_review_winner=data.get("latest_review_winner", ""),
            latest_review_reason=data.get("latest_review_reason", ""),
            latest_handoff=HandoffEnvelope.from_dict(data.get("latest_handoff")),
            approval_state=ApprovalState(data.get("approval_state", ApprovalState.NOT_REQUIRED.value)),
            interrupt_state=InterruptState(data.get("interrupt_state", InterruptState.IDLE.value)),
            schema_version=data.get("schema_version", 2),
            checkpoint_version=data.get("checkpoint_version", 0),
            last_checkpoint_at=data.get("last_checkpoint_at"),
            failure=FailureState.from_dict(data.get("failure")),
        )

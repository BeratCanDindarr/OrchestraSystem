"""Lightweight tracing and span management for Orchestra."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from orchestra.engine import artifacts

@dataclass
class Span:
    name: str
    run_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "started"  # started, completed, failed

    def finish(self, status: str = "completed", metadata: Optional[Dict[str, Any]] = None):
        self.end_time = time.time()
        self.status = status
        if metadata:
            self.metadata.update(metadata)
        self._record()

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000

    def _record(self):
        event = {
            "event": "span_finished",
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "metadata": self.metadata,
            "ts": datetime.now(timezone.utc).isoformat()
        }
        artifacts.append_event(self.run_id, event)

class Tracer:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.active_spans: Dict[str, Span] = {}

    def start_span(self, name: str, parent_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Span:
        span = Span(name=name, run_id=self.run_id, parent_id=parent_id, metadata=metadata or {})
        self.active_spans[span.span_id] = span
        return span

def get_tracer(run_id: str) -> Tracer:
    return Tracer(run_id)

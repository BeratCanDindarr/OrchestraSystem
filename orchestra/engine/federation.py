"""Federation data model and bus for cross-instance semantic memory sharing."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orchestra import config
from orchestra.engine.federation_crdt import FederatedState

logger = logging.getLogger(__name__)


@dataclass
class FederationSummary:
    """Anonymized, content-addressed summary ready for federation."""

    content_hash: str
    embedding_digest: str
    metadata: dict[str, Any] = field(default_factory=dict)
    crdt_state: FederatedState = field(default_factory=FederatedState)
    timestamp: float = field(default_factory=time.time)
    ttl_hours: int = 720
    peer_id: str = ""

    def to_dict(self) -> dict:
        return {
            "content_hash": self.content_hash,
            "embedding_digest": self.embedding_digest,
            "metadata": self.metadata,
            "crdt_state": self.crdt_state.to_dict(),
            "timestamp": self.timestamp,
            "ttl_hours": self.ttl_hours,
            "peer_id": self.peer_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FederationSummary":
        return cls(
            content_hash=data["content_hash"],
            embedding_digest=data["embedding_digest"],
            metadata=data.get("metadata", {}),
            crdt_state=FederatedState.from_dict(data.get("crdt_state", {})),
            timestamp=data.get("timestamp", 0.0),
            ttl_hours=data.get("ttl_hours", 720),
            peer_id=data.get("peer_id", ""),
        )

    def compute_hash(self) -> str:
        """Re-compute SHA-256 of the canonical JSON representation."""
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class FederationPeer:
    """Remote Orchestra instance we federate with."""

    peer_id: str
    endpoint_url: str
    auth_token: str = ""
    last_sync_at: float = 0.0
    trust_score: float = 50.0
    token_issued_at: float = 0.0
    token_expires_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "peer_id": self.peer_id,
            "endpoint_url": self.endpoint_url,
            "auth_token": self.auth_token,
            "last_sync_at": self.last_sync_at,
            "trust_score": self.trust_score,
            "token_issued_at": self.token_issued_at,
            "token_expires_at": self.token_expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FederationPeer":
        return cls(
            peer_id=data["peer_id"],
            endpoint_url=data["endpoint_url"],
            auth_token=data.get("auth_token", ""),
            last_sync_at=data.get("last_sync_at", 0.0),
            trust_score=data.get("trust_score", 50.0),
            token_issued_at=data.get("token_issued_at", 0.0),
            token_expires_at=data.get("token_expires_at", 0.0),
        )

    def is_token_expired(self) -> bool:
        """Return True if the auth token has a TTL and has expired."""
        if self.token_expires_at <= 0.0:
            return False
        return time.time() > self.token_expires_at

    def needs_refresh(self, now: float, margin_seconds: float) -> bool:
        """True when a TTL'd token is within ``margin_seconds`` of expiry.

        Signals the operator to rotate the shared secret in config before it expires
        (``now`` is a parameter so the check is deterministic in tests).
        """
        if self.token_expires_at <= 0.0:
            return False
        return now > self.token_expires_at - margin_seconds


class FederationBus:
    """Manages outbox/inbox queues for federated summaries."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (config.orchestra_root() / "federation")
        self.outbox = self.root / "outbox"
        self.inbox = self.root / "inbox"
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.inbox.mkdir(parents=True, exist_ok=True)

    def enqueue_outbox(self, summary: FederationSummary) -> Path:
        """Write a summary to the outbox. Idempotent by content hash."""
        filename = f"{summary.content_hash}.json"
        path = self.outbox / filename
        path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def dequeue_outbox(self) -> list[FederationSummary]:
        """Read all pending summaries from the outbox."""
        summaries: list[FederationSummary] = []
        for path in sorted(self.outbox.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                summaries.append(FederationSummary.from_dict(data))
            except Exception as exc:
                logger.warning("Failed to load outbox summary %s: %s", path, exc)
        return summaries

    def clear_outbox(self, content_hashes: list[str]) -> None:
        """Remove successfully synced summaries from the outbox."""
        for h in content_hashes:
            path = self.outbox / f"{h}.json"
            if path.exists():
                path.unlink()

    def write_inbox(self, summary: FederationSummary) -> Path:
        """Write a received summary to the inbox."""
        filename = f"{summary.content_hash}.json"
        path = self.inbox / filename
        path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def read_inbox(self) -> list[FederationSummary]:
        """Read all summaries from the inbox."""
        summaries: list[FederationSummary] = []
        for path in sorted(self.inbox.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                summaries.append(FederationSummary.from_dict(data))
            except Exception as exc:
                logger.warning("Failed to load inbox summary %s: %s", path, exc)
        return summaries

    def inbox_hashes(self) -> set[str]:
        return {p.stem for p in self.inbox.glob("*.json")}

    def outbox_hashes(self) -> set[str]:
        return {p.stem for p in self.outbox.glob("*.json")}

    def cleanup_expired(self, now: float) -> int:
        """Remove inbox/outbox summaries past their TTL. Returns the count removed.

        A summary expires when ``now > timestamp + ttl_hours * 3600``. ``now`` is a
        parameter (not ``time.time()``) so the sweep is deterministic in tests.
        """
        removed = 0
        for box in (self.outbox, self.inbox):
            for path in box.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    expiry = float(data.get("timestamp", 0.0)) + float(data.get("ttl_hours", 720)) * 3600
                    if now > expiry:
                        path.unlink()
                        removed += 1
                except Exception:
                    logger.warning("cleanup_expired: skipping bad summary %s", path, exc_info=True)
        return removed

    def stats(self) -> dict:
        return {
            "outbox_count": len(list(self.outbox.glob("*.json"))),
            "inbox_count": len(list(self.inbox.glob("*.json"))),
            "root": str(self.root),
        }

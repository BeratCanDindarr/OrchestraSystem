"""Anonymized summary generator for federation.

Strips all PII and raw stdout before export. Only task hash,
embedding digest, agent metadata, and confidence scores leave
the local instance.
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING

from orchestra.engine.federation import FederationSummary
from orchestra.engine.federation_crdt import FederatedState, GCounter, LWWRegister, ORSet

if TYPE_CHECKING:
    from orchestra.models import OrchestraRun

logger = logging.getLogger(__name__)


def _task_hash(task: str) -> str:
    return hashlib.sha256(task.encode("utf-8")).hexdigest()[:16]


def _model_family(alias: str) -> str:
    mapping = {
        "cdx": "codex",
        "gmn": "gemini",
        "cld": "claude",
        "oll": "ollama",
        "ds": "deepseek",
        "kimi": "kimi",
    }
    return mapping.get(alias.split("-")[0], "unknown")


def summarize_run(run: "OrchestraRun", peer_id: str = "") -> FederationSummary | None:
    """Create an anonymized federation summary from a completed run.

    Returns None if the run is not suitable for federation (e.g. failed,
    empty agents, or no stdout).
    """
    if not run.agents:
        return None

    completed = [a for a in run.agents if a.status.value == "completed"]
    if not completed:
        return None

    # Build anonymized metadata
    task = (run.task or "").strip()
    if not task:
        return None

    # Scrub task text before hashing to avoid leaking sensitive keywords
    from orchestra.engine.pii_scrubber import PiiScrubber

    scrubber = PiiScrubber()
    scrubbed_task = scrubber.scrub_text(task)

    agent_meta = []
    families: set[str] = set()
    for agent in completed:
        families.add(_model_family(agent.alias))
        agent_meta.append(
            {
                "alias": agent.alias,
                "family": _model_family(agent.alias),
                "confidence": round(agent.confidence, 3),
                "model": agent.model,
            }
        )

    # Embedding digest: hash of concatenated agent families + confidence
    embed_source = "|".join(sorted(f"{a['family']}:{a['confidence']}" for a in agent_meta))
    embedding_digest = hashlib.sha256(embed_source.encode("utf-8")).hexdigest()[:16]

    # CRDT state
    tags = ORSet[str]()
    for f in families:
        tags.add(f)
    tags.add(run.mode)

    counter = GCounter()
    counter.increment(peer_id or "local", len(completed))

    content = LWWRegister(scrubbed_task)
    content.set(scrubbed_task, time.time(), peer_id or "local")

    metadata = {
        "task_hash": _task_hash(scrubbed_task),
        "mode": run.mode,
        "agent_count": len(completed),
        "agents": agent_meta,
        "consensus_challenged": run.consensus_challenged,
    }

    summary = FederationSummary(
        content_hash=_task_hash(scrubbed_task),
        embedding_digest=embedding_digest,
        metadata=metadata,
        crdt_state=FederatedState(content=content, counter=counter, tags=tags),
        peer_id=peer_id or "local",
    )

    # Re-compute content_hash from canonical representation for integrity
    summary.content_hash = summary.compute_hash()[:16]
    return summary

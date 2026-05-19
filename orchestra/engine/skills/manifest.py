"""Skill manifest schema and validation."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

logger = logging.getLogger(__name__)

TIER = Literal["core", "user", "marketplace"]
POLICY_ACTION = Literal["auto", "gate", "deny"]


@dataclass
class SkillManifest:
    """Declarative description of a skill."""

    name: str
    version: str
    tier: TIER
    description: str = ""
    entrypoint: str = ""
    signature: str = ""  # SHA-256 of entrypoint file content
    policy: POLICY_ACTION = "auto"
    author: str = ""
    tags: list[str] = field(default_factory=list)
    reputation_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "tier": self.tier,
            "description": self.description,
            "entrypoint": self.entrypoint,
            "signature": self.signature,
            "policy": self.policy,
            "author": self.author,
            "tags": self.tags,
            "reputation_score": self.reputation_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SkillManifest":
        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            tier=data.get("tier", "user"),
            description=data.get("description", ""),
            entrypoint=data.get("entrypoint", ""),
            signature=data.get("signature", ""),
            policy=data.get("policy", "auto"),
            author=data.get("author", ""),
            tags=data.get("tags", []),
            reputation_score=float(data.get("reputation_score", 0.0)),
        )


def _compute_signature(path: Path) -> str:
    """Return SHA-256 hex digest of file content."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_manifest(path: Path) -> SkillManifest | None:
    """Load and validate a manifest from a YAML file."""
    if yaml is None:
        logger.warning("PyYAML not installed; cannot load skill manifest %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Failed to load manifest %s: %s", path, exc)
        return None

    manifest = SkillManifest.from_dict(data)

    # Validate signature if entrypoint is declared
    if manifest.entrypoint:
        entry_path = path.parent / manifest.entrypoint
        if entry_path.exists():
            expected = _compute_signature(entry_path)
            if manifest.signature and manifest.signature != expected:
                logger.warning(
                    "Signature mismatch for skill '%s' (%s vs expected %s); skipping",
                    manifest.name,
                    manifest.signature[:16],
                    expected[:16],
                )
                return None
            if not manifest.signature:
                # Auto-populate signature on first load
                manifest.signature = expected
        else:
            logger.warning(
                "Entrypoint '%s' for skill '%s' not found; skipping",
                manifest.entrypoint,
                manifest.name,
            )
            return None

    return manifest


def save_manifest(path: Path, manifest: SkillManifest) -> None:
    """Persist a manifest to a YAML file."""
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required to save manifests")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest.to_dict(), f, sort_keys=False, allow_unicode=True)

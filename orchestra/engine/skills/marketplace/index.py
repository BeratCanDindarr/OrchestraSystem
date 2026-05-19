"""Static marketplace index loader."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from orchestra.engine.skills.manifest import SkillManifest

logger = logging.getLogger(__name__)


def _index_path() -> Path:
    return Path(__file__).parent / "index.json"


def load_index() -> dict[str, SkillManifest]:
    """Load marketplace skill manifests from the static index."""
    path = _index_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load marketplace index: %s", exc)
        return {}

    manifests: dict[str, SkillManifest] = {}
    for item in raw.get("skills", []):
        try:
            manifest = SkillManifest.from_dict(item)
            manifest.tier = "marketplace"  # type: ignore[assignment]
            manifests[manifest.name] = manifest
        except Exception as exc:
            logger.warning("Skipping malformed marketplace skill entry: %s", exc)
    return manifests


def save_index(manifests: dict[str, SkillManifest]) -> None:
    """Persist marketplace index (for curation ops)."""
    path = _index_path()
    data = {"skills": [m.to_dict() for m in manifests.values()]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

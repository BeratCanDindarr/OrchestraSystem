"""Skills for Orchestra's agents: folders with a SKILL.md, the format Claude, Kimi and Gemini share.

They live in ~/.orchestra/skills (every project) and <project>/.orchestra/skills (this one; a skill
of the same name wins). Orchestra links them into one bundle and gives it to each CLI through
its own flag: Claude `--plugin-dir`, Kimi `--skills-dir`, Antigravity `--add-dir` (it reads
`.agents/skills`). A skill only adds instructions; what an agent may do is unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

PLUGIN_NAME = "orchestra-skills"


def _global_dir() -> Path:
    return Path.home() / ".orchestra" / "skills"


def _project_dir() -> Path:
    from orchestra import config

    return config.project_root() / ".orchestra" / "skills"


def sources() -> dict[str, Path]:
    """Skill name -> its folder; a project skill replaces a user skill of the same name."""
    found: dict[str, Path] = {}
    for root in (_global_dir(), _project_dir()):
        if root.is_dir():
            for folder in sorted(root.iterdir()):
                if (folder / "SKILL.md").is_file():
                    found[folder.name] = folder.resolve()
    return found


def bundle() -> Optional[Path]:
    """The folder every CLI is given, or None when there is no skill.

    Named by the skills it links, so the same skills reuse one bundle and concurrent agents
    never build into the same folder; a changed SKILL.md is seen at once through the link.
    """
    found = sources()
    if not found:
        return None
    from orchestra import config

    key = hashlib.sha256(json.dumps({k: str(v) for k, v in found.items()}, sort_keys=True).encode()).hexdigest()[:12]
    parent = config.orchestra_root() / "skills-bundle"
    root = parent / key
    if root.is_dir():
        return root
    parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(dir=parent, prefix=".building-"))
    (building / ".claude-plugin").mkdir()
    (building / ".claude-plugin" / "plugin.json").write_text(json.dumps(
        {"name": PLUGIN_NAME, "version": "1.0.0", "description": "Skills Orchestra gives its agents"}))
    (building / "skills").mkdir()
    for name, folder in found.items():
        (building / "skills" / name).symlink_to(folder, target_is_directory=True)
    (building / ".agents").mkdir()
    (building / ".agents" / "skills").symlink_to(Path("..") / "skills", target_is_directory=True)
    try:
        os.rename(building, root)
    except OSError:   # another agent built it first: theirs links the same skills
        shutil.rmtree(building, ignore_errors=True)
    return root


def listing() -> list[tuple[str, str]]:
    """(name, description) of every skill, for prompts that name them (the code review)."""
    out = []
    for name, folder in sorted(sources().items()):
        head = (folder / "SKILL.md").read_text(encoding="utf-8", errors="replace").split("\n---", 1)[0]
        described = re.search(r"^description:\s*(.+)$", head, re.M)
        out.append((name, described.group(1).strip().strip("\"'") if described else ""))
    return out

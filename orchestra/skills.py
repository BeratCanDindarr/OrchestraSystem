"""The skills a code review names as the project's rules.

Every agent CLI loads skills (folders with a SKILL.md) by itself: Claude from the project's
.claude/skills and from each `--add-dir` folder's .claude/skills, Kimi from .agents/skills and
~/.kimi-code/skills, Antigravity from .agents/skills of its workspace. Orchestra gives them nothing
but those flags (config `extra_args`). A review also lists them by name, since with the rules in
its prompt it finds their violations (Qodo benchmark run: F1 53% -> 65%).
"""
from __future__ import annotations

import re
from pathlib import Path


def _dirs() -> list[Path]:
    """Where Claude agents find skills: this project's folders and their --add-dir folders."""
    from orchestra import config

    root = config.project_root()
    found = [root / ".claude" / "skills", root / ".agents" / "skills"]
    args = [str(a) for a in config.provider_config("claude").get("extra_args") or []]
    found += [Path(args[i + 1]).expanduser() / ".claude" / "skills"
              for i, arg in enumerate(args[:-1]) if arg == "--add-dir"]
    return found


def listing() -> list[tuple[str, str]]:
    """(name, description) of every skill, the first folder that has a name winning."""
    out: dict[str, str] = {}
    for folder in _dirs():
        if not folder.is_dir():
            continue
        for skill in sorted(folder.iterdir()):
            text = skill / "SKILL.md"
            if skill.name in out or not text.is_file():
                continue
            head = text.read_text(encoding="utf-8", errors="replace").split("\n---", 1)[0]
            described = re.search(r"^description:\s*(.+)$", head, re.M)
            out[skill.name] = described.group(1).strip().strip("\"'") if described else ""
    return sorted(out.items())

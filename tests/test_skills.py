"""Each agent CLI loads skills by itself; Orchestra only names them to a code review."""
from __future__ import annotations

from pathlib import Path

from orchestra import skills

_real_dirs = skills._dirs   # conftest replaces it with an empty list in every test


def _skill(root: Path, name: str, rule: str) -> None:
    (root / name).mkdir(parents=True)
    (root / name / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {rule}\n---\n\nBody of {name}.\n")


def test_the_skills_claude_agents_load_are_listed_and_the_project_wins(tmp_path, monkeypatch):
    from orchestra import config

    project, added = tmp_path / "game", tmp_path / "agent-skills"
    _skill(project / ".claude" / "skills", "tween-style", "this game's tween style")
    _skill(project / ".agents" / "skills", "board-rules", "this game's board")
    _skill(added / ".claude" / "skills", "tween-style", "the user's tween style")
    _skill(added / ".claude" / "skills", "unity-rules", "the user's unity rules")
    monkeypatch.setattr(config, "project_root", lambda: project)
    monkeypatch.setattr(config, "provider_config", lambda key: {"extra_args": ["--strict-mcp-config", "--add-dir", str(added)]})
    monkeypatch.setattr(skills, "_dirs", _real_dirs)

    assert skills.listing() == [("board-rules", "this game's board"), ("tween-style", "this game's tween style"),
                                ("unity-rules", "the user's unity rules")]


def test_a_review_is_told_the_skills(monkeypatch):
    from orchestra.engine import code_review

    monkeypatch.setattr(skills, "listing", lambda: [("tween-style", "Every tween ends with SetId.")])

    prompt = code_review.build_prompt("diff --git a/x.cs b/x.cs\n+++ b/x.cs\n+tween();\n")

    assert "tween-style" in prompt and "Every tween ends with SetId." in prompt


def test_reading_agents_can_load_skills():
    # Claude uses a skill through the Skill tool; Kimi lists skills only where its profile has ${skills_section}.
    from orchestra.providers.claude import READ_ONLY_TOOLS

    reader = (Path(__file__).parents[1] / "orchestra" / "providers" / "kimi_reader.md").read_text()
    assert "Skill" in READ_ONLY_TOOLS
    assert "  - Skill\n" in reader and "${skills_section}" in reader

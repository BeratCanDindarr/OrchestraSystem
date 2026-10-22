"""Skills (SKILL.md folders) reach every agent CLI through its own flag; the project's win over the user's."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestra import skills


def _skill(root: Path, name: str, rule: str) -> None:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {rule}\n---\n\nBody of {name}.\n")


@pytest.fixture
def skill_dirs(tmp_path, monkeypatch):
    user, project = tmp_path / "user", tmp_path / "project"
    monkeypatch.setattr(skills, "_global_dir", lambda: user)
    monkeypatch.setattr(skills, "_project_dir", lambda: project)
    return user, project


def test_no_skill_folder_means_no_bundle(skill_dirs):
    assert skills.bundle() is None
    assert skills.listing() == []


def test_the_bundle_holds_both_folders_and_the_project_wins(skill_dirs):
    user, project = skill_dirs
    _skill(user, "unity-rules", "the user's rules")
    _skill(user, "tween-style", "the user's tween style")
    _skill(project, "tween-style", "this game's tween style")

    root = skills.bundle()

    assert json.loads((root / ".claude-plugin" / "plugin.json").read_text())["name"] == "orchestra-skills"
    assert sorted(p.name for p in (root / "skills").iterdir()) == ["tween-style", "unity-rules"]
    assert (root / "skills" / "tween-style" / "SKILL.md").read_text().count("this game's") == 1
    assert (root / ".agents" / "skills" / "unity-rules" / "SKILL.md").exists()   # where Antigravity looks
    assert skills.listing() == [("tween-style", "this game's tween style"), ("unity-rules", "the user's rules")]
    assert skills.bundle() == root   # the same skills, the same bundle


def test_each_cli_is_given_the_bundle(skill_dirs, monkeypatch):
    from orchestra.providers import antigravity, claude, kimi

    _skill(skill_dirs[1], "tween-style", "this game's tween style")
    root = skills.bundle()
    monkeypatch.setattr(claude.config, "provider_config", lambda key: {"tools": ["Read", "Grep", "Glob"]})

    command = claude.ClaudeProvider().build_command("q", "opus")
    assert command[command.index("--plugin-dir") + 1] == str(root)
    assert "Skill" in command[command.index("--tools") + 1].split(",")   # without it a skill cannot be used

    command = kimi.KimiProvider().build_command("q", "k3")
    assert command[command.index("--skills-dir") + 1] == str(root / "skills")

    monkeypatch.setattr(antigravity.config, "provider_config", lambda key: {})
    command = antigravity.AntigravityProvider().build_command("q", "argon")
    assert command[command.index("--add-dir") + 1] == str(root)


def test_without_skills_the_commands_are_as_before(skill_dirs, monkeypatch):
    from orchestra.providers import antigravity, claude, kimi

    monkeypatch.setattr(claude.config, "provider_config", lambda key: {"tools": ["Read", "Grep", "Glob"]})
    monkeypatch.setattr(antigravity.config, "provider_config", lambda key: {})

    assert "--plugin-dir" not in claude.ClaudeProvider().build_command("q", "opus")
    assert "--skills-dir" not in kimi.KimiProvider().build_command("q", "k3")
    assert "--add-dir" not in antigravity.AntigravityProvider().build_command("q", "argon")


def test_kimis_reader_profile_can_see_and_load_skills():
    # Kimi lists skills only where the profile's prompt has ${skills_section}, and loads them with Skill.
    reader = Path(__file__).parents[1] / "orchestra" / "providers" / "kimi_reader.md"
    text = reader.read_text()
    assert "  - Skill\n" in text and "${skills_section}" in text


def test_a_review_is_told_the_project_skills(skill_dirs):
    from orchestra.engine import code_review

    _skill(skill_dirs[1], "tween-style", "Every tween ends with SetId.")

    prompt = code_review.build_prompt("diff --git a/x.cs b/x.cs\n+++ b/x.cs\n+tween();\n")

    assert "tween-style" in prompt and "Every tween ends with SetId." in prompt


def test_an_ollama_agent_is_told_where_the_skills_are(skill_dirs, monkeypatch):
    # --bare offers no Skill tool: the agent is given each skill's file and reads the one it needs.
    from orchestra.providers import claude
    from orchestra.providers.ollama_agent import OllamaAgentProvider

    _skill(skill_dirs[1], "tween-style", "this game's tween style")
    monkeypatch.setattr(claude.config, "provider_config",
                        lambda key: {"tools": ["Read"], "append_system_prompt": "Answer in English."})

    command = OllamaAgentProvider().build_command("q", "gemma4:31b-cloud")
    told = command[command.index("--append-system-prompt") + 1]

    assert command.count("--append-system-prompt") == 1
    assert told.startswith("Answer in English.")
    assert "tween-style: this game's tween style" in told
    assert str(skills.bundle() / "skills" / "tween-style" / "SKILL.md") in told

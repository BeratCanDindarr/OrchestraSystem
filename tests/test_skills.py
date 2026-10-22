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
    # --bare reads only inside the project: a skill linked from ~/.claude must be added to read it
    added = [command[i + 1] for i, arg in enumerate(command) if arg == "--add-dir"]
    assert str((skill_dirs[1] / "tween-style").resolve()) in added


# ── user_setup: an agent profile may load its CLI's own setup (the user's skills, plugins, rules) ──

def _profile(monkeypatch, module, **profile):
    monkeypatch.setattr(module.config, "provider_config", lambda key: dict(profile))


def test_claude_with_user_setup_loads_the_users_settings_and_keeps_reading_only(skill_dirs, monkeypatch):
    from orchestra.providers import claude

    _profile(monkeypatch, claude, tools=["Read", "Grep", "Glob"], user_setup=True,
             extra_args=["--strict-mcp-config", "--setting-sources", "project"])

    command = claude.ClaudeProvider().build_command("q", "opus")

    assert command[command.index("--setting-sources") + 1] == "user,project,local"
    assert command.count("--setting-sources") == 1
    # The user's settings may allow a whole MCP server (UnityMCP: read and write): a reading agent
    # loads no MCP server but Orchestra's read-only proxies.
    assert "--strict-mcp-config" in command
    assert command[command.index("--tools") + 1] == "Read,Grep,Glob,Skill"   # the user's skills, usable
    # The user's hooks are for their own sessions: in an agent call one wrote to their session notes.
    assert json.loads(command[command.index("--settings") + 1]) == {"disableAllHooks": True}


def test_a_writing_profile_with_user_setup_gets_the_users_mcp_servers_too(skill_dirs, monkeypatch):
    from orchestra.providers import claude

    _profile(monkeypatch, claude, skip_permissions=True, user_setup=True, extra_args=["--strict-mcp-config"])

    command = claude.ClaudeProvider(config_key="claude_write").build_command("q", "opus")

    assert "--strict-mcp-config" not in command and command[command.index("--setting-sources") + 1] == "user,project,local"


def test_claude_without_user_setup_is_as_configured(skill_dirs, monkeypatch):
    from orchestra.providers import claude

    _profile(monkeypatch, claude, tools=["Read"], extra_args=["--strict-mcp-config", "--setting-sources", "project"])

    command = claude.ClaudeProvider().build_command("q", "opus")

    assert "--strict-mcp-config" in command and command[command.index("--setting-sources") + 1] == "project"


def test_an_ollama_agent_with_user_setup_is_not_bare(skill_dirs, monkeypatch):
    from orchestra.providers import claude
    from orchestra.providers.ollama_agent import OllamaAgentProvider

    _profile(monkeypatch, claude, tools=["Read"])
    assert "--bare" in OllamaAgentProvider().build_command("q", "gemma4:31b-cloud")

    _profile(monkeypatch, claude, tools=["Read"], user_setup=True)
    assert "--bare" not in OllamaAgentProvider().build_command("q", "gemma4:31b-cloud")


def test_kimi_with_user_setup_also_gets_its_own_skills(skill_dirs, monkeypatch, tmp_path):
    from orchestra.providers import kimi

    own = tmp_path / "kimi-home" / "skills"
    own.mkdir(parents=True)
    monkeypatch.setenv("KIMI_CODE_HOME", str(tmp_path / "kimi-home"))
    _skill(skill_dirs[1], "tween-style", "this game's tween style")

    _profile(monkeypatch, kimi, user_setup=True)
    command = kimi.KimiProvider().build_command("q", "k3")

    given = [command[i + 1] for i, arg in enumerate(command) if arg == "--skills-dir"]
    assert given == [str(skills.bundle() / "skills"), str(own)]
    assert command[command.index("--agent-file") + 1].endswith("kimi_reader.md")   # still reads only

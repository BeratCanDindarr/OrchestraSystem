"""Ollama models run as agents: the Claude CLI pointed at Ollama's Anthropic API."""
from __future__ import annotations

import json
from unittest.mock import patch

from orchestra.providers.ollama_agent import OllamaAgentProvider


def _command(provider: OllamaAgentProvider, profile: dict | None = None) -> list[str]:
    with patch.object(provider, "_binary", return_value="claude"), \
         patch("orchestra.providers.claude.config.provider_config", return_value=profile or {}):
        return provider.build_command("read the README", "analyst")


def test_the_cli_is_pointed_at_the_ollama_server_with_the_model_tag():
    cmd = _command(OllamaAgentProvider("http://192.168.1.42:11434"))

    assert cmd[0] == "env" and "ANTHROPIC_BASE_URL=http://192.168.1.42:11434" in cmd
    assert cmd[cmd.index("--model") + 1] == "glm-5.3:cloud"


def test_project_memory_files_are_not_loaded_on_every_turn():
    cmd = _command(OllamaAgentProvider("http://localhost:11434"))

    assert cmd[3:6] == ["claude", "-p", "--bare"] and "ANTHROPIC_API_KEY=ollama" in cmd


def test_without_a_profile_the_model_can_only_read():
    cmd = _command(OllamaAgentProvider("http://localhost:11434"))

    assert cmd[cmd.index("--tools") + 1] == "Read,Grep,Glob,Skill"
    assert "--dangerously-skip-permissions" not in cmd


def test_the_claude_price_the_cli_reports_is_not_charged():
    provider = OllamaAgentProvider("http://localhost:11434")
    output = json.dumps({"result": "pong", "is_error": False, "total_cost_usd": 0.06,
                         "usage": {"input_tokens": 100, "output_tokens": 2}})

    with patch("orchestra.providers.base.run_command", return_value=(output, "", 0)), \
         patch.object(provider, "_binary", return_value="claude"):
        text, code = provider.run("ping", "analyst")

    assert (text, code) == ("pong", 0)
    assert provider.last_usage() == {"prompt_tokens": 100, "completion_tokens": 2}


def test_the_label_names_the_ollama_model_so_its_own_price_applies():
    assert OllamaAgentProvider().model_label("analyst") == "ollama/glm-5.3:cloud"


def test_the_skills_folder_is_left_out_but_mcp_servers_stay():
    # --bare has no Skill tool: an --add-dir skills folder only added ~9.6K tokens a turn.
    cmd = _command(OllamaAgentProvider("http://localhost:11434"), {
        "tools": ["Read"], "extra_args": ["--strict-mcp-config", "--add-dir", "/u/agent-skills",
                                          "--mcp-config", "/u/mcp.json", "--allowed-tools", "mcp__memory__read_graph"]})

    assert "--add-dir" not in cmd and "/u/agent-skills" not in cmd
    assert cmd[cmd.index("--mcp-config") + 1] == "/u/mcp.json" and "--allowed-tools" in cmd

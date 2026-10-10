"""Kimi CLI provider adapter.

Runs the locally installed, logged-in `kimi` CLI in its non-interactive prompt
mode, so Kimi models are reached through the user's Kimi account instead of an
Ollama cloud tag.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from orchestra import config
from orchestra.providers.base import BaseProvider


# Short names (and tier shortcuts) -> model aliases defined in ~/.kimi-code/config.toml
MODELS = {
    "k3":   "kimi-code/k3",                          # kimi-deep
    "code": "kimi-code/kimi-for-coding",             # kimi-code
    "fast": "kimi-code/kimi-for-coding-highspeed",
    # tier shortcuts
    "heavy":  "kimi-code/k3",
    "medium": "kimi-code/kimi-for-coding",
    "light":  "kimi-code/kimi-for-coding-highspeed",
}

_DEFAULT_BINARY = os.path.expanduser("~/.kimi-code/bin/kimi")
_MODEL_PREFIX = "kimi-code/"


def parse_stream_json(stdout: str) -> str:
    """Return the final assistant message of a `--output-format stream-json` run.

    The CLI prints one JSON object per line: meta lines (version, resume hint)
    and assistant messages. Earlier assistant messages are narration between
    tool calls; the last one carries the answer. Returns "" when no assistant
    message could be read.
    """
    answer = ""
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        text = _content_text(message.get("content"))
        if text:
            answer = text
    return answer


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type", "text") == "text"
        ]
        return "\n".join(part for part in parts if part).strip()
    return ""


READER_AGENT = Path(__file__).with_name("kimi_reader.md")


class KimiProvider(BaseProvider):
    name = "kimi"

    def _binary(self) -> str:
        return shutil.which("kimi") or _DEFAULT_BINARY

    def is_available(self) -> bool:
        return shutil.which("kimi") is not None or os.access(_DEFAULT_BINARY, os.X_OK)

    def model_label(self, effort_or_model: str) -> str:
        model = MODELS.get(effort_or_model, effort_or_model)
        return f"kimi/{model.removeprefix(_MODEL_PREFIX)}"

    def build_command(self, prompt: str, effort_or_model: str) -> list[str]:
        model = MODELS.get(effort_or_model, effort_or_model)
        command = [self._binary(), "-p", prompt, "-m", model, "--output-format", "stream-json"]
        agent_file = self.agent_file()
        if agent_file:
            command += ["--agent-file", agent_file]
        return command

    @staticmethod
    def agent_file() -> str | None:
        """The agent definition a Kimi call runs under, or None for Kimi's own default agent.

        Kimi's default agent has Bash, Edit, Write and every MCP server the user configured
        (on this machine: the Unity editor). Orchestra's agents advise; unless
        `[providers.kimi] read_only = false`, a call runs under the bundled reader profile:
        read, search and web tools only. `agent_file` names another definition.
        """
        profile = config.provider_config("kimi")
        if profile.get("agent_file"):
            return str(Path(str(profile["agent_file"])).expanduser())
        if profile.get("read_only", True):
            return str(READER_AGENT)
        return None

    def run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        if kwargs.get("response_format") == "json":
            prompt += "\n\nPlease respond with valid JSON only."
        output, returncode = super().run(prompt, effort_or_model, timeout, **kwargs)
        if returncode != 0:
            return output, returncode

        answer = parse_stream_json(output)
        if not answer:
            return f"[ERROR] kimi returned no assistant message: {output[:200]}", 1
        return answer, 0

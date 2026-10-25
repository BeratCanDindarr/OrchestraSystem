"""Antigravity CLI (agy) provider adapter.

Runs the locally installed `agy` CLI in non-interactive prompt mode with JSON output.
Enables access to Gemini 4 Argon, Gemini 3.8 Flash, and other models configured in Antigravity.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from orchestra import config
from orchestra.providers.base import BaseProvider

# Model mapping: short names, tiers, and model keys -> agy models
MODELS = {
    # tiers
    "heavy": "gemini-4-argon-high",
    "medium": "gemini-4-argon-medium",
    "light": "gemini-3.8-flash-high",
    # argon aliases
    "argon": "gemini-4-argon-high",
    "argon-high": "gemini-4-argon-high",
    "argon-medium": "gemini-4-argon-medium",
    "argon-low": "gemini-4-argon-low",
    # flash aliases
    "flash": "gemini-3.8-flash-high",
    "flash-high": "gemini-3.8-flash-high",
    "flash-medium": "gemini-3.8-flash-medium",
    "flash-low": "gemini-3.8-flash-low",
}

_DEFAULT_BINARY = os.path.expanduser("~/.local/bin/agy")


def parse_result(output: str) -> tuple[str, dict | None, bool]:
    """Extract (answer, usage, is_error) from `agy -p ... --output-format json` output.

    Antigravity returns a JSON payload containing `response`, `status`, and `usage`.
    If standard output has leading logs or notifications, this extracts the last valid
    JSON object matching the Antigravity output schema.
    """
    if not output:
        return "", None, True

    data = None
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                candidate = json.loads(line)
                if isinstance(candidate, dict) and candidate.get("event") == "result" \
                        and isinstance(candidate.get("result"), dict):
                    candidate = candidate["result"]   # a stream ends with the same object `json` gives
                if isinstance(candidate, dict) and "status" in candidate:
                    data = candidate
                    break
            except (json.JSONDecodeError, ValueError):
                continue

    if data is None:
        try:
            data = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            return output.strip(), None, False

    if not isinstance(data, dict):
        return output.strip(), None, False

    is_error = data.get("status") != "SUCCESS"
    if is_error:
        err_msg = data.get("error") or data.get("response") or "Antigravity execution failed"
        return str(err_msg).strip(), None, True

    response = str(data.get("response") or "").strip()
    if not response:
        # Headless agy denies a tool it may not use and still reports SUCCESS with no answer.
        denied = [str(d.get("action")) for d in data.get("denied_actions") or [] if isinstance(d, dict)]
        reason = f"denied tool permission: {', '.join(denied)}" if denied else "empty response"
        return f"Antigravity returned no answer ({reason})", None, True
    used = data.get("usage")
    if not isinstance(used, dict):
        used = {}
    usage = {
        "prompt_tokens": int(used.get("input_tokens") or 0),
        "completion_tokens": int(used.get("output_tokens") or 0),
    }
    if "thinking_tokens" in used:
        usage["thinking_tokens"] = int(used.get("thinking_tokens") or 0)

    return response, usage, False


class AntigravityProvider(BaseProvider):
    name = "antigravity"

    def _binary(self) -> str:
        return shutil.which("agy") or _DEFAULT_BINARY

    def is_available(self) -> bool:
        if not config.provider_enabled(self.name):
            return False
        return shutil.which("agy") is not None or os.access(_DEFAULT_BINARY, os.X_OK)

    def can_write(self, effort_or_model: str = "") -> bool:
        return bool(config.provider_config(self.name).get("skip_permissions", False))

    def model_label(self, effort_or_model: str) -> str:
        model = MODELS.get(effort_or_model, effort_or_model)
        return f"antigravity/{model}"

    def build_command(self, prompt: str, effort_or_model: str) -> list[str]:
        model = MODELS.get(effort_or_model, effort_or_model)
        profile = config.provider_config(self.name)
        command = [
            self._binary(),
            "-p", prompt,
            "--model", model,
            "--output-format", "stream-json",   # thinking and tool steps are output: a working agent is never silent
        ]
        if profile.get("skip_permissions", False):   # like cld-fast/cld-deep: no bypass unless asked
            command.append("--dangerously-skip-permissions")
        effort = profile.get("effort")
        if effort:
            command += ["--effort", str(effort)]
        extra_args = profile.get("extra_args")
        if isinstance(extra_args, list):
            command.extend(str(arg) for arg in extra_args)
        return command

    def run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        if kwargs.get("response_format") == "json":
            prompt += "\n\nPlease respond with valid JSON only."
        output, returncode = super().run(prompt, effort_or_model, timeout, **kwargs)
        if returncode != 0:
            text, _, is_err = parse_result(output)
            if is_err and text:
                return f"[ERROR] {text}", returncode
            return output, returncode

        text, usage, is_err = parse_result(output)
        if is_err:
            return f"[ERROR] {text}", 1
        self._last_usage = usage
        return text, 0

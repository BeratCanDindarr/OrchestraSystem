"""Ollama models as agents: the Claude CLI, pointed at Ollama's Anthropic API.

Ollama (0.40+) serves `/v1/messages`, so `claude -p --model <tag>` with ANTHROPIC_BASE_URL set to
the Ollama server runs that model with the CLI's own tools: it reads the project like cld-fast
does (the `[providers.claude]` profile decides which tools). The plain HTTP client in
`ollama.py` stays for embeddings, health checks and listing models.
"""
from __future__ import annotations

from orchestra import config, skills
from orchestra.providers.base import BaseProvider
from orchestra.providers.claude import ClaudeProvider, parse_result
from orchestra.providers.ollama import MODELS, OllamaProvider


class OllamaAgentProvider(ClaudeProvider):
    name = "ollama"

    def __init__(self, base_url: str | None = None) -> None:
        super().__init__(config_key="claude")
        self._http = OllamaProvider(base_url)

    def is_available(self) -> bool:
        return self._http.is_available() and super().is_available()

    def model_label(self, effort_or_model: str) -> str:
        return self._http.model_label(effort_or_model)

    def build_command(self, prompt: str, effort_or_model: str) -> list[str]:
        model = MODELS.get(effort_or_model, effort_or_model)
        binary, print_flag, *rest = super().build_command(prompt, model)
        # --bare: no auto-loaded CLAUDE.md / rules / hooks. They cost ~9K tokens on every turn;
        # the prompt points the agent at the docs and it reads what it needs (2.4K vs 26K tokens
        # for a two-turn file question). Bare mode authenticates by ANTHROPIC_API_KEY only.
        # `env` execs the CLI in place: same pid, so cancelling the process group still works.
        if config.provider_config(self.config_key).get("user_setup"):
            # The user's own setup instead of --bare: their skills (with the Skill tool), rules, plugins.
            bare: list[str] = []
        else:
            bare, rest = ["--bare"], _with_skill_files(rest)
        return ["env", f"ANTHROPIC_BASE_URL={self._http._base_url()}", "ANTHROPIC_API_KEY=ollama",
                binary, print_flag, *bare, *rest]

    # Usage goes through the base ContextVar: async_run runs run() in a thread and carries it back.
    def last_usage(self) -> dict | None:
        return BaseProvider.last_usage(self)

    def reset_usage(self) -> None:
        BaseProvider.reset_usage(self)

    def run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        self._last_usage = None
        output, returncode = BaseProvider.run(self, prompt, effort_or_model, timeout, **kwargs)
        if returncode != 0:
            return output, returncode
        text, usage, failed = parse_result(output)
        if usage:
            # The CLI prices every model at Claude's rates; Ollama's price comes from config.price_of.
            usage.pop("cost_usd", None)
        self._last_usage = usage
        if failed:
            return f"[ERROR] {text}" if text else "[ERROR] the Ollama model reported an error", 1
        return text, 0

    async def async_run(self, prompt: str, effort_or_model: str, timeout: int = 180, **kwargs) -> tuple[str, int]:
        return await BaseProvider.async_run(self, prompt, effort_or_model, timeout, **kwargs)


def _with_skill_files(args: list[str]) -> list[str]:
    """--bare offers no Skill tool: name each skill and its file in the system prompt instead."""
    bundle = skills.bundle()
    if not bundle:
        return args
    told = "Project skills (instructions): read the SKILL.md of any that fits the task before answering.\n" + "".join(
        f"- {name}: {about} ({bundle / 'skills' / name / 'SKILL.md'})\n" for name, about in skills.listing())
    if "--append-system-prompt" in args:
        at = args.index("--append-system-prompt") + 1
        return [*args[:at], f"{args[at]}\n\n{told}", *args[at + 1:]]
    at = args.index("--output-format")
    return [*args[:at], "--append-system-prompt", told, *args[at:]]

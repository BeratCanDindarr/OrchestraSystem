"""Smart command bar widget for Orchestra TUI v3."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option


@dataclass(frozen=True)
class CommandSpec:
    """Metadata for an explicit slash or keyword command."""

    name: str
    synopsis: str
    description: str


@dataclass(frozen=True)
class SmartSubmission:
    """Normalized SmartCommandBar submission payload."""

    raw: str
    normalized: str
    kind: Literal["command", "task"]
    command: str | None = None
    args: tuple[str, ...] = ()
    task: str | None = None

    @property
    def is_command(self) -> bool:
        return self.kind == "command"

    @property
    def is_task(self) -> bool:
        return self.kind == "task"


@dataclass(frozen=True)
class _CompletionItem:
    label: str
    replacement: str
    description: str
    category: str


class HistoryRequested(Message):
    def __init__(self, step: int) -> None:
        super().__init__()
        self.step = step


class CompletionAccepted(Message):
    pass


class CompletionDismissed(Message):
    pass


class _SmartInput(Input):
    BINDINGS = [
        *Input.BINDINGS,
        Binding("up", "history_previous", show=False),
        Binding("down", "history_next", show=False),
        Binding("tab", "accept_completion", show=False),
        Binding("escape", "dismiss_completion", show=False),
    ]

    def action_history_previous(self) -> None:
        self.post_message(HistoryRequested(-1))

    def action_history_next(self) -> None:
        self.post_message(HistoryRequested(1))

    def action_accept_completion(self) -> None:
        self.post_message(CompletionAccepted())

    def action_dismiss_completion(self) -> None:
        self.post_message(CompletionDismissed())


class SmartCommandBar(Vertical):
    """Bottom command bar with history, smart completion, and task fallback."""

    DEFAULT_CSS = """
    SmartCommandBar {
        height: auto;
        background: #252526;
        border-top: solid #007acc;
        padding: 0 1 1 1;
    }

    SmartCommandBar #smart-bar-shell {
        height: 3;
        background: #1e1e1e;
        border: solid #3c3c3c;
        color: #cccccc;
        padding: 0 1;
    }

    SmartCommandBar #smart-bar-prefix {
        width: 12;
        content-align: center middle;
        color: #4ec9b0;
        text-style: bold;
    }

    SmartCommandBar #smart-bar-input {
        width: 1fr;
        border: none;
        background: transparent;
        color: #ffffff;
    }

    SmartCommandBar #smart-bar-input:focus {
        background: transparent;
    }

    SmartCommandBar #smart-bar-suggestions {
        display: none;
        background: #1e1e1e;
        border: solid #007acc;
        color: #cccccc;
        margin: 0 0 0 12;
        max-height: 8;
    }

    SmartCommandBar #smart-bar-suggestions > .option-list--option-highlighted {
        background: #37373d;
        color: #ffffff;
    }

    SmartCommandBar #smart-bar-status {
        height: 1;
        color: #6b6b6b;
        padding: 0 1 0 12;
    }
    """

    DEFAULT_COMMANDS: tuple[CommandSpec, ...] = (
        CommandSpec("help", "help", "List available command verbs."),
        CommandSpec("open", "open <run_id>", "Open a run by id."),
        CommandSpec("diff", "diff <left_alias> <right_alias>", "Compare two agent streams."),
        CommandSpec("playbook", "playbook <path>", "Run a markdown playbook."),
        CommandSpec("ask", "ask <alias> <prompt>", "Launch a single-agent run."),
        CommandSpec("dual", "dual <prompt>", "Launch a dual-agent run."),
        CommandSpec("critical", "critical <prompt>", "Launch a critical review run."),
        CommandSpec("planned", "planned <prompt>", "Launch a planned run."),
        CommandSpec("resume", "resume <run_id> [approve|reject]", "Resume a paused run."),
    )

    class Submitted(Message):
        """Posted when the bar submits either a command or a task."""

        def __init__(self, bar: "SmartCommandBar", submission: SmartSubmission) -> None:
            super().__init__()
            self.bar = bar
            self.submission = submission

    def __init__(
        self,
        *,
        placeholder: str | None = None,
        history_limit: int = 50,
        commands: Iterable[CommandSpec] | None = None,
        run_ids: Iterable[str] | None = None,
        agent_aliases: Iterable[str] | None = None,
        playbooks: Iterable[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._debounce_timer: Any = None
        self.history_limit = max(1, history_limit)
        specs = tuple(commands or self.DEFAULT_COMMANDS)
        self._commands: dict[str, CommandSpec] = {spec.name.lower(): spec for spec in specs}
        self._history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""
        self._completion_items: list[_CompletionItem] = []
        self._run_ids = _dedupe(run_ids or ())
        self._agent_aliases = _dedupe(agent_aliases or ())
        self._playbooks = _dedupe(playbooks or ())
        self._programmatic_change = False

        self._input = _SmartInput(
            placeholder=placeholder or "Type / for commands, or describe the task you want Orchestra to run.",
            id="smart-bar-input",
        )
        self._suggestions = OptionList(id="smart-bar-suggestions")
        self._status = Static(id="smart-bar-status")

    def compose(self) -> ComposeResult:
        with Horizontal(id="smart-bar-shell"):
            yield Static("MAESTRO", id="smart-bar-prefix")
            yield self._input
        yield self._suggestions
        yield self._status

    def on_mount(self) -> None:
        self._hide_suggestions()
        self._update_status()

    @property
    def value(self) -> str:
        return self._input.value

    @value.setter
    def value(self, value: str) -> None:
        self._set_input_value(value)
        self._refresh_completions()
        self._update_status()

    @property
    def history(self) -> tuple[str, ...]:
        return tuple(self._history)

    @property
    def suggestions_visible(self) -> bool:
        return bool(self._completion_items) and self._suggestions.display

    @property
    def current_suggestions(self) -> tuple[str, ...]:
        return tuple(item.label for item in self._completion_items)

    def focus_input(self) -> None:
        self._input.focus()

    def clear_history(self) -> None:
        self._history.clear()
        self._history_index = None
        self._history_draft = ""

    def push_history(self, value: str) -> None:
        entry = value.strip()
        if not entry:
            return
        self._history.append(entry)
        if len(self._history) > self.history_limit:
            self._history = self._history[-self.history_limit :]
        self._history_index = None
        self._history_draft = ""

    def set_completion_context(
        self,
        *,
        run_ids: Iterable[str] | None = None,
        agent_aliases: Iterable[str] | None = None,
        playbooks: Iterable[str] | None = None,
    ) -> None:
        if run_ids is not None:
            self._run_ids = _dedupe(run_ids)
        if agent_aliases is not None:
            self._agent_aliases = _dedupe(agent_aliases)
        if playbooks is not None:
            self._playbooks = _dedupe(playbooks)
        self._refresh_completions()

    @classmethod
    def parse_input(
        cls,
        raw: str,
        *,
        commands: Mapping[str, CommandSpec] | None = None,
    ) -> SmartSubmission | None:
        text = raw.strip()
        if not text:
            return None

        normalized = text[1:].strip() if text.startswith("/") else text
        if not normalized:
            return None

        parts = shlex.split(normalized)
        if not parts:
            return None

        command_name = parts[0].lower()
        command_map = commands or {spec.name: spec for spec in cls.DEFAULT_COMMANDS}
        if command_name in command_map:
            return SmartSubmission(
                raw=text,
                normalized=normalized,
                kind="command",
                command=command_name,
                args=tuple(parts[1:]),
            )

        return SmartSubmission(
            raw=text,
            normalized=normalized,
            kind="task",
            task=normalized,
        )

        def on_input_changed(self, event: Input.Changed) -> None:
        if event.input is not self._input:
            return
        if self._programmatic_change:
            return
            
        if self._debounce_timer:
            self._debounce_timer.stop()
        self._debounce_timer = self.set_timer(0.25, self._refresh_completions)
        self._update_status()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input is not self._input:
            return

        raw = event.value.strip()
        if not raw:
            return

        try:
            submission = self.parse_input(raw, commands=self._commands)
        except ValueError as exc:
            self._status.update(f"Parse error: {exc}")
            return

        if submission is None:
            return

        self.push_history(raw)
        self._set_input_value("")
        self._hide_suggestions()
        self._update_status(submission)
        self.post_message(self.Submitted(self, submission))

    def on_history_requested(self, message: HistoryRequested) -> None:
        message.stop()
        self._navigate_history(message.step)

    def on_completion_accepted(self, message: CompletionAccepted) -> None:
        message.stop()
        self._apply_highlighted_completion()

    def on_completion_dismissed(self, message: CompletionDismissed) -> None:
        message.stop()
        self._hide_suggestions()
        self._update_status()

    def on_option_list_option_selected(self, message: OptionList.OptionSelected) -> None:
        if message.option_list is not self._suggestions:
            return
        self._apply_completion_index(message.index)
        self._input.focus()

    def on_option_list_option_highlighted(self, message: OptionList.OptionHighlighted) -> None:
        if message.option_list is not self._suggestions:
            return
        self._update_status()

    def _navigate_history(self, step: int) -> None:
        if not self._history:
            return

        if self._history_index is None:
            self._history_index = len(self._history)
            self._history_draft = self._input.value

        next_index = max(0, min(len(self._history), self._history_index + step))
        if next_index == self._history_index:
            return

        self._history_index = next_index
        next_value = self._history_draft if next_index == len(self._history) else self._history[next_index]
        self._set_input_value(next_value)
        self._refresh_completions()
        self._update_status()

    def _apply_highlighted_completion(self) -> None:
        if not self._completion_items:
            return
        highlighted = self._suggestions.highlighted
        index = highlighted if highlighted is not None else 0
        self._apply_completion_index(index)

    def _apply_completion_index(self, index: int) -> None:
        if index < 0 or index >= len(self._completion_items):
            return
        item = self._completion_items[index]
        self._set_input_value(item.replacement)
        self._refresh_completions()
        self._update_status()

    def _set_input_value(self, value: str) -> None:
        self._programmatic_change = True
        try:
            self._input.value = value
            self._input.cursor_position = len(value)
        finally:
            self._programmatic_change = False

    def _refresh_completions(self) -> None:
        self._completion_items = self._build_completion_items(self._input.value)
        if not self._completion_items:
            self._hide_suggestions()
            return

        options = [
            Option(
                Text.assemble(
                    (item.label, "bold #ffffff"),
                    ("  ", ""),
                    (item.description, "#6b6b6b"),
                ),
                id=f"completion-{index}",
            )
            for index, item in enumerate(self._completion_items)
        ]
        self._suggestions.set_options(options)
        self._suggestions.highlighted = 0
        self._suggestions.display = True

    def _hide_suggestions(self) -> None:
        self._completion_items = []
        self._suggestions.clear_options()
        self._suggestions.display = False

    def _build_completion_items(self, value: str) -> list[_CompletionItem]:
        text = value.lstrip()
        if not text:
            return []

        uses_slash = text.startswith("/")
        working = text[1:] if uses_slash else text
        ends_with_space = working.endswith(" ")
        tokens = working.split()

        if uses_slash and (not tokens or (len(tokens) == 1 and not ends_with_space)):
            prefix = tokens[0].lower() if tokens else ""
            return [
                self._command_completion(spec, uses_slash=True)
                for spec in self._matching_commands(prefix)
            ]

        if not uses_slash and len(tokens) == 1 and not ends_with_space:
            prefix = tokens[0].lower()
            matches = self._matching_commands(prefix)
            if matches:
                return [self._command_completion(spec, uses_slash=False) for spec in matches]

        if not tokens:
            return []

        command_name = tokens[0].lower()
        if command_name not in self._commands:
            return []

        args = tokens[1:]
        arg_index = len(args) if ends_with_space else max(0, len(args) - 1)
        fragment = "" if ends_with_space or not args else args[-1]
        return self._argument_completions(
            command_name=command_name,
            uses_slash=uses_slash,
            args=args,
            arg_index=arg_index,
            fragment=fragment,
            ends_with_space=ends_with_space,
        )

    def _matching_commands(self, prefix: str) -> list[CommandSpec]:
        specs = sorted(self._commands.values(), key=lambda spec: spec.name)
        if not prefix:
            return specs
        return [spec for spec in specs if spec.name.startswith(prefix)]

    def _command_completion(self, spec: CommandSpec, *, uses_slash: bool) -> _CompletionItem:
        prefix = "/" if uses_slash else ""
        return _CompletionItem(
            label=f"{prefix}{spec.name}",
            replacement=f"{prefix}{spec.name} ",
            description=spec.synopsis,
            category="command",
        )

    def _argument_completions(
        self,
        *,
        command_name: str,
        uses_slash: bool,
        args: list[str],
        arg_index: int,
        fragment: str,
        ends_with_space: bool,
    ) -> list[_CompletionItem]:
        choices: list[str] = []
        if command_name in {"open", "resume"} and arg_index == 0:
            choices = self._run_ids
        elif command_name == "diff" and arg_index in {0, 1}:
            choices = self._agent_aliases
            if arg_index == 1 and args:
                choices = [choice for choice in choices if choice != args[0]]
        elif command_name == "ask" and arg_index == 0:
            choices = self._agent_aliases
        elif command_name == "playbook" and arg_index == 0:
            choices = self._playbooks

        if not choices:
            return []

        lowered_fragment = fragment.lower()
        filtered = [
            choice for choice in choices
            if not lowered_fragment or choice.lower().startswith(lowered_fragment)
        ]
        if not filtered:
            return []

        spec = self._commands[command_name]
        return [
            _CompletionItem(
                label=choice,
                replacement=self._replacement_for_argument(
                    command_name=command_name,
                    uses_slash=uses_slash,
                    args=args,
                    replacement=choice,
                    ends_with_space=ends_with_space,
                ),
                description=spec.synopsis,
                category="argument",
            )
            for choice in filtered
        ]

    def _replacement_for_argument(
        self,
        *,
        command_name: str,
        uses_slash: bool,
        args: list[str],
        replacement: str,
        ends_with_space: bool,
    ) -> str:
        next_args = list(args)
        if ends_with_space:
            next_args.append(replacement)
        elif next_args:
            next_args[-1] = replacement
        else:
            next_args = [replacement]

        command_prefix = f"/{command_name}" if uses_slash else command_name
        return " ".join([command_prefix, *next_args]).strip() + " "

    def _update_status(self, submission: SmartSubmission | None = None) -> None:
        if submission is not None:
            if submission.is_command:
                spec = self._commands.get(submission.command or "")
                detail = spec.synopsis if spec is not None else submission.normalized
                self._status.update(f"Command submitted: {detail}")
            else:
                self._status.update("Task submitted: routed as natural language.")
            return

        if self._completion_items:
            highlighted = self._suggestions.highlighted
            index = highlighted if highlighted is not None else 0
            item = self._completion_items[index]
            self._status.update(f"{item.description} | Tab accepts | Enter submits | Up/Down history")
            return

        text = self._input.value.strip()
        if not text:
            self._status.update("Slash commands, contextual completion, and natural language task routing.")
            return

        try:
            parsed = self.parse_input(text, commands=self._commands)
        except ValueError as exc:
            self._status.update(f"Parse error: {exc}")
            return

        if parsed is None:
            self._status.update("Ready.")
        elif parsed.is_command:
            spec = self._commands.get(parsed.command or "")
            detail = spec.description if spec is not None else parsed.normalized
            self._status.update(f"{detail} | Up/Down history")
        else:
            self._status.update("Not a known command. Enter will route this as a task.")


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        items.append(value)
    return items

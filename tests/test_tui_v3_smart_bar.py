"""Focused tests for the Orchestra TUI v3 smart command bar."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from orchestra.tui.widgets.v3.smart_bar import SmartCommandBar


class SmartBarHarness(App[None]):
    def __init__(self, widget: SmartCommandBar) -> None:
        super().__init__()
        self.widget = widget

    def compose(self) -> ComposeResult:
        yield self.widget


@pytest.mark.asyncio
async def test_smart_bar_history_is_bounded_and_navigable() -> None:
    widget = SmartCommandBar(history_limit=3)

    async with SmartBarHarness(widget).run_test() as pilot:
        widget.push_history("open run-1")
        widget.push_history("dual investigate flaky tests")
        widget.push_history("critical inspect router")
        widget.push_history("planned build rollout plan")
        await pilot.pause()

        assert widget.history == (
            "dual investigate flaky tests",
            "critical inspect router",
            "planned build rollout plan",
        )

        await pilot.click("#smart-bar-input")
        await pilot.press("up")
        assert widget.value == "planned build rollout plan"

        await pilot.press("up")
        assert widget.value == "critical inspect router"

        await pilot.press("down")
        assert widget.value == "planned build rollout plan"


@pytest.mark.asyncio
async def test_smart_bar_shows_contextual_command_and_argument_completions() -> None:
    widget = SmartCommandBar(
        run_ids=["run-001", "run-002"],
        agent_aliases=["cdx-fast", "gmn-pro", "reviewer"],
        playbooks=["docs/plans/triage.md"],
    )

    async with SmartBarHarness(widget).run_test() as pilot:
        await pilot.click("#smart-bar-input")
        await pilot.press("/")
        await pilot.pause()

        assert widget.suggestions_visible
        assert "/open" in widget.current_suggestions

        widget.value = "open "
        await pilot.pause()

        assert widget.suggestions_visible
        assert "run-001" in widget.current_suggestions

        widget.value = "diff c"
        await pilot.pause()

        assert "cdx-fast" in widget.current_suggestions


def test_smart_bar_routes_unknown_input_as_task() -> None:
    submission = SmartCommandBar.parse_input("Refactor the task router for resumable retries")

    assert submission is not None
    assert submission.is_task
    assert submission.task == "Refactor the task router for resumable retries"


def test_smart_bar_parses_explicit_slash_command() -> None:
    submission = SmartCommandBar.parse_input("/open run-123")

    assert submission is not None
    assert submission.is_command
    assert submission.command == "open"
    assert submission.args == ("run-123",)

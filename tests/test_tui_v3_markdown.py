"""Focused tests for the Orchestra TUI v3 async markdown widget."""
from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.await_complete import AwaitComplete
from textual.widgets import Collapsible
from textual.widgets._markdown import MarkdownFence

from orchestra.tui.widgets.v3.markdown_view import (
    AsyncMarkdownWidget,
    ToolUseFence,
)


class WidgetHarness(App[None]):
    """Minimal Textual app used to mount a single widget under test."""

    def __init__(self, widget) -> None:
        super().__init__()
        self.widget = widget

    def compose(self) -> ComposeResult:
        yield self.widget


@pytest.mark.asyncio
async def test_async_markdown_widget_batches_multiple_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    widget = AsyncMarkdownWidget(markdown="Intro\n", batch_interval=0.01)

    async with WidgetHarness(widget).run_test() as pilot:
        await pilot.pause()

        append_calls: list[str] = []

        def append_spy(markdown: str) -> AwaitComplete:
            append_calls.append(markdown)
            return AwaitComplete.nothing()

        monkeypatch.setattr(widget.document, "append_async", append_spy)

        widget.append_chunk("hello ")
        widget.append_chunk("world")

        await pilot.pause(0.05)

        assert append_calls == ["hello world"]
        assert widget.source == "Intro\nhello world"


@pytest.mark.asyncio
async def test_async_markdown_widget_holds_incomplete_fences_until_closed() -> None:
    widget = AsyncMarkdownWidget(batch_interval=0.01)

    async with WidgetHarness(widget).run_test() as pilot:
        await pilot.pause()

        widget.append_chunk(
            "```json\n"
            "{\n"
            '  "recipient_name": "functions.exec_command",\n'
        )
        await pilot.pause(0.05)

        assert widget.document.source == ""
        assert widget.source.startswith("```json")

        widget.append_chunk(
            '  "parameters": {"cmd": "pwd"}\n'
            "}\n"
            "```\n"
        )
        await pilot.pause(0.05)
        await pilot.pause(0.05)

        collapsible = widget.query_one(Collapsible)
        assert collapsible.collapsed is True
        assert "exec_command" in collapsible.title
        assert len(list(widget.query(ToolUseFence))) == 1


@pytest.mark.asyncio
async def test_async_markdown_widget_collapses_tool_payloads_but_keeps_normal_code_visible() -> None:
    widget = AsyncMarkdownWidget(batch_interval=0.01)
    markdown = (
        "Prelude\n\n"
        "```python\n"
        "print('visible code block')\n"
        "```\n\n"
        "```xml\n"
        "<tool_use recipient_name=\"mcp__unityMCP__.manage_scene\">\n"
        "  <parameters action=\"get_active\" />\n"
        "</tool_use>\n"
        "```\n"
    )

    async with WidgetHarness(widget).run_test() as pilot:
        await pilot.pause()
        await widget.set_markdown(markdown)
        await pilot.pause()
        await pilot.pause()

        collapsibles = list(widget.query(Collapsible))
        assert len(collapsibles) == 1
        assert collapsibles[0].collapsed is True
        assert "manage_scene" in collapsibles[0].title

        plain_fences = [
            fence for fence in widget.query(MarkdownFence) if not isinstance(fence, ToolUseFence)
        ]
        assert len(plain_fences) == 1
        assert widget.document.source == markdown


@pytest.mark.asyncio
async def test_async_markdown_widget_tracks_tool_use_batches_without_double_counting() -> None:
    widget = AsyncMarkdownWidget(batch_interval=0.01)
    markdown = (
        "```json\n"
        "{\n"
        '  "tool_uses": [\n'
        '    {"recipient_name": "functions.exec_command", "parameters": {"cmd": "pwd"}},\n'
        '    {"recipient_name": "functions.apply_patch", "parameters": {"patch": "*** Begin Patch\\n*** End Patch"}}\n'
        "  ]\n"
        "}\n"
        "```\n"
    )

    async with WidgetHarness(widget).run_test() as pilot:
        await pilot.pause()
        await widget.set_markdown(markdown)
        await pilot.pause()
        await pilot.pause()

        tool_fence = widget.query_one(ToolUseFence)
        assert tool_fence.payload.call_count == 2
        assert "2 calls" in tool_fence.payload.title

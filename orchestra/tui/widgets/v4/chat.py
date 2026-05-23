"""Unified conversational chat widgets for Orchestra TUI v4."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical, Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static, LoadingIndicator

from orchestra.tui.widgets.v3.markdown_view import AsyncMarkdownWidget


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "agent", "system"]
    content: str
    alias: str | None = None
    is_thinking: bool = False


class MessageBubble(Vertical):
    alias: reactive[str | None] = reactive(None)
    """A single message bubble in the chat stream."""
    
    DEFAULT_CSS = """
    MessageBubble {
        width: 100%;
        height: auto;
        padding: 0 1;
        margin-bottom: 0;
    }
    
    MessageBubble.-user {
        background: $primary 10%;
        border-left: solid $accent;
    }
    
    MessageBubble.-agent {
        background: $surface;
        border-left: solid $success;
    }
    
    MessageBubble.-system {
        background: transparent;
        border: none;
        color: $secondary;
        padding: 0 4;
    }
    
    .bubble-header {
        height: 1;
        margin-bottom: 1;
        text-style: bold;
    }
    
    .bubble-header.-user { color: $accent; }
    .bubble-header.-agent { color: $success; }
    
    .thinking-spinner {
        color: $warning;
        margin-left: 1;
    }
    """

    def __init__(self, message: ChatMessage, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.role = message.role
        self.content = message.content
        self.alias = message.alias
        self.is_thinking = message.is_thinking
        self.add_class(f"-{self.role}")

    def compose(self) -> ComposeResult:
        if self.role != "system":
            label = "YOU" if self.role == "user" else (self.alias or "AGENT").upper()
            with Horizontal(classes=f"bubble-header -{self.role}"):
                yield Static(label)
                if self.is_thinking:
                    yield LoadingIndicator(classes="thinking-spinner")
        
        if self.role == "system":
            yield Static(f"⮑ {self.content}", classes="system-note")
        else:
            yield AsyncMarkdownWidget(id="bubble-markdown", classes="-compact")

        def watch_alias(self, value: str | None) -> None:
        if self.is_mounted and self.role == "agent":
            try:
                self.query_one(".bubble-header", Horizontal).query_one(Static).update((value or "AGENT").upper())
            except: pass

    def on_mount(self) -> None:
        if self.role != "system":
            md = self.query_one("#bubble-markdown", AsyncMarkdownWidget)
            md.set_markdown(self.content)

    def update_content(self, content: str, is_thinking: bool = False) -> None:
        self.content = content
        self.is_thinking = is_thinking
        if self.is_thinking:
            self.add_class("-thinking")
        else:
            self.remove_class("-thinking")
            
        if self.role != "system":
            try:
                self.query_one("#bubble-markdown", AsyncMarkdownWidget).set_markdown(content)
            except: pass

    def append_chunk(self, chunk: str) -> None:
        if self.role != "system":
            try:
                self.query_one("#bubble-markdown", AsyncMarkdownWidget).append_chunk(chunk)
            except: pass


class UnifiedChatWidget(ScrollableContainer):
    """The central conversational stream for TUI v4."""

    DEFAULT_CSS = """
    UnifiedChatWidget {
        height: 1fr;
        padding: 0;
        background: $background;
    }

    AsyncMarkdownWidget.-compact {
        height: auto;
        border: none;
        padding: 0;
        margin: 0;
        background: transparent;
    }

    AsyncMarkdownWidget.-compact ToolAwareMarkdown {
        padding: 0;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._active_bubble: MessageBubble | None = None

    def add_user_message(self, text: str) -> None:
        bubble = MessageBubble(ChatMessage(role="user", content=text))
        self.mount(bubble)
        bubble.scroll_visible()

    def add_system_message(self, text: str) -> None:
        bubble = MessageBubble(ChatMessage(role="system", content=text))
        self.mount(bubble)
        bubble.scroll_visible()

        def start_agent_thinking(self, alias: str) -> None:
        # If we already have a thinking bubble, update it instead of creating a new one
        if self._active_bubble and self._active_bubble.is_thinking:
            self._active_bubble.update_content("", is_thinking=True)
            self._active_bubble.alias = alias
            return

        bubble = MessageBubble(ChatMessage(role="agent", content="", alias=alias, is_thinking=True))
        self.mount(bubble)
        self._active_bubble = bubble
        bubble.scroll_visible()

    def update_agent_message(self, chunk: str, final: bool = False) -> None:
        if not self._active_bubble:
            return
        
        if self._active_bubble.is_thinking:
            # Transition from thinking to writing
            self._active_bubble.update_content("", is_thinking=False)
            try:
                self._active_bubble.query_one(LoadingIndicator).display = False
            except: pass
            
        self._active_bubble.append_chunk(chunk)
        self._active_bubble.scroll_visible()
        
        if final:
            self._active_bubble = None

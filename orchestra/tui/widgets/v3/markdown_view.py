"""Async markdown streaming widget for Orchestra TUI v3."""
from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Iterable
from xml.etree import ElementTree

from markdown_it import MarkdownIt
from markdown_it.token import Token
from textual.app import ComposeResult
from textual.await_complete import AwaitComplete
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.timer import Timer
from textual.widgets import Collapsible, Label, Markdown
from textual.widgets._markdown import (
    MarkdownBlock,
    MarkdownBlockQuote,
    MarkdownBulletList,
    MarkdownFence,
    MarkdownH1,
    MarkdownH2,
    MarkdownH3,
    MarkdownH4,
    MarkdownH5,
    MarkdownH6,
    MarkdownHeader,
    MarkdownHorizontalRule,
    MarkdownOrderedList,
    MarkdownOrderedListItem,
    MarkdownParagraph,
    MarkdownTBody,
    MarkdownTD,
    MarkdownTH,
    MarkdownTHead,
    MarkdownTR,
    MarkdownTable,
    MarkdownUnorderedListItem,
    slug_for_tcss_id,
)


_FENCE_LINE_RE = re.compile(r"(?m)^(?P<fence>`{3,}|~{3,})[^\n]*$")
_JSON_LANGUAGE_HINTS = {"application/json", "javascript", "js", "json", "json5", "jsonc"}
_XML_LANGUAGE_HINTS = {"application/xml", "html", "svg", "xhtml", "xml"}
_TOOL_NAME_KEYS = (
    "recipient_name",
    "tool",
    "tool_name",
    "function",
    "function_name",
)
_ARGUMENT_KEYS = {"args", "arguments", "input", "parameters", "payload"}
_TOOL_HINTS = (
    "apply_patch",
    "exec_command",
    "function.",
    "functions.",
    "mcp__",
    "tool_use",
    "toolcall",
)


@dataclass(frozen=True)
class ToolPayload:
    """Detected tool payload metadata."""

    language: str
    title: str
    call_count: int = 1


class ToolUseFence(MarkdownFence):
    """Markdown fence that keeps tool payloads collapsed by default."""

    DEFAULT_CSS = """
    ToolUseFence {
        margin: 1 0;
        padding: 0;
        background: transparent;
        width: 1fr;
        height: auto;
    }

    ToolUseFence > .tool-use-collapsible {
        width: 1fr;
        height: auto;
        background: $surface-lighten-1;
        border: round $primary 20%;
        padding: 0;
    }

    ToolUseFence > .tool-use-collapsible.-collapsed {
        border: round $primary 45%;
    }

    ToolUseFence > .tool-use-collapsible > CollapsibleTitle {
        width: 1fr;
        background: $primary 60%;
        color: $text;
        padding: 0 1;
        text-style: bold;
    }

    ToolUseFence > .tool-use-collapsible > CollapsibleTitle:hover {
        background: $accent 35%;
        color: $text;
    }

    ToolUseFence .tool-use-code {
        width: 1fr;
        height: auto;
        padding: 1 2;
        background: $surface-darken-1;
        color: $text;
    }
    """

    def __init__(
        self,
        markdown: Markdown,
        token: Token,
        code: str,
        payload: ToolPayload,
    ) -> None:
        self.payload = payload
        super().__init__(markdown, token, code)

    def set_content(self, content) -> None:  # type: ignore[override]
        self._content = content
        with suppress(NoMatches):
            self.query_one(".tool-use-code", Label).update(content)

    def compose(self) -> ComposeResult:
        yield Collapsible(
            Label(self._highlighted_code, expand=True, classes="tool-use-code"),
            title=self.payload.title,
            collapsed=True,
            classes="tool-use-collapsible",
        )


class ToolAwareMarkdown(Markdown):
    """Markdown widget with async append and collapsible tool payload blocks."""

    BLOCKS = {
        "h1": MarkdownH1,
        "h2": MarkdownH2,
        "h3": MarkdownH3,
        "h4": MarkdownH4,
        "h5": MarkdownH5,
        "h6": MarkdownH6,
        "hr": MarkdownHorizontalRule,
        "paragraph_open": MarkdownParagraph,
        "blockquote_open": MarkdownBlockQuote,
        "bullet_list_open": MarkdownBulletList,
        "ordered_list_open": MarkdownOrderedList,
        "list_item_ordered_open": MarkdownOrderedListItem,
        "list_item_unordered_open": MarkdownUnorderedListItem,
        "table_open": MarkdownTable,
        "tbody_open": MarkdownTBody,
        "thead_open": MarkdownTHead,
        "tr_open": MarkdownTR,
        "th_open": MarkdownTH,
        "td_open": MarkdownTD,
        "fence": MarkdownFence,
        "code_block": MarkdownFence,
    }

    DEFAULT_CSS = (
        Markdown.DEFAULT_CSS
        + """
    ToolAwareMarkdown {
        width: 1fr;
        height: auto;
        padding: 0 1 1 1;
        color: $text;
    }

    ToolAwareMarkdown MarkdownFence {
        margin: 1 0;
        border: round $accent 15%;
        background: $surface-darken-1;
    }

    ToolAwareMarkdown MarkdownFence > Label {
        padding: 1 2;
    }

    ToolAwareMarkdown MarkdownBlockQuote {
        border-left: wide $accent 30%;
        padding-left: 1;
    }
    """
    )

    def append_async(self, markdown: str) -> AwaitComplete:
        """Append markdown while parsing off the UI thread."""
        if not markdown:
            return AwaitComplete.nothing()

        parser = (
            MarkdownIt("gfm-like")
            if self._parser_factory is None
            else self._parser_factory()
        )

        async def await_append() -> None:
            async with self.lock:
                full_markdown = self.source + markdown
                updated_source = "".join(
                    full_markdown.splitlines(keepends=True)[self._last_parsed_line :]
                )
                tokens = await asyncio.get_running_loop().run_in_executor(
                    None, parser.parse, updated_source
                )

                self._markdown = full_markdown
                existing_blocks = [
                    child for child in self.children if isinstance(child, MarkdownBlock)
                ]
                start_line = self._last_parsed_line

                for token in reversed(tokens):
                    if token.map is not None and token.level == 0:
                        self._last_parsed_line += token.map[0]
                        break

                new_blocks = list(self._parse_markdown(tokens))
                any_headers = any(
                    isinstance(block, MarkdownHeader) for block in new_blocks
                )
                for block in new_blocks:
                    start, end = block.source_range
                    block.source_range = (
                        start + start_line,
                        end + start_line,
                    )

                with self.app.batch_update():
                    if existing_blocks and new_blocks:
                        last_block = existing_blocks[-1]
                        last_block.source_range = new_blocks[0].source_range
                        try:
                            await last_block._update_from_block(new_blocks[0])
                        except IndexError:
                            pass
                        else:
                            new_blocks = new_blocks[1:]

                    if new_blocks:
                        await self.mount_all(new_blocks)

                if any_headers:
                    self._table_of_contents = None
                    self.post_message(
                        Markdown.TableOfContentsUpdated(
                            self, self.table_of_contents
                        ).set_sender(self)
                    )

        return AwaitComplete(await_append())

    def _parse_markdown(self, tokens: Iterable[Token]) -> Iterable[MarkdownBlock]:
        """Create markdown blocks, swapping tool payload fences for collapsibles."""
        stack: list[MarkdownBlock] = []
        stack_append = stack.append
        get_block_class = self.get_block_class

        for token in tokens:
            token_type = token.type
            if token_type == "heading_open":
                stack_append(get_block_class(token.tag)(self, token))
            elif token_type == "hr":
                yield get_block_class("hr")(self, token)
            elif token_type == "paragraph_open":
                stack_append(get_block_class("paragraph_open")(self, token))
            elif token_type == "blockquote_open":
                stack_append(get_block_class("blockquote_open")(self, token))
            elif token_type == "bullet_list_open":
                stack_append(get_block_class("bullet_list_open")(self, token))
            elif token_type == "ordered_list_open":
                stack_append(get_block_class("ordered_list_open")(self, token))
            elif token_type == "list_item_open":
                if token.info:
                    stack_append(
                        get_block_class("list_item_ordered_open")(
                            self, token, token.info
                        )
                    )
                else:
                    item_count = sum(
                        1
                        for block in stack
                        if isinstance(block, MarkdownUnorderedListItem)
                    )
                    stack_append(
                        get_block_class("list_item_unordered_open")(
                            self,
                            token,
                            self.BULLETS[item_count % len(self.BULLETS)],
                        )
                    )
            elif token_type == "table_open":
                stack_append(get_block_class("table_open")(self, token))
            elif token_type == "tbody_open":
                stack_append(get_block_class("tbody_open")(self, token))
            elif token_type == "thead_open":
                stack_append(get_block_class("thead_open")(self, token))
            elif token_type == "tr_open":
                stack_append(get_block_class("tr_open")(self, token))
            elif token_type == "th_open":
                stack_append(get_block_class("th_open")(self, token))
            elif token_type == "td_open":
                stack_append(get_block_class("td_open")(self, token))
            elif token_type.endswith("_close"):
                block = stack.pop()
                if token.type == "heading_close":
                    block.id = (
                        f"heading-{slug_for_tcss_id(block._content.plain)}-{id(block)}"
                    )
                if stack:
                    stack[-1]._blocks.append(block)
                else:
                    yield block
            elif token_type == "inline":
                stack[-1].build_from_token(token)
            elif token_type in ("fence", "code_block"):
                code = token.content.rstrip()
                payload = _detect_tool_payload(token.info, code)
                fence: MarkdownBlock
                if payload is None:
                    fence_class = get_block_class(token_type)
                    assert issubclass(fence_class, MarkdownFence)
                    fence = fence_class(self, token, code)
                else:
                    fence = ToolUseFence(self, token, code, payload)
                if stack:
                    stack[-1]._blocks.append(fence)
                else:
                    yield fence
            else:
                external = self.unhandled_token(token)
                if external is not None:
                    if stack:
                        stack[-1]._blocks.append(external)
                    else:
                        yield external


class AsyncMarkdownWidget(VerticalScroll, can_focus=True, can_focus_children=True):
    """High-throughput markdown stream widget with collapsible tool payloads."""

    DEFAULT_CSS = """
    AsyncMarkdownWidget {
        height: 1fr;
        width: 1fr;
        background: $surface;
        color: $text;
        border: tall $primary;
        scrollbar-gutter: stable;
        padding: 0 0 1 0;
    }

    AsyncMarkdownWidget:focus-within {
        border: tall $accent;
    }
    """

    def __init__(
        self,
        markdown: str | None = None,
        *,
        batch_interval: float = 0.05,
        auto_scroll: bool = True,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )
        self.batch_interval = max(0.01, batch_interval)
        self.auto_scroll = auto_scroll
        self._initial_markdown = markdown or ""
        self._rendered_source = self._initial_markdown
        self._pending_chunks: deque[str] = deque()
        self._pending_fence_tail = ""
        self._pending_char_count = 0
        self._flush_timer: Timer | None = None
        self._flush_scheduled = False
        self._flush_in_progress = False
        self._flush_requested = False
        self._force_flush_requested = False

    @property
    def document(self) -> ToolAwareMarkdown:
        """The inner markdown document widget."""
        return self.query_one(ToolAwareMarkdown)

    @property
    def source(self) -> str:
        """Current rendered and buffered source."""
        return (
            self._rendered_source
            + self._pending_fence_tail
            + "".join(self._pending_chunks)
        )

    def compose(self) -> ComposeResult:
        yield ToolAwareMarkdown(self._initial_markdown, id="async-markdown-document")

    def append_chunk(self, chunk: str) -> None:
        """Queue a markdown fragment for batched rendering."""
        if not chunk:
            return
        self._pending_chunks.append(chunk)
        self._pending_char_count += len(chunk)
        if self.is_mounted:
            self._ensure_flush_timer()

    def append_chunks(self, chunks: Iterable[str]) -> None:
        """Queue multiple markdown fragments for batched rendering."""
        appended = False
        for chunk in chunks:
            if not chunk:
                continue
            self._pending_chunks.append(chunk)
            self._pending_char_count += len(chunk)
            appended = True
        if appended and self.is_mounted:
            self._ensure_flush_timer()

    def set_markdown(self, markdown: str) -> AwaitComplete:
        """Replace the current document contents."""
        self._pending_chunks.clear()
        self._pending_fence_tail = ""
        self._pending_char_count = 0
        self._rendered_source = markdown
        if self._flush_timer is not None:
            self._flush_timer.pause()
        self._flush_scheduled = False
        if not self.is_mounted:
            self._initial_markdown = markdown
            return AwaitComplete.nothing()
        return self.document.update(markdown)

    def flush(self, *, force: bool = False) -> AwaitComplete:
        """Flush queued fragments into the markdown document."""
        return AwaitComplete(self._flush_pending(force=force))

    async def _on_mount(self) -> None:
        self._flush_timer = self.set_interval(
            self.batch_interval,
            self._flush_pending,
            pause=True,
        )
        if self._pending_chunks:
            self._ensure_flush_timer()

    async def _on_unmount(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None

    def _ensure_flush_timer(self) -> None:
        if self._flush_timer is None:
            return
        if not self._flush_scheduled:
            self._flush_timer.reset()
            self._flush_timer.resume()
            self._flush_scheduled = True

    def _pause_flush_timer(self) -> None:
        if self._flush_timer is None:
            return
        self._flush_timer.pause()
        self._flush_scheduled = False

    async def _flush_pending(self, force: bool = False) -> None:
        if self._flush_in_progress:
            self._flush_requested = True
            self._force_flush_requested = self._force_flush_requested or force
            return

        self._flush_in_progress = True
        try:
            queued = "".join(self._pending_chunks)
            self._pending_chunks.clear()
            self._pending_char_count = 0
            combined = self._pending_fence_tail + queued

            if not combined:
                if not self._pending_fence_tail:
                    self._pause_flush_timer()
                return

            if force:
                stable_markdown = combined
                pending_tail = ""
            else:
                stable_markdown, pending_tail = _split_stable_markdown(combined)

            self._pending_fence_tail = pending_tail
            if not stable_markdown:
                if not self._pending_chunks:
                    self._pause_flush_timer()
                return

            if self._rendered_source:
                await self.document.append_async(stable_markdown)
                self._rendered_source += stable_markdown
            else:
                await self.document.update(stable_markdown)
                self._rendered_source = stable_markdown

            if self.auto_scroll:
                self.scroll_end(animate=False, immediate=True, force=True, x_axis=False)
        finally:
            self._flush_in_progress = False
            follow_up_force = self._force_flush_requested
            follow_up_requested = self._flush_requested or bool(self._pending_chunks)
            self._force_flush_requested = False
            self._flush_requested = False

            if follow_up_force:
                await self._flush_pending(force=True)
            elif follow_up_requested:
                self._ensure_flush_timer()
            elif not self._pending_fence_tail:
                self._pause_flush_timer()


def _split_stable_markdown(markdown: str) -> tuple[str, str]:
    """Split markdown into a renderable prefix and an incomplete fence tail."""
    stack: list[tuple[str, int]] = []
    for match in _FENCE_LINE_RE.finditer(markdown):
        fence = match.group("fence")
        if stack:
            open_fence, _start = stack[-1]
            if open_fence[0] == fence[0] and len(fence) >= len(open_fence):
                stack.pop()
                continue
        stack.append((fence, match.start()))

    if not stack:
        return markdown, ""

    _fence, start = stack[-1]
    return markdown[:start], markdown[start:]


def _detect_tool_payload(language_hint: str, code: str) -> ToolPayload | None:
    """Detect tool-use JSON/XML payloads and describe how they should render."""
    normalized_language = _normalize_language(language_hint)

    json_payload = _detect_json_tool_payload(code)
    if json_payload is not None:
        return ToolPayload(language="json", title=json_payload.title, call_count=json_payload.call_count)

    xml_payload = _detect_xml_tool_payload(code)
    if xml_payload is not None:
        return ToolPayload(language="xml", title=xml_payload.title, call_count=xml_payload.call_count)

    if normalized_language in _JSON_LANGUAGE_HINTS or normalized_language in _XML_LANGUAGE_HINTS:
        return None

    return None


def _detect_json_tool_payload(code: str) -> ToolPayload | None:
    stripped = code.strip()
    if not stripped or stripped[0] not in "[{":
        return None

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    match_count, names = _scan_json_tool_payload(payload)
    if match_count == 0:
        return None
    return ToolPayload(
        language="json",
        title=_build_tool_title(names, match_count, "JSON"),
        call_count=match_count,
    )


def _detect_xml_tool_payload(code: str) -> ToolPayload | None:
    stripped = code.strip()
    if not stripped.startswith("<"):
        return None

    try:
        root = ElementTree.fromstring(stripped)
    except ElementTree.ParseError:
        return None

    match_count, names = _scan_xml_tool_payload(root)
    if match_count == 0:
        return None
    return ToolPayload(
        language="xml",
        title=_build_tool_title(names, match_count, "XML"),
        call_count=match_count,
    )


def _scan_json_tool_payload(payload: Any) -> tuple[int, list[str]]:
    matches = 0
    names: list[str] = []

    if isinstance(payload, dict):
        lowered = {str(key).lower(): value for key, value in payload.items()}
        tool_uses = lowered.get("tool_uses")
        if isinstance(tool_uses, list):
            for item in tool_uses:
                child_matches, child_names = _scan_json_tool_payload(item)
                matches += child_matches or 1
                names.extend(child_names)
        elif _dict_looks_like_tool_call(lowered):
            matches += 1
            name = _extract_tool_name(lowered)
            if name:
                names.append(name)

        for key, value in lowered.items():
            if key == "tool_uses":
                continue
            child_matches, child_names = _scan_json_tool_payload(value)
            matches += child_matches
            names.extend(child_names)

    elif isinstance(payload, list):
        for item in payload:
            child_matches, child_names = _scan_json_tool_payload(item)
            matches += child_matches
            names.extend(child_names)

    return matches, _dedupe(names)


def _scan_xml_tool_payload(root: ElementTree.Element) -> tuple[int, list[str]]:
    matches = 0
    names: list[str] = []

    for element in root.iter():
        tag = _normalize_xml_name(element.tag)
        attrs = {_normalize_xml_name(key): value for key, value in element.attrib.items()}
        text = (element.text or "").strip()

        if tag in {"tool", "tool_call", "tool_use", "function", "function_call", "call"}:
            matches += 1

        for key in _TOOL_NAME_KEYS:
            value = attrs.get(key)
            if isinstance(value, str) and value:
                if _looks_like_tool_name(value):
                    names.append(value)
                    matches += 1

        if tag in _TOOL_NAME_KEYS and text and _looks_like_tool_name(text):
            names.append(text)
            matches += 1

        if any(key in attrs for key in _ARGUMENT_KEYS):
            name = attrs.get("recipient_name") or attrs.get("tool") or attrs.get("function")
            if isinstance(name, str) and _looks_like_tool_name(name):
                names.append(name)
                matches += 1

    return matches, _dedupe(names)


def _dict_looks_like_tool_call(payload: dict[str, Any]) -> bool:
    keys = set(payload)
    if "recipient_name" in keys and "parameters" in keys:
        return True
    if "tool" in keys and keys & _ARGUMENT_KEYS:
        return True
    if "function" in keys and keys & _ARGUMENT_KEYS:
        return True
    if payload.get("type") == "function" and "name" in keys and keys & _ARGUMENT_KEYS:
        return _looks_like_tool_name(str(payload.get("name", "")))
    if any(key.startswith("mcp__") for key in keys):
        return True

    name = _extract_tool_name(payload)
    if name and keys & _ARGUMENT_KEYS:
        return True
    return False


def _extract_tool_name(payload: dict[str, Any]) -> str | None:
    for key in _TOOL_NAME_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    name = payload.get("name")
    if isinstance(name, str) and _looks_like_tool_name(name):
        return name
    return None


def _build_tool_title(names: list[str], call_count: int, label: str) -> str:
    unique_names = _dedupe(names)
    if len(unique_names) == 1:
        return f"Tool Use · {unique_names[0]}"
    if unique_names:
        return f"Tool Use · {len(unique_names)} calls"
    if call_count > 1:
        return f"Tool Use · {call_count} {label} calls"
    return f"Tool Use · {label} payload"


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _normalize_language(language_hint: str) -> str:
    head = language_hint.strip().split(None, 1)[0] if language_hint.strip() else ""
    return head.lower()


def _normalize_xml_name(value: str) -> str:
    return value.rsplit("}", 1)[-1].replace("-", "_").lower()


def _looks_like_tool_name(value: str) -> bool:
    lowered = value.lower()
    return any(hint in lowered for hint in _TOOL_HINTS)


__all__ = [
    "AsyncMarkdownWidget",
    "ToolAwareMarkdown",
    "ToolUseFence",
]

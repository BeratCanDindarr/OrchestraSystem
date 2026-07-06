# Plan: TUI consolidation — remove dead versions, fix silent errors

## Status
Draft

## Goal
Reduce the TUI to the two shipped variants (v4 default, v2 `--classic`), eliminate silent
exception swallowing in widgets, and make the markdown parser maintainable.

## Context
Audit findings H-8, LOW dead-code, depth-19 parser (docs/quality-audit.md). `tui/app.py` plus
the root-level widgets `agent_tree.py`, `header_bar.py`, `stream_view.py` (~280 lines) are
unreachable from the entry points. `widgets/v4/chat.py` hides widget lifecycle errors behind
four `except: pass` blocks — the exact class of bug (NONE alias, freezes) recently debugged
blind.

## Phases

### Phase 1: Delete dead code
- [ ] Remove `tui/app.py`, `tui/widgets/agent_tree.py`, `header_bar.py`, `stream_view.py`;
  clean `tui/__init__.py` exports
- [ ] Verify: `orchestra dashboard` (v4) and `--classic` (v2) both launch
- **Acceptance:** grep finds no imports of removed modules; both UIs boot.

### Phase 2: Un-silence widget errors
- [ ] `widgets/v4/chat.py:95,126,132,224` and `widgets/v2/navigator.py:230` — replace bare
  `except:` with the specific expected exceptions (`textual.css.query.NoMatches`,
  `AttributeError` where a widget may be unmounted), logging at debug level
- **Acceptance:** TUI e2e tests (`tests/test_tui_e2e.py`, maestro tests) green; intentional
  error injection surfaces in logs instead of vanishing.

### Phase 3: Markdown parser dispatch refactor
- [ ] `widgets/v3/markdown_view.py:270` `_parse_markdown` (85 lines, nesting depth 19) →
  dispatch table: `handlers = {"fence": _handle_fence, "list": _handle_list, ...}`; each handler
  ≤30 lines, nesting ≤3
- [ ] Snapshot tests: render a corpus of markdown fixtures (code fences, tool payloads, nested
  lists) before/after; outputs must be identical
- **Acceptance:** snapshot parity; AST metric scan shows no depth-19 hotspot.

## Dependencies
- None; independent of engine/cli refactors.

## Risks
- Textual version sensitivity in exception types → pin the exceptions actually raised by the
  installed textual version; verify with the e2e suite.

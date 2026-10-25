# Interactive Chat REPL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `orchestra chat [alias]` — an interactive terminal REPL that converses with any resolved model (local Ollama, `model@backend`, or `alias@node`), with persisted sessions, slash commands, and streaming.

**Architecture:** A new `orchestra/chat/` package: `store.py` (JSONL persistence), `history.py` (concatenated-prompt rendering), `session.py` (ChatSession state), `commands.py` (slash dispatch), `loop.py` (injectable REPL). `resolve()` is reused per turn. `cli.py` gets a `chat` command.

**Tech Stack:** Python 3.11+, Typer CLI, rich console, stdlib (json/datetime/uuid). No new deps. No Rust. Reuses `fallback.resolve` and the `BaseProvider.stream`/`run` contract.

## Global Constraints

- Pure Python; NO new dependencies; NO Rust.
- Reuse `fallback.resolve(alias) -> (provider, effort)` unchanged; it already handles `@backend` and `@node`.
- Context strategy is a single concatenated prompt (`User: …\nAssistant: …\nUser: <new>`), NOT native message arrays (non-goal for v1) — must work identically on Ollama `/api/generate`, subprocess CLIs, and the remote node's one-shot `/run`.
- Streaming selection: try `provider.stream(...)`; on `NotImplementedError` fall back to `provider.run(...)`. Do not add per-provider flags (`BaseProvider.stream` raises NotImplementedError; only Ollama + Claude SDK override).
- Sessions persist to `.orchestra/chat_sessions/<id>.jsonl` (append-only): line 1 = meta `{"session_id","alias","model","created_at"}`, each later line = `{"role","content","ts"}`. `.orchestra/` is gitignored. Use `config.orchestra_root()` to locate `.orchestra/`.
- `/model` switching is session-scoped; the meta line is NOT rewritten; `--resume` reopens with `meta.alias`.
- The REPL loop must accept injectable `input_fn`/`output_fn` (default `input`/`print`) so it is unit-testable without a terminal.
- No silent exception swallowing; functions <50 lines; files <400 typical; immutability where practical.
- Run pytest FROM A SCRATCH DIR: `cd /tmp && /abs/.venv/bin/pytest <abs paths> -q --no-cov -p no:cacheprovider`.
- Conventional Commits; NO Co-Authored-By lines.

## File Structure

- `orchestra/chat/__init__.py` — package marker (empty).
- `orchestra/chat/history.py` — `render_history(history) -> str` (concatenated prompt).
- `orchestra/chat/store.py` — JSONL persistence: `new_session_id`, `session_path`, `write_meta`, `append_message`, `load_session`, `list_sessions`.
- `orchestra/chat/session.py` — `ChatSession` dataclass-ish holder (id, alias, provider, effort, history) + `send(message) -> str` (renders history, streams/runs, records both turns).
- `orchestra/chat/commands.py` — slash dispatch: `is_command`, `handle_command(session, line, output_fn) -> str` returning a control token ("continue"/"exit").
- `orchestra/chat/loop.py` — `run_repl(session, input_fn=input, output_fn=print)`.
- `orchestra/cli.py` — `chat` command (`alias` arg optional, `--resume`, `--list`).
- Tests: `tests/test_chat_store.py`, `tests/test_chat_history.py`, `tests/test_chat_session.py`, `tests/test_chat_commands.py`, `tests/test_chat_loop.py`, `tests/test_cli_chat.py`.

---

### Task 1: history rendering

**Files:**
- Create: `orchestra/chat/__init__.py` (empty), `orchestra/chat/history.py`
- Test: `tests/test_chat_history.py`

**Interfaces:**
- Produces: `render_history(history: list[dict]) -> str` where each item is `{"role": "user"|"assistant", "content": str}`. Renders `User:`/`Assistant:` lines separated by blank lines; the final line is always the last user message with a trailing `Assistant:` cue so the model continues. Empty history with a single user turn renders just that user message.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chat_history.py
from orchestra.chat.history import render_history


def test_single_user_turn():
    h = [{"role": "user", "content": "hello"}]
    assert render_history(h) == "User: hello\n\nAssistant:"


def test_multi_turn_prompt_ends_with_last_user_and_assistant_cue():
    h = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
        {"role": "user", "content": "how are you?"},
    ]
    out = render_history(h)
    assert out == (
        "User: hi\n\n"
        "Assistant: hello there\n\n"
        "User: how are you?\n\n"
        "Assistant:"
    )


def test_empty_history_is_empty_string():
    assert render_history([]) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_chat_history.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`No module named 'orchestra.chat'`).

- [ ] **Step 3: Create the package + history renderer**

Create empty `orchestra/chat/__init__.py`. Create `orchestra/chat/history.py`:

```python
"""Render conversation history into a single concatenated prompt (portable across providers)."""
from __future__ import annotations

_ROLE_LABEL = {"user": "User", "assistant": "Assistant"}


def render_history(history: list[dict]) -> str:
    """Turn [{role, content}, ...] into 'User: …\\n\\nAssistant: …\\n\\nUser: …\\n\\nAssistant:'."""
    if not history:
        return ""
    lines = [f"{_ROLE_LABEL.get(m['role'], m['role'].title())}: {m['content']}" for m in history]
    return "\n\n".join(lines) + "\n\nAssistant:"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_chat_history.py -q --no-cov -p no:cacheprovider`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add orchestra/chat/__init__.py orchestra/chat/history.py tests/test_chat_history.py
git commit -m "feat(chat): concatenated-prompt history renderer"
```

---

### Task 2: JSONL session store

**Files:**
- Create: `orchestra/chat/store.py`
- Test: `tests/test_chat_store.py`

**Interfaces:**
- Consumes: `config.orchestra_root()`.
- Produces:
  - `sessions_dir() -> Path` = `config.orchestra_root() / "chat_sessions"` (created on demand).
  - `new_session_id(alias: str) -> str` = `<YYYYMMDD>-<slug>-<4hex>` (slug = alias lowercased, non-alnum → "").
  - `session_path(session_id) -> Path`.
  - `write_meta(session_id, alias, model) -> None` (writes line 1; creates file).
  - `append_message(session_id, role, content) -> None` (appends a `{"role","content","ts"}` line).
  - `load_session(session_id) -> tuple[dict, list[dict]]` = (meta, messages). Raises `FileNotFoundError` with a helpful message listing available ids if missing; skips corrupt lines.
  - `list_sessions() -> list[dict]` = `[{"session_id","alias","created_at","first_user"}]`, newest first.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chat_store.py
from unittest.mock import patch
import pytest
from orchestra.chat import store


def _patch_root(tmp_path):
    return patch("orchestra.config.orchestra_root", return_value=tmp_path)


def test_new_session_id_shape(tmp_path):
    with _patch_root(tmp_path):
        sid = store.new_session_id("cdx-deep@windows")
    parts = sid.split("-")
    assert len(parts) == 3
    assert parts[0].isdigit() and len(parts[0]) == 8      # YYYYMMDD
    assert parts[1] == "cdxdeepwindows"                    # slug, non-alnum stripped
    assert len(parts[2]) == 4                              # short hex


def test_write_and_load_round_trip(tmp_path):
    with _patch_root(tmp_path):
        sid = "20260713-x-aaaa"
        store.write_meta(sid, "oll-fast", "ollama/qwen2.5:7b")
        store.append_message(sid, "user", "hi")
        store.append_message(sid, "assistant", "hello")
        meta, msgs = store.load_session(sid)
    assert meta["alias"] == "oll-fast"
    assert meta["model"] == "ollama/qwen2.5:7b"
    assert [m["content"] for m in msgs] == ["hi", "hello"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_load_missing_raises_with_available_ids(tmp_path):
    with _patch_root(tmp_path):
        store.write_meta("20260713-a-bbbb", "oll-fast", "m")
        with pytest.raises(FileNotFoundError) as exc:
            store.load_session("nope")
    assert "20260713-a-bbbb" in str(exc.value)


def test_list_sessions_returns_snippet(tmp_path):
    with _patch_root(tmp_path):
        store.write_meta("20260713-a-cccc", "oll-fast", "m")
        store.append_message("20260713-a-cccc", "user", "first question here")
        listing = store.list_sessions()
    assert listing[0]["session_id"] == "20260713-a-cccc"
    assert listing[0]["alias"] == "oll-fast"
    assert "first question" in listing[0]["first_user"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_chat_store.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`No module named 'orchestra.chat.store'`).

- [ ] **Step 3: Create the store**

```python
# orchestra/chat/store.py
"""JSONL persistence for chat sessions under .orchestra/chat_sessions/."""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from orchestra import config


def sessions_dir() -> Path:
    d = config.orchestra_root() / "chat_sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_session_id(alias: str) -> str:
    slug = re.sub(r"[^a-z0-9]", "", alias.lower()) or "chat"
    return f"{datetime.now().strftime('%Y%m%d')}-{slug}-{uuid.uuid4().hex[:4]}"


def session_path(session_id: str) -> Path:
    return sessions_dir() / f"{session_id}.jsonl"


def write_meta(session_id: str, alias: str, model: str) -> None:
    meta = {
        "session_id": session_id, "alias": alias, "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    session_path(session_id).write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")


def append_message(session_id: str, role: str, content: str) -> None:
    line = {"role": role, "content": content, "ts": datetime.now(timezone.utc).isoformat()}
    with open(session_path(session_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def load_session(session_id: str) -> tuple[dict, list[dict]]:
    path = session_path(session_id)
    if not path.exists():
        avail = ", ".join(s["session_id"] for s in list_sessions()) or "(none)"
        raise FileNotFoundError(f"No chat session {session_id!r}. Available: {avail}")
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    meta: dict = {}
    messages: list[dict] = []
    for i, ln in enumerate(lines):
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if i == 0 and "session_id" in obj:
            meta = obj
        elif "role" in obj:
            messages.append(obj)
    return meta, messages


def list_sessions() -> list[dict]:
    result: list[dict] = []
    for path in sessions_dir().glob("*.jsonl"):
        try:
            meta, msgs = load_session(path.stem)
        except Exception:
            continue
        first_user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        result.append({
            "session_id": meta.get("session_id", path.stem),
            "alias": meta.get("alias", "?"),
            "created_at": meta.get("created_at", ""),
            "first_user": first_user[:60],
        })
    result.sort(key=lambda s: s["created_at"], reverse=True)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_chat_store.py -q --no-cov -p no:cacheprovider`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add orchestra/chat/store.py tests/test_chat_store.py
git commit -m "feat(chat): JSONL session store (write/load/list)"
```

---

### Task 3: ChatSession (send a turn, stream-or-run, persist)

**Files:**
- Create: `orchestra/chat/session.py`
- Test: `tests/test_chat_session.py`

**Interfaces:**
- Consumes: `render_history` (Task 1), `store.append_message` (Task 2), `fallback.resolve`.
- Produces: `ChatSession` with attributes `session_id, alias, provider, effort, history: list[dict]`;
  classmethod `start(alias, session_id=None) -> ChatSession` (resolves alias, writes meta if new);
  method `send(message, on_chunk=None) -> str` — appends user turn to history + store, renders the
  prompt, streams via `provider.stream` (calling `on_chunk(text)` per chunk) or falls back to
  `provider.run` on `NotImplementedError`, appends the assistant reply to history + store, returns
  the full reply; method `switch_model(alias) -> None` (re-resolve, keep history).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chat_session.py
from unittest.mock import patch
from orchestra.chat.session import ChatSession


class _StreamProvider:
    name = "mock"
    def model_label(self, e): return f"mock/{e}"
    def stream(self, prompt, effort, timeout=180, **k):
        assert "User: hi" in prompt
        yield "hel"; yield "lo"


class _RunOnlyProvider:
    name = "mockrun"
    def model_label(self, e): return f"mockrun/{e}"
    def stream(self, prompt, effort, timeout=180, **k):
        raise NotImplementedError
    def run(self, prompt, effort, timeout=180, **k):
        return "one-shot reply", 0


def _patch_root(tmp_path):
    return patch("orchestra.config.orchestra_root", return_value=tmp_path)


def test_send_streams_and_records(tmp_path):
    with _patch_root(tmp_path), patch("orchestra.chat.session.resolve", return_value=(_StreamProvider(), "fast")):
        s = ChatSession.start("mock-alias")
        chunks = []
        reply = s.send("hi", on_chunk=chunks.append)
    assert reply == "hello"
    assert chunks == ["hel", "lo"]
    assert s.history[-2:] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_send_falls_back_to_run(tmp_path):
    with _patch_root(tmp_path), patch("orchestra.chat.session.resolve", return_value=(_RunOnlyProvider(), "fast")):
        s = ChatSession.start("mock-alias")
        reply = s.send("hi")
    assert reply == "one-shot reply"


def test_switch_model_keeps_history(tmp_path):
    with _patch_root(tmp_path), patch("orchestra.chat.session.resolve", return_value=(_StreamProvider(), "fast")):
        s = ChatSession.start("mock-alias")
        s.send("hi")
        with patch("orchestra.chat.session.resolve", return_value=(_RunOnlyProvider(), "slow")) as r:
            s.switch_model("other")
        assert r.called
        assert len(s.history) == 2  # preserved
        assert s.alias == "other"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_chat_session.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`No module named 'orchestra.chat.session'`).

- [ ] **Step 3: Create ChatSession**

```python
# orchestra/chat/session.py
"""ChatSession: one conversation with a resolved provider, streamed and persisted."""
from __future__ import annotations

from orchestra.chat import store
from orchestra.chat.history import render_history
from orchestra.providers.fallback import resolve


class ChatSession:
    def __init__(self, session_id: str, alias: str, provider, effort: str, history: list[dict]):
        self.session_id = session_id
        self.alias = alias
        self.provider = provider
        self.effort = effort
        self.history = history

    @classmethod
    def start(cls, alias: str, session_id: str | None = None) -> "ChatSession":
        provider, effort = resolve(alias)
        if session_id is None:
            session_id = store.new_session_id(alias)
            store.write_meta(session_id, alias, provider.model_label(effort))
        return cls(session_id, alias, provider, effort, [])

    @classmethod
    def resume(cls, session_id: str) -> "ChatSession":
        meta, messages = store.load_session(session_id)
        alias = meta.get("alias", "cdx-deep")
        provider, effort = resolve(alias)
        return cls(session_id, alias, provider, effort, messages)

    def switch_model(self, alias: str) -> None:
        self.provider, self.effort = resolve(alias)
        self.alias = alias

    def send(self, message: str, on_chunk=None) -> str:
        self.history.append({"role": "user", "content": message})
        store.append_message(self.session_id, "user", message)
        prompt = render_history(self.history)
        reply = self._generate(prompt, on_chunk)
        self.history.append({"role": "assistant", "content": reply})
        store.append_message(self.session_id, "assistant", reply)
        return reply

    def _generate(self, prompt: str, on_chunk) -> str:
        try:
            parts: list[str] = []
            for chunk in self.provider.stream(prompt, self.effort):
                parts.append(chunk)
                if on_chunk:
                    on_chunk(chunk)
            return "".join(parts)
        except NotImplementedError:
            reply, _ = self.provider.run(prompt, self.effort)
            if on_chunk:
                on_chunk(reply)
            return reply
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_chat_session.py -q --no-cov -p no:cacheprovider`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add orchestra/chat/session.py tests/test_chat_session.py
git commit -m "feat(chat): ChatSession — send turn, stream-or-run, persist, switch model"
```

---

### Task 4: slash commands

**Files:**
- Create: `orchestra/chat/commands.py`
- Test: `tests/test_chat_commands.py`

**Interfaces:**
- Consumes: `ChatSession` (Task 3), `store.new_session_id`/`write_meta` (Task 2), `available_aliases`.
- Produces: `is_command(line: str) -> bool` (line stripped starts with "/"); `handle_command(session, line, output_fn) -> str` returning "exit" for `/exit`, else "continue". Handles `/model <alias>`, `/clear`, `/new`, `/models`, `/help`, `/exit`; unknown → output "unknown command" hint, return "continue".

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chat_commands.py
from unittest.mock import patch, MagicMock
from orchestra.chat import commands


def _session():
    s = MagicMock()
    s.history = [{"role": "user", "content": "x"}]
    s.session_id = "20260713-a-aaaa"
    s.alias = "oll-fast"
    return s


def test_is_command():
    assert commands.is_command("/model x")
    assert not commands.is_command("hello")


def test_exit_returns_exit():
    out = []
    assert commands.handle_command(_session(), "/exit", out.append) == "exit"


def test_model_switches():
    s = _session()
    assert commands.handle_command(s, "/model gmn-pro", lambda *_: None) == "continue"
    s.switch_model.assert_called_once_with("gmn-pro")


def test_clear_empties_history():
    s = _session()
    commands.handle_command(s, "/clear", lambda *_: None)
    assert s.history == []


def test_unknown_command_hints():
    out = []
    assert commands.handle_command(_session(), "/wat", out.append) == "continue"
    assert any("unknown" in o.lower() for o in out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_chat_commands.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`No module named 'orchestra.chat.commands'`).

- [ ] **Step 3: Create commands**

```python
# orchestra/chat/commands.py
"""Slash-command dispatch for the chat REPL."""
from __future__ import annotations

_HELP = (
    "/model <alias>  switch model (history kept)\n"
    "/clear          clear conversation history\n"
    "/new            start a new session\n"
    "/models         list available aliases\n"
    "/help           this help\n"
    "/exit           quit"
)


def is_command(line: str) -> bool:
    return line.strip().startswith("/")


def handle_command(session, line: str, output_fn) -> str:
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0]
    arg = parts[1].strip() if len(parts) > 1 else ""
    if cmd == "/exit":
        return "exit"
    if cmd == "/help":
        output_fn(_HELP)
    elif cmd == "/clear":
        session.history.clear()
        output_fn("history cleared")
    elif cmd == "/model":
        if not arg:
            output_fn("usage: /model <alias>")
        else:
            try:
                session.switch_model(arg)
                output_fn(f"switched to {arg}")
            except Exception as exc:
                output_fn(f"cannot switch to {arg!r}: {exc}")
    elif cmd == "/new":
        from orchestra.chat import store
        session.session_id = store.new_session_id(session.alias)
        store.write_meta(session.session_id, session.alias, session.provider.model_label(session.effort))
        session.history.clear()
        output_fn(f"new session {session.session_id}")
    elif cmd == "/models":
        from orchestra.providers.fallback import available_aliases
        for a in available_aliases():
            output_fn(a)
    else:
        output_fn(f"unknown command {cmd!r}; type /help")
    return "continue"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_chat_commands.py -q --no-cov -p no:cacheprovider`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add orchestra/chat/commands.py tests/test_chat_commands.py
git commit -m "feat(chat): slash-command dispatch (/model /clear /new /models /help /exit)"
```

---

### Task 5: REPL loop (injectable IO)

**Files:**
- Create: `orchestra/chat/loop.py`
- Test: `tests/test_chat_loop.py`

**Interfaces:**
- Consumes: `ChatSession` (Task 3), `commands.is_command`/`handle_command` (Task 4).
- Produces: `run_repl(session, input_fn=input, output_fn=print) -> None`. Loops: read a line via
  `input_fn`; `EOFError`/None → exit; empty → skip; slash → `handle_command` (stop on "exit");
  else `session.send(msg, on_chunk=lambda t: output_fn(t, end=""))` then a trailing newline. A
  `KeyboardInterrupt` during a turn cancels that turn and continues the loop.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chat_loop.py
from unittest.mock import MagicMock
from orchestra.chat.loop import run_repl


def test_loop_sends_then_exits():
    session = MagicMock()
    session.send.return_value = "reply"
    inputs = iter(["hello", "/exit"])
    def _in(_prompt=""):
        return next(inputs)
    outputs = []
    run_repl(session, input_fn=_in, output_fn=lambda *a, **k: outputs.append(a[0] if a else ""))
    session.send.assert_called_once()
    assert session.send.call_args[0][0] == "hello"


def test_loop_skips_empty_and_stops_on_eof():
    session = MagicMock()
    seq = iter(["", "   "])
    def _in(_prompt=""):
        try:
            return next(seq)
        except StopIteration:
            raise EOFError
    run_repl(session, input_fn=_in, output_fn=lambda *a, **k: None)
    session.send.assert_not_called()


def test_loop_keyboardinterrupt_cancels_turn_not_loop():
    session = MagicMock()
    calls = {"n": 0}
    def _send(msg, on_chunk=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        return "ok"
    session.send.side_effect = _send
    seq = iter(["first", "second", "/exit"])
    run_repl(session, input_fn=lambda _p="": next(seq), output_fn=lambda *a, **k: None)
    assert calls["n"] == 2  # first cancelled, second completed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_chat_loop.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`No module named 'orchestra.chat.loop'`).

- [ ] **Step 3: Create the loop**

```python
# orchestra/chat/loop.py
"""Interactive REPL loop with injectable IO for testing."""
from __future__ import annotations

from orchestra.chat import commands


def run_repl(session, input_fn=input, output_fn=print) -> None:
    while True:
        try:
            line = input_fn("you> ")
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return
        if line is None:
            return
        if not line.strip():
            continue
        if commands.is_command(line):
            if commands.handle_command(session, line, output_fn) == "exit":
                return
            continue
        output_fn(f"{session.alias}> ", end="")
        try:
            session.send(line, on_chunk=lambda t: output_fn(t, end=""))
            output_fn("")
        except KeyboardInterrupt:
            output_fn("\n[cancelled]")
        except Exception as exc:
            output_fn(f"\n[error] {exc}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_chat_loop.py -q --no-cov -p no:cacheprovider`
Expected: PASS (3 tests). NOTE: the tests use `output_fn` that accepts `*a, **k` because the loop calls it with `end=` — ensure your test doubles tolerate the `end` kwarg.

- [ ] **Step 5: Commit**

```bash
git add orchestra/chat/loop.py tests/test_chat_loop.py
git commit -m "feat(chat): injectable REPL loop (stream output, slash commands, Ctrl-C cancels turn)"
```

---

### Task 6: CLI `chat` command

**Files:**
- Modify: `orchestra/cli.py` (add near the `ask` command)
- Test: `tests/test_cli_chat.py`

**Interfaces:**
- Consumes: `ChatSession.start`/`resume` (Task 3), `store.list_sessions` (Task 2), `run_repl` (Task 5).
- Produces: `orchestra chat [ALIAS] [--resume ID] [--list]`. `--list` prints sessions and returns.
  `--resume ID` resumes. Otherwise `ALIAS` (default `cdx-deep`) starts a new session. Then calls
  `run_repl(session)`.

⚠️ cli.py historically had duplicated command groups — grep for an existing `chat` command and add this ONCE, near `ask`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_chat.py
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from orchestra.cli import app

runner = CliRunner()


def test_chat_list_prints_sessions():
    fake = [{"session_id": "20260713-a-aaaa", "alias": "oll-fast", "created_at": "2026-07-13", "first_user": "hi there"}]
    with patch("orchestra.chat.store.list_sessions", return_value=fake):
        result = runner.invoke(app, ["chat", "--list"])
    assert result.exit_code == 0
    assert "20260713-a-aaaa" in result.output
    assert "oll-fast" in result.output


def test_chat_starts_session_and_runs_repl():
    sess = MagicMock()
    with patch("orchestra.chat.session.ChatSession.start", return_value=sess) as start, \
         patch("orchestra.chat.loop.run_repl") as repl:
        result = runner.invoke(app, ["chat", "oll-fast"])
    assert result.exit_code == 0
    start.assert_called_once_with("oll-fast")
    repl.assert_called_once_with(sess)


def test_chat_resume_calls_resume():
    sess = MagicMock()
    with patch("orchestra.chat.session.ChatSession.resume", return_value=sess) as res, \
         patch("orchestra.chat.loop.run_repl"):
        result = runner.invoke(app, ["chat", "--resume", "20260713-a-aaaa"])
    assert result.exit_code == 0
    res.assert_called_once_with("20260713-a-aaaa")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_cli_chat.py -q --no-cov -p no:cacheprovider`
Expected: FAIL (`No such command 'chat'`).

- [ ] **Step 3: Add the command**

In `orchestra/cli.py`, after the `ask` command:

```python
@app.command()
def chat(
    alias: str = typer.Argument("cdx-deep", help="Model alias (supports model@backend, alias@node)"),
    resume: str = typer.Option("", "--resume", help="Resume a saved session by id"),
    list_sessions: bool = typer.Option(False, "--list", help="List saved chat sessions"),
):
    """Interactive chat REPL with any resolved model."""
    from orchestra.chat import store
    from orchestra.chat.session import ChatSession
    from orchestra.chat.loop import run_repl

    if list_sessions:
        rows = store.list_sessions()
        if not rows:
            console.print("[dim]No saved chat sessions.[/dim]")
            return
        for r in rows:
            console.print(f"  {r['session_id']:<26} {r['alias']:<14} {r['first_user']}")
        return

    try:
        session = ChatSession.resume(resume) if resume else ChatSession.start(alias)
    except (FileNotFoundError, KeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[dim]chat: {session.alias}  (/help for commands, /exit to quit)[/dim]")
    run_repl(session)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp && /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/.venv/bin/pytest /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_cli_chat.py /Users/omnio/Desktop/Projects/OrchestraSystem/OrchestraSystem/tests/test_cli_command_registry.py -q --no-cov -p no:cacheprovider`
Expected: PASS (3 chat tests + registry green, no duplicate command).

- [ ] **Step 5: Commit**

```bash
git add orchestra/cli.py tests/test_cli_chat.py
git commit -m "feat(cli): interactive 'chat' command (--resume, --list)"
```

---

### Task 7: docs + live smoke

**Files:**
- Modify: `CLAUDE.md` (commands), `docs/architecture.md` (a chat note).
- No new test; a local smoke.

- [ ] **Step 1: Add commands to `CLAUDE.md`**

In `## Setup & Commands`:

```
orchestra chat oll-fast                  # interactive REPL with a model (local/remote)
orchestra chat cdx-deep@windows          # chat with a remote node's logged-in agent
orchestra chat --list                    # list saved sessions; --resume <id> to continue
```

- [ ] **Step 2: Architecture note**

In `docs/architecture.md`, add a line under Entry points / a new note: `orchestra/chat/` — interactive REPL (session/store/commands/loop) over `resolve()`, sessions in `.orchestra/chat_sessions/`.

- [ ] **Step 3: Local smoke (real Ollama)**

Run (Ollama must be up):
```
printf 'say hi in 3 words\n/exit\n' | .venv/bin/python -m orchestra chat oll-fast
```
Expected: the model streams a short reply, then exits cleanly; `.orchestra/chat_sessions/` gains one `.jsonl`. Then `.venv/bin/python -m orchestra chat --list` shows it.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/architecture.md
git commit -m "docs(chat): document the interactive chat REPL"
```

---

## Verification (end-to-end)

1. `orchestra chat oll-fast` → streams a live reply; multi-turn context works (ask a follow-up referencing the prior answer).
2. `/model gmn-pro` mid-chat switches model, history preserved; `/clear`, `/new`, `/models`, `/help`, `/exit` behave.
3. Session persisted to `.orchestra/chat_sessions/<id>.jsonl`; `orchestra chat --list` shows it; `--resume <id>` continues it.
4. `orchestra chat cdx-deep@windows` chats through a remote node (one-shot per turn, history re-sent).
5. Existing `orchestra ask` unaffected; full fast suite from a scratch dir shows no new regressions:
   `cd /tmp && /abs/.venv/bin/pytest /abs/tests -m "not slow" -q --no-cov -p no:cacheprovider --continue-on-collection-errors`.

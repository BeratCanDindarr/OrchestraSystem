# Design: Interactive chat REPL (`orchestra chat`)

## Status
Approved (brainstorming complete, 2026-07-13)

## Problem

Every Orchestra invocation is single-shot and stateless (`orchestra ask <alias> "task"` →
new `OrchestraRun`, no memory of prior turns). The user wants an interactive REPL — pick a
model and converse continuously in the terminal, like `ollama run` — across ANY backend the
engine now supports: local Ollama, a remote Ollama backend (`model@backend`), or a remote
node's logged-in agents (`alias@node`). This is the payoff that sits on top of the
multi-backend Ollama and remote-node-gateway features already built.

The engine gives us the pieces: `resolve(alias)` already returns a reusable
`(provider, effort)` and already handles `@backend`/`@node`. What's missing: a conversation
loop, multi-turn context (the engine has ZERO conversation state), session persistence, slash
commands, and streaming output. All of that lives in a new `orchestra/chat/` package + a `chat`
CLI command.

## Goals

- `orchestra chat [alias]` — interactive loop; `resolve(alias)` once, reuse the provider per turn.
  Works with plain aliases, `model@backend`, and `alias@node`.
- Multi-turn context: the REPL keeps history and re-sends it as a single concatenated prompt
  each turn (portable across all providers).
- Persistence: each chat is auto-appended to `.orchestra/chat_sessions/<id>.jsonl`;
  `orchestra chat --resume <id>` continues it; `orchestra chat --list` lists sessions.
- Slash commands: `/model <alias>`, `/clear`, `/new`, `/models`, `/help`, `/exit`.
- Streaming output when the provider supports it (Ollama, Claude SDK), one-shot otherwise.

## Non-Goals (deferred)

- Native multi-turn message APIs (Ollama `/api/chat`, Claude SDK `messages=[...]`). v1 uses a
  concatenated prompt so it works identically on `/api/generate`, subprocess CLIs, and remote
  nodes. Native messages is a later refinement.
- Multi-agent chat (a `dual`/`critical` REPL). v1 is one model at a time.
- Chat search / tagging / export; rich full-screen TUI (removed, not returning).

## Architecture

New package `orchestra/chat/` (small, focused files) + a `chat` CLI command. `resolve()` is
reused unchanged.

```
orchestra chat cdx-deep@windows
  → resolve("cdx-deep@windows") -> (RemoteNodeProvider, "cdx-deep")   [once]
  → loop:
      you> <msg>
        history.append(user, msg)
        prompt = render_history(history)          # "User: …\nAssistant: …\nUser: <msg>"
        reply = stream-or-run(provider, prompt, effort)  # live tokens if provider.stream()
        history.append(assistant, reply)
        store.append_turn(session, user_msg, reply)       # JSONL, auto-saved
```

### 1. Conversation loop + context

- `session.py` — `ChatSession` holds `session_id`, `alias`, `provider`, `effort`, and
  `history: list[dict]` (`{"role": "user"|"assistant", "content": str}`).
- Context strategy: `render_history(history) -> str` concatenates prior turns into one prompt
  (`User: …\nAssistant: …\nUser: <new>`). This is what goes to `provider.stream/run`. Chosen for
  portability — identical behavior on Ollama `/api/generate`, subprocess CLIs, and the remote
  node's one-shot `/run`. (Native messages arrays are a non-goal for v1.)
- `loop.py` — the REPL, written with injectable `input_fn`/`output_fn` (default `input`/`print`)
  so it is unit-testable without a real terminal.

### 2. Persistence

- `store.py` — pure read/write/list functions over `.orchestra/chat_sessions/<id>.jsonl`
  (`.orchestra/` is already gitignored).
- `session_id`: readable + unique — `<YYYYMMDD>-<alias-slug>-<short-random>`, e.g.
  `20260713-cdxdeep-a3f2`. (CLI code — `datetime.now()`/random are allowed here, unlike engine
  runtime.)
- File format: line 1 is meta (`{"session_id","alias","model","created_at"}`); each later line
  is a message (`{"role","content","ts"}`). The meta's `alias` lets `--resume` re-`resolve` the
  right provider.
- Auto-save: after each turn, the user+assistant messages are appended (append-only; survives a
  crash). No separate `/save`.
- `orchestra chat --resume <id>` loads the JSONL, restores history + model, continues.
- `orchestra chat --list` lists sessions: id, date, alias, first-user-message snippet.

### 3. Slash commands

- `commands.py` — a dispatch table; each command is a small function (testable).
- `/model <alias>` — re-`resolve` and swap the provider; **history is preserved** (the new model
  sees prior turns via the concatenated prompt). `@backend`/`@node` valid. The switch is
  session-scoped: the JSONL is append-only, so the meta line (initial model) is NOT rewritten;
  `--resume` reopens with `meta.alias` and the user can `/model` again. (A model-change marker
  line could track the active model across resume later — a non-goal for v1.)
- `/clear` — empty the history (same session file, fresh context).
- `/new` — start a new session (new JSONL file, empty history).
- `/models` — list available aliases/models (like `orchestra aliases`).
- `/help` — command list.
- `/exit` (and Ctrl+D) — clean exit.
- Unknown `/command` → "unknown command, type /help"; never sent to the model. Non-slash input
  always goes to the model.

### 4. Streaming + provider selection + output

- Hybrid: try `provider.stream(prompt, effort)`; on `NotImplementedError` fall back to
  `provider.run(...)`. `BaseProvider.stream` raises `NotImplementedError`; only Ollama and
  Claude SDK override it, so this selects streaming automatically without a per-provider flag.
- Streaming providers (Ollama, Claude SDK) print tokens live (`print(chunk, end="", flush=True)`).
  Non-streaming (CLIs, remote node, OpenAI/DeepSeek) show a brief "thinking…" then the full reply.
- Output: `you> …` / `<alias>> …`, light rich coloring (prompt cyan, model name dim). No heavy UI.

### 5. Error handling & resilience

- Invalid alias (resolve fails) → clear error; chat doesn't start.
- `--resume` on a missing/corrupt file → clear error listing available ids.
- Provider mid-chat error (`[ERROR] …`) → shown; loop continues.
- Ctrl+C cancels the current turn (back to the loop, does NOT exit); Ctrl+D / `/exit` exits
  cleanly (session already persisted); empty input is ignored.

## Testing Strategy (pytest; injectable IO, mocked providers — no real terminal/network)

- `store.py`: write/read/list round-trip in a tmp dir; meta + messages; `--resume` load;
  corrupt-file handling.
- `commands.py`: `/model` swaps the resolved provider; `/clear` empties history; `/new` opens a
  new file; unknown command; dispatch.
- `render_history`: correct `User:`/`Assistant:` format, new message last.
- Provider selection: `stream()` used when present; `NotImplementedError` → `run()` fallback
  (mock providers).
- REPL loop: drive `input_fn` with a scripted sequence ending in `/exit`, mock the provider,
  assert history + file contents.
- CLI: `chat --list`, `chat --resume`.

## Change Surface (new, isolated)

- New package `orchestra/chat/`: `session.py`, `store.py`, `commands.py`, `loop.py`.
- `orchestra/cli.py` — `chat` command with `--resume` / `--list`.
- `resolve()` reused unchanged.

## Alternatives Considered

| Approach | Why rejected (for v1) |
|---|---|
| **Native messages API per provider** (Ollama `/api/chat`, Claude SDK `messages`) | Richer multi-turn, but provider-specific and two code paths; the concatenated prompt works identically everywhere including subprocess CLIs and the remote node. Deferred as a refinement. |
| **In-memory only, no persistence** | User explicitly wants save + `--resume`. |
| **Single fixed model, no slash commands** | User explicitly wants in-chat `/model` switching. |
| **Reuse the removed TUI** | It's gone and not returning; a plain rich-console loop is simpler and sufficient. |

## Open Questions

None — all resolved during brainstorming.

# CLAUDE.md

Orchestra: multi-agent AI orchestration engine (~29K lines Python 3.11+). Dispatches tasks to
LLM agents (Claude CLI, Kimi CLI, Codex CLI, Ollama cloud models, API providers; Gemini is
switched off) with routing, review/consensus loops, cost
tracking, and event-sourced persistence. Full map: `docs/architecture.md`. Known issues and
refactor roadmap: `docs/quality-audit.md`.

## Setup & Commands

The checked-in `.venv` may be bare. Create a working one first:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -m "not slow"          # fast suite (~2000 tests, ~30 s)
.venv/bin/pytest --cov=orchestra        # coverage, opt-in (gate: 60%, set in pyproject.toml)
.venv/bin/mypy orchestra/               # type check
orchestra --help                        # CLI (entry: orchestra.cli:app)
orchestra ask cdx-deep "task"           # single agent
orchestra ask cdx-deep#calibrated "task" # alias#mode: concise|calibrated + 6 thinking modes
orchestra run dual "task"               # parallel dual-agent + review
orchestra supervise "big task"          # hierarchical: decompose -> subagents -> synthesize
orchestra graph list                    # list blueprint workspaces built in the dashboard (~/.orchestra/*.json)
orchestra graph run "Deep Think"        # run a dashboard workspace/graph from the terminal (--task overrides Start)
orchestra models                        # list Ollama models per backend (local + remote)
orchestra ask qwen2.5:7b@windows-gpu "…" # query a specific model on a named backend
orchestra serve --token <secret>        # run THIS machine as an agent gateway for others
orchestra nodes-scan --token <secret>   # find LAN gateways and auto-add them
orchestra ask cdx-deep@windows "task"   # run a remote node's agent (its logged-in CLI account)
orchestra chat oll-fast                  # interactive REPL with a model (streams; local/remote)
orchestra chat --list                    # list saved sessions; --resume <id> to continue
orchestra claude                         # launch Claude Code against a LOCAL gateway (auto start/stop)
orchestra claude -m oll-coder -p "task"  # pick model; extra args pass through to `claude`
orchestra serve --token <secret>         # ALSO serves an OpenAI API at http://<host>:7070/v1
# point opencode/aider/Continue at base_url=http://<host>:7070/v1, api_key=<secret>; models:
# any alias (oll-fast, cdx-deep@windows) or orchestra-dual / orchestra-consensus / orchestra-auto
# the SAME server also speaks the Anthropic API at http://<host>:7070/v1/messages — for Claude Code:
#   ANTHROPIC_BASE_URL=http://<host>:7070 ANTHROPIC_API_KEY=<secret> ANTHROPIC_MODEL=oll-fast claude
#   (unknown model names fall back to oll-fast; tool use WORKS for Ollama models — Claude Code is
#   fully agentic via e.g. ANTHROPIC_MODEL=oll-glm; CLI aliases + orchestra-* modes stay text-only)
```

Tests are isolated from the working tree by `tests/conftest.py` (artifacts, databases and the
answer cache go to a temporary directory), so the suite can be run from the repo root. pytest is
configured in `pyproject.toml` only.

## Architecture (10 lines)

- Entry: `cli.py` (Typer) / `mcp_server.py` (MCP) / `server.py` (WebSocket) → `service.py:run_task()`
- Routing: `router/classifier.py` (heuristics, policy.toml) → `outcome_router.py` (learned EV) → `RouteDecision(mode, alias)`
- Modes: ask / dual (consensus = dual with stances) / planned / resume → `engine/runners/*_runner.py`; review → `engine/code_review.py`
- Shared machinery: `engine/runners/common.py` (cache, idempotency, PII scrub, approval gates, synthesis)
- Concurrency: `engine/parallel.py` (asyncio + thread fallback, budget guard, cancellation)
- Providers: `providers/fallback.py:resolve(alias)` → `BaseProvider.run/async_run/stream`
- Persistence: 4 SQLite WAL DBs under `.orchestra/` — orchestra.db (runs), events.db (append-only log), memory.db (+HNSW index); the answer cache `fts_cache.db` lives in `~/.orchestra/` and is shared by all projects
- Events: `events/bus.py` (in-process pub-sub) + durable `storage/event_log.py`; runs are replayable via seq-keyed idempotency
- Federation (`engine/federation*.py`, CRDT-based, **disabled by default** in `.orchestra/config.toml`)
- Toolmaker: agent-authored tools, AST-validated (`toolmaker/validator.py`) + sandboxed (`sandbox.py`)

## Active vs Dead Code

- TUI: **fully removed** (2026-07-09, `orchestra/tui/` + FleetDashboard + `launch`/`dashboard`
  commands + textual dep). Web dashboard (`monitoring/web_dashboard.py`, aiohttp) remains.
- `engine/command_queue.py` is dead (0 importers); graph/patterns/roles/group_chat/speculative/
  mcp_fastlane are experimental, several behind disabled config flags.
- `ForestBuildingSystem/` is an unrelated Unity C# demo — not part of the engine.

## Conventions

- Conventional Commits (`feat:`/`fix:`/`refactor:`/…); no bare `except Exception: pass`
  (CONTRIBUTING.md) — the codebase has legacy violations, don't add more.
- Config: TOML at `.orchestra/config.toml` (+ `ORCHESTRA_*` env overrides), read through
  `orchestra/config.py` accessor functions — never parse the file directly.
- Multi-backend Ollama: define named backends and address models as `model@backend`.
  Remote machine must expose Ollama to the LAN first (`OLLAMA_HOST=0.0.0.0`). Example:
  ```toml
  [providers.ollama.backends.macbook]
  base_url = "http://localhost:11434"
  [providers.ollama.backends.windows-gpu]
  base_url = "http://192.168.1.42:11434"
  ```
- Remote nodes: run `orchestra serve --token X` on the machine whose agents you want to share
  (its logged-in CLI accounts + Ollama); address them as `alias@node`. `alias@node` resolution
  checks `[remote_nodes.*]` first, then `[providers.ollama.backends.*]`. Config:
  ```toml
  [remote_nodes.windows]
  url = "http://192.168.1.42:7070"
  token = "X"
  ```
  Security: the token grants remote code execution on the serving machine (agents run code) —
  trusted networks only; `--host 0.0.0.0` is refused without a token. See `gateway.py`,
  `providers/remote_node.py`, `remote_nodes.py`.
- Provider contract: subclass `providers/base.py`, report usage through `last_usage()`
  (the provider object is shared by concurrent agents: keep per-call state in a ContextVar as
  `providers/ollama.py` does, not on `self`; Kimi and the Claude CLI report no usage, so their
  token counts are estimates), register alias in `fallback.py` ALIAS_TABLE **and** `registry.py`.
- Which model does which kind of work is decided by measurement: `docs/model-routing.md`,
  `tools/model_bench/bench.py`. Kimi is reached through the Kimi CLI only, never an Ollama tag.
- Claude agents: `cld-fast`/`cld-deep` read `[providers.claude]`, `cld-write` reads
  `[providers.claude_write]` (tools, permission mode, extra args). Without a profile the agent
  runs with `--dangerously-skip-permissions`; see `config.toml.example` for a read-only one.
- Stored answers: a single-agent run is answered from the replay store (48 h, finished runs
  only) or the answer cache (same prompt + same alias, 7 days). `fresh=True` / `--fresh` asks
  the model again. Never match cached answers by similarity: `FtsSearchCache.search` ranks by
  shared words and returned other questions' answers.
- Internal steps (verifier, synthesiser, critic, eval judge, note summary) call
  `run_ask(..., internal=True, parent_run_id=...)`: no prompt prefix, no cache, no approval
  gate, not indexed to memory, and cancelled with their parent. A planned run's nodes pass
  `parent_run_id` only (they are real tasks, but still steps). Steps are hidden from
  `orchestra ps` and `list_runs` unless `--all` / `include_steps`.
- Memory holds two kinds of rows: notes someone added, and the task text of every finished
  run (`metadata.run_id` set; the router uses them). Only notes may reach an agent's prompt:
  use `storage.memory.search_notes`, never `memory.search`, when building context.
- An alias whose provider cannot run (not installed, disabled, logged out) is answered by the
  next alias in its fallback chain; the run logs an `agent_fallback` event. `orchestra doctor`
  lists such providers and any role that points at one.
- A run may continue another (`follows_run_id`, set from `follow_up_to` / `--follow`): the
  discussion is the chain of runs read back from their manifests (`engine/follow_up.py`), and a
  follow-up bypasses every prompt-keyed store. Consensus mode is `run_dual(stances=True)`
  (`engine/stances.py`). Code review is `engine/code_review.py`: a finding is kept only if its
  quote is in the diff; never return unchecked findings.
- MCP run tools return a compact payload (`engine/mcp_registry._compact_run_payload`);
  `orchestra_get_run` is the one that returns the full manifest and events.
- Events: emit via `events/bus.py` with a valid event type; artifacts/manifests via
  `engine/artifacts.py` only.
- Prompts live in `orchestra/prompts/**/*.txt` (packaged via pyproject package-data).

## Gotchas

- `config.py` locates `.orchestra/config.toml` by walking up from **CWD at import time** —
  running from another directory silently loads different (or no) config.
- `engine/parallel.py` mutates shared `OrchestraRun` under a `threading.Lock`; never `await`
  while holding it (a known defect of this kind lives at parallel.py:576/660 — fix planned).
- cli.py defines several command groups twice (lines ~1788 and ~1874) — edit the later copy or,
  better, execute `docs/plans/refactor-cli-split.md` Phase 0 first.
- 4 separate SQLite files open a fresh connection per operation; heavy parallel writes can hit
  `database is locked` — keep `busy_timeout` in mind.

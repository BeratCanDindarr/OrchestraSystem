# CLAUDE.md

Orchestra: multi-agent AI orchestration engine (~29K lines Python 3.11+). Dispatches tasks to
LLM agents (Claude/Codex/Gemini/OpenAI/Ollama) with routing, review/consensus loops, cost
tracking, and event-sourced persistence. Full map: `docs/architecture.md`. Known issues and
refactor roadmap: `docs/quality-audit.md`.

## Setup & Commands

The checked-in `.venv` may be bare. Create a working one first:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -m "not slow"          # fast suite (~91 files, ~1326 tests)
.venv/bin/pytest --cov=orchestra        # coverage (gate: 60%)
.venv/bin/mypy orchestra/               # type check
orchestra --help                        # CLI (entry: orchestra.cli:app)
orchestra ask cdx-deep "task"           # single agent
orchestra run dual "task"               # parallel dual-agent + review
orchestra models                        # list Ollama models per backend (local + remote)
orchestra ask qwen2.5:7b@windows-gpu "…" # query a specific model on a named backend
orchestra serve --token <secret>        # run THIS machine as an agent gateway for others
orchestra nodes-scan --token <secret>   # find LAN gateways and auto-add them
orchestra ask cdx-deep@windows "task"   # run a remote node's agent (its logged-in CLI account)
```

⚠️ Tests currently write artifacts to CWD (no path-isolation fixture — see
`docs/plans/refactor-repo-hygiene.md`). Run pytest from a scratch directory or expect junk
files (`:memory:*`, `<MagicMock…>`) in the repo root.

## Architecture (10 lines)

- Entry: `cli.py` (Typer) / `mcp_server.py` (MCP) / `server.py` (WebSocket) → `service.py:run_task()`
- Routing: `router/classifier.py` (heuristics, policy.toml) → `outcome_router.py` (learned EV) → `RouteDecision(mode, alias)`
- Modes: ask / dual / critical / planned / resume → `engine/runners/*_runner.py`
- Shared machinery: `engine/runners/common.py` (cache, idempotency, PII scrub, approval gates, synthesis)
- Concurrency: `engine/parallel.py` (asyncio + thread fallback, budget guard, cancellation)
- Providers: `providers/fallback.py:resolve(alias)` → `BaseProvider.run/async_run/stream`
- Persistence: 4 SQLite WAL DBs under `.orchestra/` — orchestra.db (runs), events.db (append-only log), memory.db (+HNSW index), fts_cache.db (BM25 cache)
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
- Provider contract: subclass `providers/base.py`, store usage in `self._last_usage`
  (three providers get this wrong today — see quality-audit H-4), register alias in
  `fallback.py` ALIAS_TABLE **and** `registry.py`.
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

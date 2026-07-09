# CLAUDE.md

Orchestra: multi-agent AI orchestration engine (~29K lines Python 3.11+). Dispatches tasks to
LLM agents (Claude/Codex/Gemini/OpenAI/Ollama) with routing, review/consensus loops, cost
tracking, and event-sourced persistence. Full map: `docs/architecture.md`. Known issues and
refactor roadmap: `docs/quality-audit.md`.

## Setup & Commands

The checked-in `.venv` may be bare. Create a working one first:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -m "not slow"          # fast suite (97 files, 1302 tests)
.venv/bin/pytest --cov=orchestra        # coverage (gate: 60%)
.venv/bin/mypy orchestra/               # type check
orchestra --help                        # CLI (entry: orchestra.cli:app)
orchestra ask cdx-deep "task"           # single agent
orchestra run dual "task"               # parallel dual-agent + review
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

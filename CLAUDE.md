# CLAUDE.md

Orchestra: multi-agent AI orchestration engine (~25K lines Python 3.11+). Dispatches tasks to
agents that can read the project: Claude CLI, Kimi CLI, Codex CLI, Gemini through the
Antigravity CLI (`agy-*`, `gmn-*`), and Ollama models run through the Claude CLI. Used from
Claude Code only: the MCP server plus the statusline. Routing, review/consensus loops, cost
tracking, and event-sourced persistence. Full map: `docs/architecture.md`. Known issues and
refactor roadmap: `docs/quality-audit.md`.

## Setup & Commands

The checked-in `.venv` may be bare. Create a working one first:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -m "not slow"          # fast suite (~1700 tests, ~20 s)
.venv/bin/pytest --cov=orchestra        # coverage, opt-in (gate: 60%, set in pyproject.toml)
.venv/bin/mypy orchestra/               # type check (113 known errors; reported, not gated)
.venv/bin/ruff check orchestra tests    # correctness rules only: syntax (py311), undefined names
orchestra --help                        # CLI (entry: orchestra.cli:app)
orchestra ask cdx-deep "task"           # single agent
orchestra ask cdx-deep#calibrated "task" # alias#mode: concise|calibrated + 6 thinking modes
orchestra run dual "task"               # parallel dual-agent + review
orchestra supervise "big task"          # hierarchical: decompose -> subagents -> synthesize
orchestra models                        # list Ollama models per backend (local + remote)
orchestra ask glm-5.3:cloud@windows-gpu "…" # a specific model on a named Ollama backend
orchestra statusline install            # per-agent progress in Claude Code's statusline
orchestra review --staged --fail-on high # the commit gate: exit 1 on a checked high-or-worse finding
orchestra hook install                   # run that gate as the git pre-commit hook (uninstall removes it)
orchestra review --base main --comment-pr 42 # post the checked findings on a PR (gh CLI)
```

Tests are isolated from the working tree by `tests/conftest.py` (artifacts, databases, the
memory store and the answer cache go to a temporary directory), so the suite can be run from the
repo root. No test reaches an agent CLI or an embedding endpoint: both are refused there, and a
test that needs the memory store to accept rows takes the `fake_embeddings` fixture. pytest,
coverage, mypy and ruff are configured in `pyproject.toml` only; CI is `.github/workflows/ci.yml`
(ruff, the fast suite with the coverage gate on 3.11 and 3.12, mypy reported).

## Architecture (10 lines)

- Entry: `mcp_fastmcp.py` (the MCP server Claude Code starts; tools registered in `mcp_server.py`) / `cli.py` (Typer) → `service.py:execute_task()` (the one place a task is routed and run; `run_task` wraps it for MCP). The CLI never imports `engine.runners` (a test enforces it)
- Routing: `router/classifier.py` (heuristics, policy.toml) → `outcome_router.py` (learned EV) → `RouteDecision(mode, alias)`
- Modes: ask / dual (consensus = dual with stances) / planned / resume → `engine/runners/*_runner.py`; review → `engine/code_review.py`
- Shared machinery: `engine/runners/common.py` (cache, idempotency, PII scrub, approval gates, synthesis)
- Concurrency: `engine/parallel/` (one async agent runner, budget guard, cancellation)
- Providers: `providers/fallback.py:resolve(alias)` → `BaseProvider.run/async_run/stream`
- Persistence: SQLite WAL DBs under `.orchestra/` — orchestra.db (runs, agents and the append-only event log), memory.db (+HNSW index, `config.memory_db_path()`), runs/event_log.db (left from the removed replay store); the answer cache `fts_cache.db` lives in `~/.orchestra/` and is shared by all projects. `[eventlog] db_path` is not where events go
- Events: `events/bus.py` (in-process pub-sub) + durable `storage/event_log.py`; runs are replayable via seq-keyed idempotency
- Federation (`engine/federation*.py`, CRDT-based, **disabled by default** in `.orchestra/config.toml`)

## Active vs Dead Code

- Removed 2026-10-20 (`docs/plans/refactor-payload-slimming.md`): the web dashboard, the
  WebSocket daemon, `chat`, the gateway (`serve`, OpenAI/Anthropic-compatible APIs,
  `orchestra claude`, remote nodes), `federation_server/`, batch runner, MCP client, cue,
  playbook, session discovery, toolmaker, the text-only API providers and the Gemini CLI
  provider, `ForestBuildingSystem/`. Earlier: the TUI (2026-07), the toolmaker sandbox,
  `engine/skills/`, `patterns`, `integration_hooks`, `mcp_fastlane` (2026-10). speculative/cascade are
  behind disabled config flags. Outcome routing (`router/outcome_router.py`: the mode that
  succeeded on similar earlier tasks) is wired into `auto` and off unless
  `[outcome_routing] enabled = true`.

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
- Every alias is an agent that can read the project; there are no text-only API providers.
  Ollama models are run by `providers/ollama_agent.py`: the Claude CLI with
  `ANTHROPIC_BASE_URL` set to the Ollama server (Ollama 0.40+ serves `/v1/messages`), the
  `[providers.claude]` tools, and `--bare` (no CLAUDE.md / rules auto-loaded on every turn:
  ~9K tokens a turn). The CLI's Claude-priced cost is dropped; `config.price_of` prices the
  Ollama model. `providers/ollama.py` is the HTTP client left for embeddings, health checks,
  model lists and the router's short classification call.
- Unity: with `[unity] enabled = true` a reading Claude/Ollama agent gets `--mcp-config` for
  `orchestra/unity_read.py`, a proxy of the project's UnityMCP that offers only reading actions
  (`READ_ACTIONS`) and refuses the rest. Never give an agent UnityMCP itself: its tools read and
  write through one `action`, and permissions are per tool. A writing profile gets no Unity.
- What agents get: the task, a role line, a pointer to CLAUDE.md / AGENTS.md / docs (they read
  what they need), the confidence rule, the Unity annotation rule on Unity tasks only, and
  memory notes. Files the task mentions are not pasted in (`[compression]
  code_context_enabled`, default off). The synthesiser gets the candidates alone, without the
  verifier's output.
- Provider contract: subclass `providers/base.py`, report usage through `last_usage()`
  (the provider object is shared by concurrent agents: keep per-call state in a ContextVar as
  `providers/ollama.py` does, not on `self`; Kimi and the Claude CLI report no usage, so their
  token counts are estimates), register alias in `fallback.py` ALIAS_TABLE **and** `registry.py`.
- Which model does which kind of work is decided by measurement: `docs/model-routing.md`,
  `tools/model_bench/bench.py`. Kimi is reached through the Kimi CLI only, never an Ollama tag.
- Claude agents: `cld-fast`/`cld-deep` read `[providers.claude]`, `cld-write` reads
  `[providers.claude_write]` (tools, permission mode, extra args). Without a profile `cld-fast` /
  `cld-deep` can only read (`--tools Read,Grep,Glob`): the config is found from the CWD, so a
  missing profile must not make a reviewer able to write. Only `cld-write` defaults to
  `--dangerously-skip-permissions`; Antigravity (`agy-*`) bypasses permissions only with
  `[providers.antigravity] skip_permissions = true` (headless agy can do nothing, not even read,
  without it). `[providers.<name>] enabled = false` is enforced for every provider in
  `fallback._usable`.
- Stored answers: a single-agent run is answered from the answer cache only (7 days; the
  48 h replay store that did the same was removed 2026-10-21); it needs the same prompt + same alias + same project state (the state
  is the project root, git HEAD and a hash of `git diff HEAD`, `common._cache_scope`). A hit
  keeps the confidence its agent stated. `fresh=True` / `--fresh` asks the model again. Never
  match cached answers by similarity: the cache has exact lookup only (its BM25 `search` was
  removed: it returned other questions' answers and ranked them backwards).
- Internal steps (verifier, synthesiser, critic, eval judge, note summary) call
  `run_ask(..., internal=True, parent_run_id=...)`: no prompt prefix, no cache, no approval
  gate, not indexed to memory, and cancelled with their parent. A planned run's nodes pass
  `parent_run_id` only (they are real tasks, but still steps). Steps are hidden from
  `orchestra ps` and `list_runs` unless `--all` / `include_steps`.
- Memory holds the notes someone added. Finished runs are no longer stored there (2026-10-21:
  an embedding per run, read only by a router that weighed self-stated confidence); rows an
  earlier version stored have `metadata.run_id` set. Only notes may reach an agent's prompt:
  use `storage.memory.search_notes`, never `memory.search`, when building context.
- An alias whose provider cannot run (not installed, disabled, logged out) is answered by the
  next alias in its fallback chain (`fallback.safe_chain`: a read-only alias skips every agent
  whose `provider.can_write()` is true); the run logs an `agent_fallback` event. `orchestra doctor`
  lists such providers and any role that points at one. Codex runs with `-s read-only`.
- A run may continue another (`follows_run_id`, set from `follow_up_to` / `--follow`): the
  discussion is the chain of runs read back from their manifests (`engine/follow_up.py`), and a
  follow-up bypasses every prompt-keyed store. Consensus mode is `run_dual(stances=True)`
  (`engine/stances.py`). Code review is `engine/code_review.py`: a finding is kept only if its
  quote is in the diff; never return unchecked findings.
- Agents are asked to end with `Confidence: high|medium|low` (`_CONFIDENCE_RULE`);
  `normalize.stated_confidence` reads it. Do not weigh answers by `AgentRun.confidence` without
  checking that the agent stated one (0.5 is the "none stated" default).
- Retries: any failed attempt moves on to the alias's fallback chain. A successful answer is
  never retried for containing an error word unless it is short (`retries._should_retry_output`).
- `orchestra undo` restores a snapshot that no run mode currently takes; it must stay unable to
  discard anything when there is no snapshot (`backtrack.find_snapshot` first).
- The caller of the MCP tools is an agent: `orchestra_continue_run` refuses unless
  `[mcp] allow_approve` is set (approval goes through `orchestra_request_approval`, which asks
  the owner). There is no config tool over MCP. A new MCP tool that changes state needs the
  same thought; a new tool at all needs a reason, since every schema lands in Claude's context.
- Every byte an MCP tool returns lands in Claude's context: run tools return a compact payload
  (`engine/mcp_registry._compact_run_payload`, compact JSON, no repeated blocks);
  `orchestra_get_run(detail=true)` is the one that returns the full manifest and events,
  `orchestra_get_logs` returns the tail of failed agents unless `full=true`, `orchestra_list_runs`
  cuts each task to 80 characters.
- Events: emit via `events/bus.py` with a valid event type; artifacts/manifests via
  `engine/artifacts.py` only.
- A run's verdict belongs to its runner: set `run.status` / `run.failure` *before* `_finalize`.
  `_final_status` keeps FAILED and CANCELLED and only otherwise derives the status from the
  agents; a run with no agents fails. A denied run does not get its agents' output as summary.
- One budget setting: `config.max_budget_usd()` (`[limits] max_budget_usd`, then the legacy
  `[execution]` key, else no limit). Do not read either section directly.
- A name that enters a run goes through `fallback.require_known` (run_ask and run_dual's
  `agents` do it): `_canonical_alias` maps anything unknown to a same-family agent or cdx-deep.
- Code review: `read_answer` returns None for an answer that is not a review and [] for a
  reviewer that found nothing. "No defects found" needs a non-empty `reviewed_by`.
- Writing agents: with `[isolation] enabled = true` (default off) an ask on a writing alias
  (`cld-write`) runs in a git worktree at HEAD (`engine/isolation.py`), its changes become
  `changes.patch` in the run folder, and `orchestra apply|discard <run_id>` settles them (`apply --pick`: hunk by hunk, the rest kept as `rejected.patch`). Such a
  run is never answered from a store. `cwd` reaches an agent's process only through
  `run_parallel(..., cwd=...)`.
- Cost: `engine/costing.py` is the only place that turns usage into money (`charge(agent, …)`).
  Prices come from `config.price_of` (a number or an input/output/cached table per model); an
  unlisted local Ollama model costs nothing. `agent.cost_source` says "reported" or "estimated".
- `alias#mode` is handled in `run_ask` (`ask_runner._apply_mode`), not by each caller.
- Reading the history (`list_runs`) must not call `backfill()`;
  the index is rebuilt on a read only when it is empty.
- Prompts live in `orchestra/prompts/**/*.txt` (packaged via pyproject package-data).

## Gotchas

- `config.py` locates `.orchestra/config.toml` by walking up from **CWD at import time** —
  running from another directory silently loads different (or no) config.
- `engine/parallel/` mutates shared `OrchestraRun` under a `threading.Lock`; never `await`
  while holding it (an AST test enforces this).
- There is one agent runner, `engine/parallel._run_one_agent_async`; `run_parallel` calls it
  with `asyncio.run`, in a worker thread when the caller is inside a loop (the sync twin and
  `retries.run_with_retry` were removed). The approval-gate blocks of the three runners are
  still near-copies: a fix in one is usually needed in the others.
- `tests/test_migrations.py::test_concurrent_migrations_serialized` fails now and then under
  the full suite and passes alone; it is timing, not a regression.
- `docs/quality-audit.md` §7 lists what the 2026-10 review found and what is still open.
- Open a store with `storage/sqlite.connect` / `transaction` (WAL, 10 s busy wait, closed after
  use), never `sqlite3.connect` directly; the answer cache (`fts_cache`) alone keeps one long-lived
  connection. Write a manifest through `artifacts.write_manifest_data` (atomic replace): a reader
  must never see a half-written run.

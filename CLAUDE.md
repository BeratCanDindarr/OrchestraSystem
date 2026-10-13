# CLAUDE.md

Orchestra: multi-agent AI orchestration engine (~36K lines Python 3.11+). Dispatches tasks to
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
.venv/bin/mypy orchestra/               # type check (113 known errors; reported, not gated)
.venv/bin/ruff check orchestra tests    # correctness rules only: syntax (py311), undefined names
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

Tests are isolated from the working tree by `tests/conftest.py` (artifacts, databases, the
memory store and the answer cache go to a temporary directory), so the suite can be run from the
repo root. No test reaches an agent CLI or an embedding endpoint: both are refused there, and a
test that needs the memory store to accept rows takes the `fake_embeddings` fixture. pytest,
coverage, mypy and ruff are configured in `pyproject.toml` only; CI is `.github/workflows/ci.yml`
(ruff, the fast suite with the coverage gate on 3.11 and 3.12, mypy reported).

## Architecture (10 lines)

- Entry: `cli.py` (Typer) / `mcp_server.py` (MCP) / `server.py` (WebSocket) → `service.py:run_task()`
- Routing: `router/classifier.py` (heuristics, policy.toml) → `outcome_router.py` (learned EV) → `RouteDecision(mode, alias)`
- Modes: ask / dual (consensus = dual with stances) / planned / resume → `engine/runners/*_runner.py`; review → `engine/code_review.py`
- Shared machinery: `engine/runners/common.py` (cache, idempotency, PII scrub, approval gates, synthesis)
- Concurrency: `engine/parallel.py` (asyncio + thread fallback, budget guard, cancellation)
- Providers: `providers/fallback.py:resolve(alias)` → `BaseProvider.run/async_run/stream`
- Persistence: SQLite WAL DBs under `.orchestra/` — orchestra.db (runs, agents and the append-only event log), memory.db (+HNSW index, `config.memory_db_path()`), runs/event_log.db (the replay store of single-agent answers); the answer cache `fts_cache.db` lives in `~/.orchestra/` and is shared by all projects. `[eventlog] db_path` is not where events go
- Events: `events/bus.py` (in-process pub-sub) + durable `storage/event_log.py`; runs are replayable via seq-keyed idempotency
- Federation (`engine/federation*.py`, CRDT-based, **disabled by default** in `.orchestra/config.toml`)
- Tool proposals: `storage/toolmaker.py` (review / promote workflow over MCP); there is no sandbox that runs agent-written code

## Active vs Dead Code

- TUI: **fully removed** (2026-07-09, `orchestra/tui/` + FleetDashboard + `launch`/`dashboard`
  commands + textual dep). Web dashboard (`monitoring/web_dashboard.py`, aiohttp) remains.
- Not reachable from any entry point: `engine/batch_runner.py` (only the eval tests use it),
  `engine/mcp_client.py`, `federation_server/` (no command starts it). The toolmaker sandbox,
  `engine/skills/`, `patterns`, `integration_hooks` and `mcp_fastlane` were removed in 2026-10
  (`docs/plans/roadmap-2026-10-engine.md`). speculative/cascade are
  behind disabled config flags. Outcome routing (`router/outcome_router.py`: the mode that
  succeeded on similar earlier tasks) is wired into `auto` and off unless
  `[outcome_routing] enabled = true`.
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
  `[providers.claude_write]` (tools, permission mode, extra args). Without a profile `cld-fast` /
  `cld-deep` can only read (`--tools Read,Grep,Glob`): the config is found from the CWD, so a
  missing profile must not make a reviewer able to write. Only `cld-write` defaults to
  `--dangerously-skip-permissions`. `[providers.<name>] enabled = false` is enforced for every
  provider in `fallback._usable`; text-only API aliases (`openai-*`, `ds-*`, `pplx-*`, `az-*`)
  have no fallback, so a missing key never becomes a CLI agent.
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
- Agents are asked to end with `Confidence: high|medium|low` (`_CONFIDENCE_RULE`);
  `normalize.stated_confidence` reads it. Do not weigh answers by `AgentRun.confidence` without
  checking that the agent stated one (0.5 is the "none stated" default).
- Retries: any failed attempt moves on to the alias's fallback chain. A successful answer is
  never retried for containing an error word unless it is short (`retries._should_retry_output`).
- `orchestra undo` restores a snapshot that no run mode currently takes; it must stay unable to
  discard anything when there is no snapshot (`backtrack.find_snapshot` first).
- MCP run tools return a compact payload (`engine/mcp_registry._compact_run_payload`);
  `orchestra_get_run` is the one that returns the full manifest and events.
- Events: emit via `events/bus.py` with a valid event type; artifacts/manifests via
  `engine/artifacts.py` only.
- A run's verdict belongs to its runner: set `run.status` / `run.failure` *before* `_finalize`.
  `_final_status` keeps FAILED and CANCELLED and only otherwise derives the status from the
  agents; a run with no agents fails. A denied run does not get its agents' output as summary.
- One budget setting: `config.max_budget_usd()` (`[limits] max_budget_usd`, then the legacy
  `[execution]` key, else no limit). Do not read either section directly.
- A name that enters a run goes through `fallback.require_known` (run_ask and run_dual's
  `agents` do it): `_canonical_alias` maps anything unknown to a same-family agent or cdx-deep.
- Servers that run agents (gateway, dashboard, daemon, federation) use `utils/browser_guard`:
  no cross-origin or cross-site request, loopback Host only when bound to loopback, token
  required on a public host. A new HTTP app must take the middleware; a state-changing or
  model-running endpoint should not be a GET. `nodes-scan` saves a host only after the
  gateway's `/identify` proof (`remote_nodes.verify_gateway`); the token is never sent to find out.
- Code review: `read_answer` returns None for an answer that is not a review and [] for a
  reviewer that found nothing. "No defects found" needs a non-empty `reviewed_by`.
- Cost: `engine/costing.py` is the only place that turns usage into money (`charge(agent, …)`).
  Prices come from `config.price_of` (a number or an input/output/cached table per model); an
  unlisted local Ollama model costs nothing. `agent.cost_source` says "reported" or "estimated".
- Gateway: a plain reply goes through `run_ask(internal=True)`; streamed text and native
  tool-call turns are served straight from the provider and recorded by `gateway_dispatch._record`
  as steps under the parent `gateway` (hidden from `ps`, counted in spend). `alias#mode` is
  handled in `run_ask` (`ask_runner._apply_mode`), not by each caller.
- Reading the history (`list_runs`, `get_stats`, `get_reputation`) must not call `backfill()`;
  the index is rebuilt on a read only when it is empty.
- Prompts live in `orchestra/prompts/**/*.txt` (packaged via pyproject package-data).

## Gotchas

- `config.py` locates `.orchestra/config.toml` by walking up from **CWD at import time** —
  running from another directory silently loads different (or no) config.
- `engine/parallel/` mutates shared `OrchestraRun` under a `threading.Lock`; never `await`
  while holding it (an AST test enforces this).
- `_run_one_agent` and `_run_one_agent_async` in `engine/parallel/__init__.py` are near-copies
  (229 / 244 lines), as are the approval-gate blocks of the three runners: a fix in one is
  usually needed in the others.
- `tests/test_migrations.py::test_concurrent_migrations_serialized` fails now and then under
  the full suite and passes alone; it is timing, not a regression.
- `docs/quality-audit.md` §7 lists what the 2026-10 review found and what is still open.
- 4 separate SQLite files open a fresh connection per operation; heavy parallel writes can hit
  `database is locked` — keep `busy_timeout` in mind.

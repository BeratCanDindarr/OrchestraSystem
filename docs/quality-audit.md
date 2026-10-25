# Orchestra — Code Quality Audit

**Date:** 2026-07-06 · **Method:** static metrics (wc/AST scan) + 4 subsystem audits + manual
spot-checks of every CRITICAL claim. No production code was modified during this audit.

## 1. Methodology & Standards

Audited against the project owner's global standards:
files ≤800 lines · functions ≤50 lines · nesting ≤4 · immutability across boundaries ·
no silent exception swallowing · schema validation at system boundaries · 80% test coverage.

Baseline metrics (repo @ `691a46d`, verified):

| Metric | Value |
|---|---|
| Python LOC in `orchestra/` | 29,008 (159 files) |
| Files >800 lines | **5** — cli.py (2066), service.py (1280), engine/runners/common.py (1188), config.py (881), engine/parallel.py (874) |
| Functions >50 lines | **97** (worst: parallel.py:449 `_run_one_agent_async`, 219 lines) |
| Functions with nesting >4 | **27** (worst: tui/widgets/v3/markdown_view.py:270 `_parse_markdown`, depth 19) |
| `except Exception` occurrences | 188 (111 in engine/) |
| Bare `except: pass/continue` | 8 — cli.py:1805,1894 · tui/widgets/v4/chat.py:95,126,132,224 · tui/widgets/v2/navigator.py:230 · engine/artifacts.py:189 |
| Test suite | 97 files, 1,302 test functions; coverage gate `fail_under = 60` (target: 80) |

## 2. Subsystem Scorecard

| Subsystem | File size | Fn size | Nesting | Error handling | Boundary validation | Verdict |
|---|---|---|---|---|---|---|
| engine/ | ✗ (2 files) | ✗ | ⚠ | ✗ (111 broad catches) | ⚠ | Needs refactor |
| router/ + providers/ | ✓ | ⚠ | ⚠ | ⚠ | ✗ (alias KeyError) | Needs consistency pass |
| storage/ | ✓ | ⚠ | ⚠ | ⚠ | ✓ | Acceptable |
| federation | ✓ | ✓ | ✓ | ⚠ | ✗ (no schema/auth default) | CRDTs sound; server not prod-ready |
| server/service/mcp | ✗ (service.py) | ✗ | ✗ | ✗ | ✗ | Worst boundary surface |
| tui/ | ✓ | ⚠ | ✗ (depth 19) | ✗ (4× bare pass) | n/a | Stabilizing; dead code |
| toolmaker/ | ✓ | ⚠ | ✓ | ✓ | ⚠ | Sandbox needs hardening |
| tests/ | — | — | — | — | — | No path isolation fixture |

## 3. Findings by Severity

Focus areas (engine/router, federation/infra) are ranked one level up per review scope.

### CRITICAL

| ID | Location | Finding | Evidence (spot-checked) | Fix |
|---|---|---|---|---|
| C-1 | engine/parallel.py:576→660 | `threading.Lock` held across `await` | `with lock:` block at 576 contains `await _enforce_budget_guard_async(...)` at 660 — can stall the event loop and every other agent thread; likely contributor to past TUI freezes | Restructure so the guard runs outside the lock, or switch the async path to `asyncio.Lock` with narrow critical sections |
| C-2 | cli.py:1788–1857 vs 1874–1938 | Entire command groups defined **twice** | `playbook_app`/`cue_app`/`sessions_app`/`debate_app` + their commands duplicated; `app.add_typer(config_app)` at both 589 and 1858. Second definitions silently override the first | Delete one copy; add a smoke test asserting unique command names |
| C-3 | engine/runners/common.py:117–121 | Broad `except Exception` falls back to `FtsSearchCache(db_path=":memory:")`, which creates a **literal file named `:memory:`** in CWD | The `:memory:*` files committed at repo root are this bug's output; failure is reduced to a `logger.warning` | Catch specific config errors; use a real in-memory SQLite handle or disable cache on failure |
| C-4 | repo root / git index | No `.gitignore`; runtime state and junk committed | 5,950 of 6,285 tracked files are `.orchestra/` (incl. live `*.db`, `-wal`, `-shm`); plus `:memory:*`, `<MagicMock…>`, 13 `launch_*.log`, `.coverage`, `=3.9.0`, `orchestra.egg-info/`. Secret scan of `.orchestra/runs/` came back **clean** → plain `git rm --cached` suffices, no history rewrite | `docs/plans/refactor-repo-hygiene.md` |
| C-5 | toolmaker/validator.py:21–56 | Sandbox validator bypassable | `os`/`sys` absent from `_BLOCKED_IMPORTS` (verified) → `import os; os.system("curl …")` passes AST validation and escapes the env-var-only network block; `open(f, mode_var)` bypasses write check (only `ast.Constant` inspected, line 104); `getattr` reaches blocked dunders | **Calibrated fix (2026-07-06):** do NOT block `import os` — an existing test (`test_allows_stdlib_imports`) and legitimate tools rely on `os.path`/`os.getenv`. Instead block exec-style CALLS by name (`system`/`popen`/`exec*`/`spawn*`/`fork`; alias-robust via `_call_name`), reject non-constant `open()` modes, and block `getattr`/`setattr`/`delattr` with non-constant/dunder attribute names. Kernel-level enforcement (seccomp/sandbox-exec) is Phase 2 in refactor-toolmaker-hardening.md |

### HIGH

| ID | Location | Finding | Fix |
|---|---|---|---|
| H-1 | 5 files listed in §1 | >800-line rule violations; cli.py is 2.6× the limit | Refactor plans: `refactor-cli-split.md`, `refactor-runners-common.md` |
| H-2 | server.py:589 · mcp_server.py:90 · federation_server/federation_http.py:43 | External payloads (`WebSocket`, MCP stdin, federation `/push`) parsed with `json.loads`/`request.json()` and **no schema validation**; malformed input is swallowed by broad handlers | Introduce validated request dataclasses (or pydantic); reject with explicit 400/error frames |
| H-3 | federation_server/federation_http.py:24–28 | Auth is a static bearer token and **disabled when unset** (`_check_auth` returns True with no token) | Require token when `federation.enabled`; document; add rate limiting on `/push` |
| H-4 | providers/perplexity.py:109, deepseek.py:100, openai_sdk.py:101 | Providers set `self.last_usage` instead of the base-class `self._last_usage` → usage/cost tracking silently lost for 3 providers | Rename to `_last_usage`; add a contract test over all providers |
| ~~H-5~~ → LOW | providers/fallback.py:116,130 | **Re-verified 2026-07-06: false positive.** `resolve()` calls `_canonical_alias(alias)` first, which always maps to a valid `ALIAS_TABLE` key (`oll-dev`/`oll-coder` are table keys; unknown → `cdx-deep`), so no `KeyError` is reachable. The TUI "NONE" alias bug was fixed in an earlier commit. Remaining (LOW): line 130 returns an unavailable provider by design with a console warning — acceptable. No code change. | (none — keep `_canonical_alias` as the guard) |
| H-6 | models.py + engine/parallel.py (20+ sites) | `OrchestraRun`/`AgentRun` are mutable dataclasses mutated in place from multiple threads (immutability standard) | Short term: confine mutation to lock-guarded helpers; long term: frozen dataclasses + explicit state transitions |
| H-7 | cli.py:848 `eval_cmd` (210 lines) et al. | CLI bypasses service.py and calls runners directly; utility fns duplicated between cli.py and service.py (`_parse_cli_value`, `_toml_literal`) | Route all business logic through `service.run_task()`; extract shared utils |
| H-8 | 8 bare `except: pass/continue` (§1) | Fully silent swallowing in cli.py, chat.py (widget lifecycle), navigator.py. **Update 2026-07-09:** TUI removed — only the cli.py/artifacts.py sites remain relevant | Catch specific exceptions, log at debug |
| H-9 | tests/conftest.py (5 lines) | No `tmp_path`/`artifact_root` isolation fixture → tests write to CWD; MagicMock-named files came from patching `config.artifact_root` with a bare `MagicMock` | Session-scoped autouse fixture pointing `artifact_root`/db paths at `tmp_path`; see refactor-repo-hygiene.md |

### MEDIUM

| ID | Location | Finding |
|---|---|---|
| M-1 | registry.py (19 aliases) vs fallback.py (23) vs classifier/cascade hardcoded maps | Alias→provider/model mapping fragmented across 4+ modules; adding a provider requires ~5 edits |
| M-2 | outcome_router.py:22–24, cascade_router.py:31–32,79,100, semantic.py:41,94, classifier.py:106–177 | 15+ magic thresholds/confidence constants outside config |
| M-3 | service.py (70 public functions) | God module mixing runs, jobs, config, memory, toolmaker, speculation, federation |
| M-4 | storage/db.py:37–46, event_log.py:180 | New SQLite connection per operation, 4 DBs, no pooling → lock-contention risk under parallel agents |
| M-5 | storage/migrations.py:175–227 | Version gating is idempotent but individual `upgrade()` idempotency is an undocumented contract |
| M-6 | storage/memory.py:59–62,152–189 | HNSW index freshness by mtime only; corrupt/stale index loads silently (load failure returns False unlogged) |
| M-7 | config.py:18–34 | Module-level mutable config dict + `_find_config()` executed at import time from CWD → hidden global state, hard to isolate in tests |
| M-8 | models.py:142 | `to_manifest()` omits `consensus_graph` → field lost on manifest round-trip |
| M-9 | pyproject.toml:91 | Coverage gate 60% vs 80% standard; actual coverage **unverified** (venv has no pytest — run `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/pytest --cov=orchestra` from a scratch CWD after H-9 lands) |
| M-10 | claude_sdk.py run/async_run/stream (84/75/66 lines) | Cache-zone + streaming logic triplicated; fixes must be applied 3× |
| M-11 | engine coupling | `_finalize` chains consensus→federation→memory with no transactional guarantee; approval/policy logic duplicated across 3 runners |

### LOW

- Dead code: `tui/app.py` + 3 root widgets (~280 lines), `engine/command_queue.py` (0 importers).
- Hardcoded model names/prices in `registry.py` and 7+ provider `_DEFAULT_MODELS` — will rot.
- Timeout semantics differ per provider (aiohttp `ClientTimeout` vs subprocess kwarg vs SDK param).
- 13 `launch_*.log` debug files at repo root (covered by C-4 cleanup).
- Error-message format inconsistent across providers (`[ERROR] …` vs bare exceptions).
- **New (2026-07-06):** a misplaced `if __name__ == "__main__": app()` guard sits mid-file in `cli.py` (~line 1783), BEFORE several `app.add_typer()` calls — so `python -m orchestra.cli` exits before registering playbook/cue/sessions/debate/report. Harmless in normal use (the real entry point `orchestra` / `python -m orchestra` imports the module fully first), but the guard should move to the end of the file. Discovered during C-2. LOW.
- **New (2026-07-06), now FIXED:** `engine/parallel.py`'s `_enforce_budget_guard[_async]` called `logger.warning(...)` but `logger` was never defined/imported in the module → a latent `NameError` crash on the budget-exceeded branch (no test exercised it). Inadvertently fixed by the parallel-package split (commit cef5c97): the guards moved to `parallel/budget.py`, which defines `logger`. The budget guard now logs correctly instead of crashing.
- **New (2026-07-06):** the engine test suite patches `orchestra.engine.parallel.<name>` (and `orchestra.engine.runners.common.<name>`) via `mock.patch`, which only resolves names whose defining module IS the package `__init__`. This forces most runner/orchestration functions to stay in `__init__.py`, capping how far parallel.py (713 now) and especially common.py (1188) can be split by pure movement. A deeper split of common.py requires first refactoring the tests' patch targets — a separate, larger task. LOW/structural.
- **New (2026-07-06):** `toolmaker/validator.py` over-blocks `re.compile()` (and any `.compile()` method): `_call_name()` returns `"compile"` for `x.compile(...)`, which hits `_BLOCKED_CALL_NAMES`. Tool scripts cannot use `re.compile`. Fix: narrow the `compile`/`eval`/`exec` bare-builtin block to `ast.Name` func nodes only (not `ast.Attribute`). Pre-existing, discovered during C-5. LOW.

## 4. Test Suite Assessment

97 files / 1,302 test functions across unit, integration, e2e, adversarial and benchmark
categories — a substantial suite. Two structural problems: (1) **no path isolation** (H-9), the
proven source of repo-root pollution; (2) coverage gate at 60% vs the 80% standard, with the
actual number currently unverifiable (M-9). `pytest.ini`/`pyproject` markers (`slow`,
`integration`, `e2e`) are in place and `--strict-markers` is set.

## 5. Prioritized Refactor Roadmap

Order minimizes risk (hygiene first, then correctness, then structure):

| # | Plan | Addresses | Size |
|---|---|---|---|
| 1 | [refactor-repo-hygiene.md](plans/refactor-repo-hygiene.md) | C-3, C-4, H-9 | S — one sitting |
| 2 | Hotfixes (no plan file needed): C-1 lock-across-await, C-2 cli duplication, H-4 `_last_usage` | C-1, C-2, H-4 | S each, independently committable |
| 3 | [refactor-toolmaker-hardening.md](plans/refactor-toolmaker-hardening.md) | C-5 | M |
| 4 | [refactor-cli-split.md](plans/refactor-cli-split.md) | H-1, H-7, H-8 | M |
| 5 | [refactor-runners-common.md](plans/refactor-runners-common.md) | H-1, H-6, M-11 | L — phased |
| 6 | ~~refactor-tui-consolidation~~ — superseded 2026-07-09: entire TUI removed ([archived plan](plans/archive/refactor-tui-consolidation.md)); TUI-scoped findings (H-8 widget excepts, depth-19 parser) are moot | H-8 (tui kısmı), LOW dead code | done |
| 7 | Boundary validation pass (server/mcp/federation) — fold into future federation work | H-2, H-3 | M |
| 8 | [roadmap-claude-code-agent-visibility.md](plans/roadmap-claude-code-agent-visibility.md) — live agent visibility inside Claude Code (statusline + MCP); Phase 0 also fixes bus.py:81 logger bug, service.py:262/583 worker-CWD bug, mcp_server tools/list bug | new feature + 3 verified bugs | L — 4 phases |

Per the refactoring protocol: one plan = one change-set series, tests green after each phase,
`refactor:` commits, never mixed with behavior changes.

## 6. Future-Development Readiness (focus areas)

**Engine/routing:** biggest friction is the sync/async duplication in parallel.py
(`_run_one_agent` 207 lines ≈ `_run_one_agent_async` 219 lines), policy logic copied across the
three runners, and alias fragmentation (M-1). Land roadmap items 2 and 5 before feature work.

**Federation/infra:** the CRDT core is verified sound and a good foundation. Production
enablement is blocked by: optional-by-default auth (H-3), no payload schema (H-2), no delivery
ACK/retry, no inbox garbage collection (TTL exists but nothing prunes), static peer list with no
health checks, and silent LWW conflict resolution. Treat these as the feature backlog for any
federation milestone.

## 7. Re-review 2026-10-12

**Method:** four subsystem reviews (engine core; providers and network boundary; storage, config
and routing; CLI, dashboard, tests and repo health), each critical claim re-checked in the source,
plus the fast suite, coverage, mypy and a fresh Python 3.11 install. Repo at `e35314f`:
186 files, 36,336 lines, 2,086 tests, 68% coverage, 117 mypy errors, 108 functions over 50 lines,
263 broad `except Exception`.

**State of the 2026-07 findings:** C-1, C-2, C-3, C-4, H-4 (rename only, see P-1), H-9 fixed.
C-5 is not fixed in substance (T-1). H-2, H-3 (now guarded against browsers, still token-optional),
M-1, M-2, M-4, M-5, M-8 remain open.

### Fixed on `fix/audit-2026-10`

| Finding | What was wrong | Commit |
|---|---|---|
| Tests wrote into the project's real memory.db and asked a live Ollama for vectors; replay logs were created relative to the CWD | 2,510 of 2,646 memory rows were test tasks, read by the router | `e64463c` |
| Gateway (token-less on localhost), dashboard and WebSocket daemon ran agents for any web page; one cross-site POST opened the dashboard's dispatch and approval gates | remote code execution from a visited page | `7641541`, `b91339e` |
| `nodes-scan` sent the gateway token to all 254 probed hosts and saved whatever answered under the name it reported | token disclosure, node hijack, TOML injection through the name | `7641541`, `b91339e` |
| `orchestra claude` started a token-less gateway and sent the user's own ANTHROPIC_API_KEY to it | | `7641541`, `b91339e` |
| `_finalize` recomputed the status: denied and over-budget runs ended COMPLETED, cancelled runs FAILED, an empty plan passed | | `97889c8` |
| The budget check read `[execution] max_budget_usd`; the documented `[limits]` key was only reported | no budget was ever enforced | `97889c8` |
| A mistyped alias ran (and billed) cdx-deep → cld-deep | | `331bfe4` |
| A code review with no readable reviewer answer said "No defects found" | | `950ff2e` |
| `list_runs` / `get_stats` / `get_reputation` re-indexed every run folder and dropped the events of runs whose events.jsonl was missing | | `55dffaf` |
| Every run start erased the recorded review outcomes of all aliases | reputation delta was always 0 | `cd1742b` |
| `ask` / `run` exited 0 on failure; `logs`, `cancel`, `trace`, `checkpoints` exited 0 for unknown runs and created their folders; `cancel` bypassed the service | | `f007978` |
| dashboard_render.py used Python 3.12 f-string syntax in a package that declares 3.11 | `orchestra dashboard` could not start on 3.11 | `7ce5952` |
| No CI; `.coveragerc` and `mypy.ini` shadowed pyproject (coverage gate never applied); a fresh `pip install -e ".[dev]"` could not run the suite (3 missing packages, mcp 2.x) | | `e043df8` |
| Dashboard audit and incident panels read an empty database; an unused http.server dashboard bound 0.0.0.0 without auth | | `e806a3d` |

### Open — correctness

| ID | Location | Finding |
|---|---|---|
| E-1 | providers/base.py:79, engine/runners/planned_runner.py:102 | Cancel stops only Claude/Codex/Gemini processes: Kimi and API providers run to their timeout; `run_planned` keeps starting nodes after a cancel; runners never re-check cancellation between phases |
| E-2 | engine/runners/common.py (ask), dual_runner.py, planned_runner.py | An exception leaves the run RUNNING forever (no status is set before re-raising; `run_planned` has no handler; nothing reaps stale runs) |
| E-3 | models.py `to_manifest`, engine/state_suspension.py:35 | Resume loses data: manifests hold no `stdout_log`, the paused-run row holds no `stances`, `follows_run_id`, `parent_run_id` or costs; a resumed consensus run is synthesised as plain dual on empty outputs |
| E-4 | engine/provider_guard.py:15 | A successful answer that contains "auth", "timeout", "connection"… counts as a provider failure (threshold 2 in the example config → 300 s lock-out); `start()`/`finish()` are not in try/finally, so a cancelled agent leaks a concurrency slot |
| E-5 | engine/runners/common.py:953 | A verifier outage fails an otherwise good run; the default verifier alias is `gmn-fast` although Gemini is off |
| E-6 | engine/reflection.py:140, degradation.py:108, supervisor.py:46, eval_harness.py:95, calibration.py:65 | Internal steps are run without `internal=True, parent_run_id=…`: they are prefixed, cached, indexed to memory as tasks and not cancelled with their parent |
| E-7 | engine/runners/common.py:1204 | Approval and deny are evaluated after the agents ran; the answer is cached before the gate; `sandbox_only` is never acted on |
| E-8 | engine/process_group.py:95 | The synchronous Popen path (pid registration, inactivity timeout) is unreachable; API providers raise NotImplementedError through it; `cwd` is dropped on both paths |
| E-9 | gateway_dispatch.py:63 | `/v1/*` calls `provider.run` directly: no run record, cost, fallback or timeout; `alias@node` and Ollama `model@backend` do not work there; a provider error is returned as a 200 answer |
| E-10 | engine/planned_runner.py:156, engine/graph.py:95 | A planned node is queued when any one predecessor finishes (no join) |
| E-11 | cli.py, engine/parallel | `alias#mode` is applied by the gateway only; on the CLI path the mode is silently ignored |

### Open — stored data

| ID | Location | Finding |
|---|---|---|
| D-1 | storage/fts_cache.py:123, config.py:183 | The answer cache key is alias + scrubbed prompt in a database shared by all projects: the same question in another repo gets this repo's answer; prompts that differ only in an e-mail or number collide; a hit sets confidence 1.0 |
| D-2 | service.py `forget_runs`, storage/memory.py:404 | `forget` leaves the text in `memory_events`, in the replay store (keyed `global_idempotency`) and in the answer cache; with `memory.typed_enabled` it can delete a note with the same id |
| D-3 | ~~engine/idempotency.py:133~~ (removed 2026-10-21), storage/memory.py | The replay store is scanned in full on every ask and never pruned; memory TTL is never enforced (`cleanup_expired` has no caller); `memory_events` grows with every search |
| D-4 | storage/memory.py:281 | Changing the embedding model breaks memory search (mixed vector sizes raise on index rebuild) |
| D-5 | engine/artifacts.py:63 | Manifests are written non-atomically and read-modified-written across processes |
| D-6 | storage/migrations.py:72 | A v1 `events` table makes `get_db()` raise before migration 001 runs; migrations 003/004 run without a transaction (was M-5) |

### Open — security

| ID | Location | Finding |
|---|---|---|
| T-1 | toolmaker/validator.py:115, sandbox.py:151 | Validator and sandbox do not contain anything (`f = os.system; f(...)`, `sys.modules[...]`, `pathlib.write_text`, readable `~/.ssh`, no network block). Nothing wires them in today |
| T-2 | mcp_server.py:415, engine/policy_engine.py:109 | Over MCP an agent can approve its own gated run, write config (`skip_permissions`, node URLs) and read tokens; the default policy is allow-all |
| T-3 | providers/claude.py:109, fallback.py:149 | Permission profiles fail open: no profile → `--dangerously-skip-permissions`; `[providers.*] enabled = false` is honoured by Codex and Gemini only; text-only API aliases fall back to agentic CLIs |
| T-4 | engine/federation.py:146 | `/federation/push` writes a file named by the untrusted `content_hash` (path traversal); auth is optional. No command starts the server |
| T-5 | service.py `_flatten_toml_lines` | The config writer does not quote keys or escape newlines; reachable through MCP `orchestra_config_set` |
| T-6 | engine/pii_scrubber.py:25 | The scrubber rewrites ordinary numbers and e-mails in answers and MCP results, and misses `password=`, bearer tokens and config tokens |
| T-7 | remote nodes | Tokens travel over plain HTTP on every `/run`; `/identify` lets anyone on the LAN test a guessed token offline — use long random tokens |

### Open — providers and cost

| ID | Location | Finding |
|---|---|---|
| P-1 | openai_sdk.py:101, claude_sdk.py:124, deepseek.py, perplexity.py, azure_openai.py | Per-call usage is kept on the shared provider object; concurrent agents overwrite each other |
| P-2 | providers/ollama.py:53 | `temperature` and `seed` are sent at the top level of `/api/generate`, where Ollama ignores them; `async_run` drops HTTP errors |
| P-3 | engine/parallel/__init__.py:262, common.py:401 | Cost is completion tokens × one blended price (prompt tokens ignored); the approval gate prices every alias outside six hard-coded ones as gpt-5.4; unknown local models are billed |
| P-4 | providers/claude_sdk.py:21 | Pinned model ids are old; `seed` is passed to an API that has no such parameter |

### Open — routing

| ID | Location | Finding |
|---|---|---|
| R-1 | service.py:172, router/outcome_router.py:129 | Learned outcome routing never runs (wrong call signature, error swallowed); the history it would read has no writer |
| R-2 | router/classifier.py:132 | Word-prefix and substring matching sends plain questions to dual ("const" → "cons") at a confidence above the model-router threshold |
| R-3 | config.py:133 | `timeout_for_alias` does not match `alias#mode` or `alias@node` |
| R-4 | config.toml.example, config.py | Tier and persona defaults still name `gmn-*`; a run that asks Gemini gets its fallback, so dual can compare a model with itself |

### Open — structure and tests

- Sync/async twins in `engine/parallel/__init__.py` (229 / 244 lines) and the approval-gate block
  copied in three runners; `cli.py` 2,400 lines, `service.py` and `runners/common.py` ~1,400.
- Unwired modules (see CLAUDE.md "Active vs Dead Code"); ~25 config accessors with no caller.
- Coverage: `engine/eval_judge.py` 0%, `cli.py` 32%, `server.py` 31%, `engine/parallel` 37%,
  `service.py` 44%. About 23 tests assert nothing (18 in `tests/test_tracer.py`).
- Dashboard: `characters.build_roster()` probes every provider on the event loop on each poll;
  dispatch and graph runs bypass the run pipeline (no run record, cost, budget or approval gate).
- Rich markup: task and model text is printed unescaped in a dozen places in `cli.py`; a task
  containing `[/x]` breaks `orchestra ps`.
- SETUP.md and docs/architecture.md describe removed features (TUI, `critical` runner).

### Status on 2026-10-13, after `docs/plans/archive/roadmap-2026-10-engine.md`

Closed (each with a test that failed before the change):

| ID | How |
|---|---|
| E-1 | CLIs started through `BaseProvider.run` register their process group; the cancel is written before the kill; runners stop between phases. Open part: an API provider's HTTP call finishes and is discarded |
| E-2 | `_fail_crashed` in the three runners; `service.reap_stale_runs` |
| E-3 | A pause stores and restores the whole run; `artifacts.load_run`. Open part: the planned resume re-runs the verifier; `release_resume_claim` has no caller |
| E-4 | Guard slots are released on interruption; only a failed call is read for failure words |
| E-6 | Judges, critics and planners are internal steps; a retry keeps the first call's options |
| E-7 | A denied provider or an exhausted daily budget fails the run before any agent starts; the cache is written after the policy. Gates stay after the agents; `sandbox_only` is still not acted on |
| E-9 | Plain gateway replies are runs; streamed and native replies are recorded as steps |
| E-11 | `alias#mode` is applied in `run_ask` |
| D-2, D-4 | `forget` cleans memory, replay store and answer cache; the index survives a change of embedding size |
| T-1 | The toolmaker sandbox was removed, not hardened |
| T-2, T-5 | MCP approve / config-write are off by default, secrets are redacted, the TOML writer quotes and escapes |
| T-3 | No profile means read-only; `enabled = false` holds for every provider; API aliases have no agentic fallback |
| T-4 | The federation hash is checked before it names a file |
| P-1, P-3 | Usage per call; `engine/costing.py`. Open part: the approval-gate estimator (`common._estimate_run_cost`) has its own table |
| R-1 – R-4 | Outcome routing wired (opt-in); whole-word matching; timeouts for `#mode` / `@node`; tier agents that can run |
| M-8 | `consensus_graph` is in the manifest |

Found on the way: OTLP export never worked (the tracer imported an exporter the project does
not install); the suite's guard against real agents did not cover the async process path or
live Ollama.

Still open: E-5 (a verifier outage fails a good run), E-8 (the unreachable Popen path in
`process_group.py`), E-10 (planned nodes have no join), D-1 (the answer cache is shared across
projects), D-5 (manifests are not
written atomically), D-6 (migrations), T-6 (the PII scrubber), T-7 (tokens over plain HTTP),
P-2 (Ollama ignores temperature and seed as sent), P-4 (old pinned Claude SDK models), and the
structure, dashboard and Rich-markup items above. Not built from the roadmap: an MCP client
for agents (6.1), CLI session resume (6.3), a `ProviderResult` contract (3.1b), one eval
module (5.2b).

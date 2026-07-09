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

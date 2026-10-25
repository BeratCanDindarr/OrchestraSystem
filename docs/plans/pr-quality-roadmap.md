# PR Draft: Quality audit — verified bug fixes + structural refactors

Branch: `refactor/quality-roadmap` → `main` (14 commits)
Status: pushed manually; open PR at
https://github.com/BeratCanDindarr/OrchestraSystem/compare/main...refactor/quality-roadmap

---

## Title

refactor: quality-audit roadmap — verified bug fixes, repo hygiene, structural splits

## Summary

Executes the first tranche of `docs/quality-audit.md`: full architecture/quality review
deliverables, five verified bug fixes, repo hygiene, and three behavior-preserving
structural refactors. Full-suite verified after each batch — **zero new regressions**
(141 → 93 failures; every remaining failure pre-dates this branch and was individually
triaged as environmental/stale).

### Docs & review deliverables
- `docs/architecture.md`, `docs/quality-audit.md` (severity-ranked C/H/M findings),
  `docs/plans/refactor-*.md` (5 plans), root `CLAUDE.md`

### Repo hygiene
- Added `.gitignore`; untracked `.orchestra/` runtime state, live SQLite DBs, logs,
  `:memory:*` / MagicMock artifacts, egg-info (index 6293 → 309 files)

### Bug fixes (each with new covering tests)
- **C-1** `engine/parallel`: never hold `threading.Lock` across `await` in the budget
  guard (deadlock risk); enforced by a new AST lock-discipline test. Side effect: fixed
  a latent `logger` NameError on the budget-exceeded path.
- **C-2** `cli.py`: removed duplicated Typer command groups that silently shadowed the
  real ones (`config validate/init` were unreachable)
- **C-3** `FtsSearchCache(":memory:")` now truly in-memory (was creating a literal
  `:memory:` file on disk; root cause of repo-root junk files)
- **C-5** `toolmaker/validator.py`: block exec-style calls (`system/popen/exec*/spawn*/fork`),
  non-constant `open()` modes, and reflection with non-constant/dunder attrs
  (31 new adversarial tests)
- **H-4** providers (perplexity/deepseek/openai_sdk): store usage in `_last_usage` per
  the `BaseProvider` contract, so `last_usage()` works; new AST contract test
- **H-5** was re-verified and downgraded to false positive — intentionally not "fixed"

### Breaking change
- **TUI removed entirely** (user request, `feat!`): `orchestra/tui/` (3,621 lines),
  FleetDashboard, `launch`/`dashboard` CLI commands, `textual` dependency, 6 TUI test
  files. The aiohttp web dashboard remains.

### Structural refactors (behavior-preserving)
- `engine/parallel.py` (874 lines) → package `engine/parallel/` (`ui`/`budget`/`executor`
  + `__init__` re-exports; public API byte-identical, mock.patch targets preserved)
- Dead TUI v1 removed (`tui/app.py`, root widgets, dedicated e2e test; −517 lines)
- `cli.py`: shared config-value utils extracted to `orchestra/utils/config_values.py`;
  misplaced `__main__` guard moved to EOF

## Test plan
- [x] Full fast suite before/after each batch (`pytest -m "not slow"` from a scratch dir)
- [x] Baseline 141 failed / 1093 passed → now 93 failed / 1229 passed / 18 errors
- [x] Every remaining failure triaged: pre-existing (stale imports, headless TUI,
      live-provider, PII/redaction, approval/wasm categories) — reproduced identically
      on pre-branch commit `691a46d`
- [x] New tests added by this branch: AST lock guard, provider usage contract,
      CLI command registry, config-value utils, in-memory FTS cache, 31 adversarial
      validator tests
- [ ] CI run on GitHub (no CI config in repo yet — see quality-audit roadmap)

## Known follow-ups (tracked in docs/quality-audit.md)
- sync/async executor merge (runners-common P2), cli-split P2/P3, common.py split
  (blocked by mock.patch constraint), conftest path isolation, TUI P2/P3
- LOW: `federation_http.py:66` unvalidated `?limit=` (500 on non-int),
  perf-baseline auto-overwrite churn, validator `re.compile` over-block

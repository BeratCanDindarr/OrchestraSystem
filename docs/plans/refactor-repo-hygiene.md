# Plan: Repo Hygiene — untrack runtime state, fix pollution root causes

## Status
Draft

## Goal
Stop the repository from tracking its own runtime state and test artifacts, and fix the two
code-level root causes so the pollution cannot recur.

## Context
Audit findings C-3, C-4, H-9 (docs/quality-audit.md). 5,950 of 6,285 tracked files are
`.orchestra/` runtime state (including live SQLite DBs with `-wal`/`-shm` sidecars). 27 junk
files are tracked at the root. A secret scan of `.orchestra/runs/` found **no** credentials, so
plain `git rm --cached` is sufficient — no history rewrite needed.

## Phases

### Phase 1: Add .gitignore
- [ ] Create root `.gitignore`:
  ```
  .orchestra/
  .venv/
  *.egg-info/
  __pycache__/
  .pytest_cache/
  .coverage*
  htmlcov/
  launch_*.log
  :memory:*
  =*
  .DS_Store
  ```
- **Acceptance:** `git status` no longer offers runtime files.

### Phase 2: Untrack committed state (single commit)
- [ ] `git rm -r --cached .orchestra orchestra.egg-info`
- [ ] `git rm --cached ':memory:'* '<MagicMock'* launch_*.log .coverage '=3.9.0'`
- [ ] Delete the junk files from the working tree (keep `.orchestra/` locally — it is live state)
- **Acceptance:** `git ls-files | wc -l` drops from 6,285 to ~330; app still runs.

### Phase 3: Fix the `:memory:` fallback (C-3)
- [ ] `orchestra/engine/runners/common.py:117–121` — replace the broad
  `except Exception → FtsSearchCache(db_path=":memory:")` fallback: catch specific errors and
  either construct a true in-memory cache (explicit `sqlite3.connect(":memory:")` support inside
  `FtsSearchCache`) or disable caching for the session with a clear warning.
- [ ] Regression test: simulate `config.fts_cache_db_path()` failure; assert no `:memory:` file
  appears in CWD.
- **Acceptance:** test green; no file created on fallback.

### Phase 4: Test path isolation (H-9)
- [ ] Add session-scoped autouse fixture in `tests/conftest.py` that monkeypatches
  `config.artifact_root`, `config.db_path`, `config.fts_cache_db_path`, eventlog/memory paths to
  `tmp_path_factory` dirs (real `Path`s — never bare `MagicMock`).
- [ ] Grep tests for `MagicMock` patched into path functions; convert to `tmp_path`.
- **Acceptance:** full suite leaves `git status --porcelain` empty from a repo-root CWD.

## Risks
- Removing `.orchestra/` from the index while users have live state → use `--cached` only; verify
  nothing under `.orchestra/` is needed by fresh clones (config template should move to
  `mcp.json.example`-style `config.toml.example` if needed).

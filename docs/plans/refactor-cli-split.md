# Plan: Split cli.py and remove duplicated command groups

## Status
Draft

## Goal
Bring `orchestra/cli.py` (2,066 lines — 2.6× the 800-line standard) down to a thin Typer layer
that delegates all business logic to `service.py`.

## Context
Audit findings C-2, H-1, H-7, H-8 (docs/quality-audit.md). cli.py currently defines the
playbook/cue/sessions/debate command groups **twice** (lines 1788–1857 and 1874–1938, both
`add_typer`-ed), contains 200-line command bodies (`eval_cmd` 210, `diff_cmd` 188), calls
runners directly instead of `service.run_task()`, and duplicates utils that exist in service.py.

## Phases

### Phase 0: De-duplicate (behavior-preserving hotfix, ship first)
- [ ] Delete one copy of the duplicated groups; keep the later (currently-winning) definitions
- [ ] Remove the duplicate `app.add_typer(config_app)` (lines 589 vs 1858)
- [ ] Replace `except: continue` at cli.py:1805/1894 with specific exceptions + debug log
- [ ] Smoke test: assert `orchestra --help` command names are unique
- **Acceptance:** `orchestra --help` output unchanged except duplicates gone; tests green.

### Phase 1: Extract shared utils
- [ ] Move `_parse_cli_value` / `_toml_literal` / scalar parsing into `orchestra/utils/values.py`;
  import from both cli.py and service.py
- **Acceptance:** no copy-pasted util bodies remain (grep).

### Phase 2: Package the CLI by command group
- [ ] Create `orchestra/cli/` package: `__init__.py` (assembles `app`), `run_cmds.py`,
  `eval_cmds.py`, `inspect_cmds.py` (timeline/diff/replay), `playbook_cmds.py`, `cue_cmds.py`,
  `sessions_cmds.py`, `debate_cmds.py`, `config_cmds.py`, `dashboard_cmds.py`
- [ ] Keep `orchestra.cli:app` import path working (pyproject entry point unchanged)
- [ ] Target: no module >400 lines, no command body >50 lines (extract table-builders/helpers)
- **Acceptance:** `pip install -e . && orchestra --help` identical; e2e CLI tests green.

### Phase 3: Delegate to service.py
- [ ] `eval_cmd` and `run` commands call `service.run_task()` instead of `run_ask/dual/critical`
  directly (removes routing/cost logic duplication)
- **Acceptance:** behavior parity on `tests/test_cli_*`; no direct runner imports left in cli/.

## Dependencies
- None (Phase 0 is independently shippable; do it before any other cli work).

## Risks
- Typer callback/order sensitivities when moving commands → move one group per commit, run
  `tests/test_cli_dx.py` after each.
| 2026-10-22 | 2 | cli.py split into cli_observe.py and cli_admin.py next to it, not a package (`864a3d5`, docs/plans/archive/refactor-oversized-modules.md) |

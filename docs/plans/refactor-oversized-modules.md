# Plan: The four modules over 800 lines

## Status
In Progress

## Goal
Bring `config.py` (1132), `engine/runners/common.py` (1411), `cli.py` (1748) and
`storage/memory.py` (843) under 800 lines without changing behaviour.

`common.py` follows Phase 4 of `refactor-runners-common.md` (module names `policy_gate.py`,
`synthesis.py`); `cli.py` is a lighter form of `refactor-cli-split.md` Phase 2 (command
modules next to `cli.py`, not a package). This plan adds `config.py` and `memory.py`.

## Rules for every phase
- Delete before moving: a definition nothing calls goes (removed features left them).
- A moved name stays importable from its old module (re-export), so callers do not change.
- A test that patched a moved name in its old module patches it where it now lives.
- One phase, one commit; the fast suite and ruff pass after each.

## Phases

### Phase 1: config.py
- [ ] Remove the 23 accessors with no caller (dashboard_*, fleet_*, remote nodes, MCP client
  and config write, skills, federation sync, *_config() aggregates only tests read)
- [ ] If still over: move `validate_config` and block accessors to `orchestra/config_blocks.py`,
  re-exported from `config`, reading through `config.get` so env overrides and patches apply
- **Acceptance**: `config.py` < 800, suite green

### Phase 2: engine/runners/common.py
- [ ] Remove `_compress_blackboard_context`, `run_ask_async`, `run_dual_async` (no caller)
- [ ] Verification loop and synthesis → `runners/synthesis.py`
- [ ] Policy evaluation and approval gates → `runners/policy_gate.py`
- [ ] Further groups (prompt building, answer-cache helpers) only if still over
- **Acceptance**: `common.py` < 800, suite green

### Phase 3: cli.py
- [ ] Observability commands (trace, eval, timeline, diff, replay, checkpoints, undo,
  eval-stats, report) → `orchestra/cli_observe.py`
- [ ] Admin commands (statusline, config, memory, aliases, models, backends, backfill,
  version, doctor) → `orchestra/cli_admin.py`
- Both register on the same `app`; `cli.py` imports them last; command names unchanged
- **Acceptance**: `cli.py` < 800, `orchestra --help` lists the same commands, suite green

### Phase 4: storage/memory.py
- [ ] One cohesive method group of `SemanticMemory` (housekeeping) → a mixin module
- **Acceptance**: `memory.py` < 800, suite green

## Risks
- A test patches `module.name` and the code now looks the name up elsewhere → the suite shows
  it; the patch target moves with the code
- Circular import between `cli` and its command modules → they import `app` from `cli`, and
  `cli` imports them at the end

## Progress Log
| Date | Phase | Notes |
|------|-------|-------|
| 2026-10-22 | - | Written. Dead definitions counted: config 23 (120 lines), common 3 (24), cli 0, memory 0 |

# Plan: The four modules over 800 lines

## Status
Completed

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
- [x] Remove the 23 accessors with no caller (dashboard_*, fleet_*, remote nodes, MCP client
  and config write, skills, federation sync, *_config() aggregates only tests read)
- [x] If still over: move `validate_config` and block accessors to `orchestra/config_blocks.py`,
  re-exported from `config`, reading through `config.get` so env overrides and patches apply
- **Acceptance**: `config.py` < 800, suite green

### Phase 2: engine/runners/common.py
- [x] Remove `_compress_blackboard_context`, `run_ask_async`, `run_dual_async` (no caller)
- [x] Verification loop and synthesis → `runners/synthesis.py`
- [x] Policy evaluation and approval gates → `runners/policy_gate.py`
- [x] Further groups (prompt building, answer-cache helpers) only if still over
- **Acceptance**: `common.py` < 800, suite green

### Phase 3: cli.py
- [x] Observability commands (trace, eval, timeline, diff, replay, checkpoints, undo,
  eval-stats, report) → `orchestra/cli_observe.py`
- [x] Admin commands (statusline, config, memory, aliases, models, backends, backfill,
  version, doctor) → `orchestra/cli_admin.py`
- Both register on the same `app`; `cli.py` imports them last; command names unchanged
- **Acceptance**: `cli.py` < 800, `orchestra --help` lists the same commands, suite green

### Phase 4: storage/memory.py
- [x] One cohesive method group of `SemanticMemory` (housekeeping) → a mixin module
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
| 2026-10-22 | 1 | `fe55148` config.py 1132 → 740: 34 dead accessors removed, infrastructure blocks + validate_config → config_blocks.py |
| 2026-10-22 | 2 | `a44622c` common.py 1411 → 712: synthesis.py, policy_gate.py, prompting.py (the last two re-exported from common) |
| 2026-10-22 | 3 | `864a3d5` cli.py 1748 → 680: cli_observe.py, cli_admin.py; same 47 commands |
| 2026-10-22 | 4 | `19e59ec` memory.py 843 → 678: HNSW index → memory_index.py (HnswIndexMixin) |
| 2026-10-22 | - | `3808e12` mypy 90 → 85. Not done: [redaction] patterns in config.toml.example are read by nothing (never were) |

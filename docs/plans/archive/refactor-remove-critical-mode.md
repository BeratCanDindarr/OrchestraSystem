# Plan: Take out the critical mode

## Status
Completed

## Goal
Orchestra has no `critical` mode: consensus mode covers what it was used for, and one honest
mode replaces a name ("multi-round consensus") the code never lived up to.

## Phases

### Phase 1: Nothing calls it
- [x] Router: mutation requests and "urgent" tasks route to `dual`; the LLM route no longer offers it
- [x] Service, job queue, MCP schemas: the mode and its `pause_after_round1` flag go
- [x] CLI: `run critical`, `--require-approval`; `eval` compares ask / dual / consensus
- [x] Gateways (OpenAI and Anthropic compatible), Unity daemon, batch runner, patterns, learning adapter
- [x] Resume: a paused run of the old mode is refused with a clear message
- **Acceptance criteria**: no import of `critical_runner` outside the runner package; fast suite green

### Phase 2: The runner goes
- [x] `engine/runners/critical_runner.py`, `run_critical_async`, the exports, and the tests of the mode
- [x] README, skill, CLAUDE.md, config example
- **Depends on**: Phase 1
- **Acceptance criteria**: `grep run_critical` is empty; fast suite green; `orchestra run auto` and
  `orchestra eval` work against real agents

## Risks
- Runs of the old mode already on disk → they stay readable (`ps`, `logs`, `timeline`); only resuming one is refused
- The approval gate lived partly in this mode → `ask`, `dual` and `planned` keep their own gate (privileged phrases), covered by existing tests
- Outcome history contains the old mode name → the learned router ignores modes it does not know

## Progress Log
| Date | Phase | Notes |
|------|-------|-------|
| 2026-10-10 | 1 | Callers moved to dual / consensus; `pause_after_round1` and `--require-approval` gone |
| 2026-10-10 | 2 | Runner and its tests gone; README, skill, guides and example config updated; fast suite green |

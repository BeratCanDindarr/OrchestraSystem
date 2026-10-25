# Plan: Decompose engine/runners/common.py and engine/parallel.py

## Status
Draft

## Goal
Fix the lock-across-await defect, merge the duplicated sync/async agent executors, and split the
two oversized engine modules (1,188 + 874 lines) into focused units — without changing runner
behavior.

## Context
Audit findings C-1, H-1, H-6, M-11 (docs/quality-audit.md). This is the highest-risk refactor
(hot path for every run); phase it strictly and keep each phase independently green.

## Phases

### Phase 0: Correctness hotfixes (ship before any restructuring)
- [ ] **C-1:** `parallel.py:576` — `with lock:` block contains
  `await _enforce_budget_guard_async(...)` (line 660). Move the guard call outside the critical
  section (compute snapshot under lock, await after release), or use `asyncio.Lock` in the async
  path with narrow scopes
- [ ] Add a regression test: two concurrent agents where the budget guard sleeps; assert the
  second agent's status updates are not blocked
- **Acceptance:** `tests/test_runner_integration.py` + parallel tests green; no `await` inside
  any lock context (enforce with the AST check below).

### Phase 1: Guardrail
- [ ] Add a lightweight AST lint (test or pre-commit) failing on `Await` nodes inside
  `with *lock*:` blocks in `orchestra/engine/`
- **Acceptance:** lint green on main, red on a synthetic violation.

### Phase 2: Unify agent execution
- [ ] Merge `_run_one_agent` (240–446) and `_run_one_agent_async` (449–668) into one async
  implementation; the sync entry becomes `asyncio.run()`/`asyncio.to_thread()` glue
- [ ] Extract completion handling (`status/error/failure_kind` assignment blocks) into
  `_handle_agent_completion(agent, result)` — all `OrchestraRun` mutation happens in
  lock-guarded helpers only (first step toward H-6 immutability)
- **Acceptance:** dual/critical e2e tests green; line-for-line diff of emitted events identical
  on a recorded run replay.

### Phase 3: Split parallel.py (874 → 3 modules)
- [ ] `engine/parallel/ui.py` — Rich Live table rendering
- [ ] `engine/parallel/executor.py` — unified agent execution + retry
- [ ] `engine/parallel/budget.py` — budget guard + cost tracking
- [ ] `engine/parallel/__init__.py` re-exports `run_parallel`/`run_parallel_async` (import
  compatibility)
- **Acceptance:** all engine tests green; no module >400 lines.

### Phase 4: Split runners/common.py (1,188 → 4 modules)
- [ ] `runners/ask_impl.py` — `_run_ask_impl` + cache lookup
- [ ] `runners/policy_gate.py` — `_evaluate_policy`, approval gates, suspension
- [ ] `runners/synthesis.py` — `_synthesize_if_possible`, `_run_verification_loop`,
  `_record_round1_review`
- [ ] `runners/common.py` keeps shared helpers (`_prefixed`, budget checks, `_finalize` split
  into `finalize_consensus/federation/memory`) and re-exports the old names
- **Acceptance:** grep shows no stale imports; full suite green after each extraction commit.

## Dependencies
- `docs/plans/refactor-repo-hygiene.md` Phase 4 (test isolation) should land first so the suite
  can be trusted as the safety net.

## Risks
- Hidden import cycles between runners ↔ parallel ↔ artifacts → extract in dependency order
  (ui/budget first, policy_gate last); one extraction per commit, revert-friendly.
- Behavior drift in event emission order → use event-log replay comparison as the oracle.
| 2026-10-22 | 4 | common.py split differently: synthesis.py, policy_gate.py, prompting.py (`a44622c`, docs/plans/archive/refactor-oversized-modules.md); `_run_ask_impl` stays in common |

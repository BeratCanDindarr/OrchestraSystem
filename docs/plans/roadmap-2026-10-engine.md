# Plan: Engine roadmap after the 2026-10 review

## Status
In Progress

## Goal
Make the run pipeline do what it promises (cancel, fail, resume, account), remove what is not
wired, then add the features a single developer with local CLIs and Ollama gets most from.

Source: `docs/quality-audit.md` §7 (finding IDs below) and Orchestra run `35b5ca4d` (web
research, dual mode). The UI is out of scope.

## Phases

Each block is one change set: tests first, suite green, one commit. A block is not started
before the previous one is committed.

### Phase 1: Correctness of a run
- [x] 1.1 A run that raises ends FAILED, and stale RUNNING runs are reaped (E-2)
- [x] 1.2 Cancel stops every provider and the runners between phases (E-1, E-4 slot leak)
- [x] 1.3 Resume keeps what the run had: outputs, stances, follow-up, costs (E-3)
- [x] 1.4 Per-call usage in a ContextVar for every provider (P-1)
- **Acceptance criteria**: each finding has a test that failed before the change

### Phase 2: Remove what is not wired, wire routing
- [x] 2.1 Delete unreachable modules: toolmaker chain, `engine/skills/`, `mcp_fastlane`,
      `patterns`, `integration_hooks`, `adapters/`, `integrations/`.
      Kept on purpose: `batch_runner` (the eval tests use it; decide in 5.2), `storage/toolmaker.py`
      (service and MCP use it), `mcp_client` (6.1 wires it). `federation_server/` is kept too:
      removing only the transport would leave `federation.enabled` filling an outbox nobody
      reads, so it is all or nothing — owner's decision; its path traversal (T-4) is fixed
- [x] 2.2 Outcome routing reads the outcomes that finalize writes (R-1)
- [x] 2.3 Classifier matches whole words; tiers and personas name agents that can run (R-2, R-4)
- [x] 2.4 `timeout_for_alias` understands `alias#mode` and `alias@node` (R-3)
- **Depends on**: Phase 1
- **Acceptance criteria**: no module without an importer; a routed task records and reads outcomes

### Phase 3: Cost and one run pipeline
- [x] 3.1 Usage (prompt / completion / cached) and cost in one module, price table from config,
      local models cost nothing (P-3)
- [ ] 3.1b `ProviderResult` returned by providers instead of `tuple[str, int]` + `last_usage()`
      (touches every provider and its tests; not started)
- [ ] 3.2 Gateway `/v1/*` goes through the run pipeline; `alias#mode` works on the CLI (E-9, E-11)
- **Depends on**: 1.4
- **Acceptance criteria**: a gateway call leaves a run with cost; a local model run costs 0

### Phase 4: Security boundary
- [ ] 4.1 No profile means read-only, not `--dangerously-skip-permissions`; `enabled = false`
      is honoured by every provider; text-only aliases do not fall back to agentic CLIs (T-3)
- [ ] 4.2 MCP: an agent cannot approve its own run, write config or read tokens by default (T-2, T-5)
- [ ] 4.3 Deny and gate are evaluated before the agents run where the policy allows it (E-7)
- **Acceptance criteria**: each rule has a test; behaviour changes are listed in CLAUDE.md

### Phase 5: Observability and evaluation
- [ ] 5.1 OpenTelemetry GenAI attribute names, OTLP export (Langfuse / Phoenix can receive it)
- [ ] 5.2 One eval harness (judge steps are internal steps, E-6); results reach routing

### Phase 6: Features
- [ ] 6.1 MCP client: agents can use external MCP tools
- [ ] 6.2 A writing agent works in its own git worktree; apply or discard afterwards
- [ ] 6.3 CLI session resume for follow-ups (`claude --resume`)
- [ ] 6.4 Structured output: schema, validation, one retry (review, supervisor, judges)
- [ ] 6.5 Memory hygiene: `forget` covers every store, TTL is enforced, embedding model is versioned (D-2, D-3, D-4)

## Dependencies
- Internal: 3.1 needs 1.4; 2.2 is more useful after 3.1 (cost data); 6.2 needs 4.1
- External: none; nothing here needs a new service

## Risks
- Tests patch names per runner module → keep the names, inject collaborators
- Deleting modules removes their tests too → delete in one commit per module group, suite green
- Fail-closed permissions change what `cld-*` agents may do → document, keep an explicit opt-in

## Not doing
Temporal/DBOS, A2A beyond the agent card, a trained router, a vector database, a WASM sandbox,
a group-chat engine (reasons in run `35b5ca4d`).

## Progress Log
| Date | Phase | Notes |
|------|-------|-------|
| 2026-10-12 | — | Plan written |
| 2026-10-12 | 1.1 | `_fail_crashed` in the three runners; `service.reap_stale_runs` on `ps` / `list_runs` (`[orchestra] stale_run_seconds`, default 7200) |
| 2026-10-12 | 1.2 | CLIs started in `BaseProvider.run` register their process group (Kimi included); cancel is written before the kill; `artifacts.adopt_cancel` keeps a cancel from being overwritten and stops dual/planned between phases; guard slots are released on interruption and a good answer is no longer read for failure words. API providers (HTTP in a thread) still finish their call: their result is discarded |
| 2026-10-12 | 1.3 | A pause stores the whole manifest plus each answer and is restored with `from_manifest`; `artifacts.load_run` reads answers back from the agent logs for the manifest-based resume and fleet finalize; `consensus_graph` is in the manifest (M-8). Still open from E-3: the planned resume re-runs the verifier, `release_resume_claim` has no caller |
| 2026-10-12 | 1.4 | `BaseProvider._last_usage` is a property over a ContextVar (the SDK providers' assignments did not change); the default `async_run` brings usage back from its worker thread; `reset_usage()` before every attempt. Phase 1 done |
| 2026-10-12 | 2.1 | Removed 9 unreachable module groups and their tests (suite 2190 → 2076). Federation kept, `/push` hash and `?limit=` validated |
| 2026-10-12 | 2.2 | `append_outcome` stores task tokens (steps and cancelled runs are not outcomes); `load_history` + `OutcomeRouter.suggest` (success rate on similar tasks, cheaper mode on a tie, None when the history is silent); `auto` consults it only with `[outcome_routing] enabled = true`. Removed: EV = confidence/cost, the 5 % random mode, the substring keyword fallback |
| 2026-10-12 | 2.3 | `_mentions` matches an English term as a word with its inflections and a Turkish stem at a word start; `classify` uses it too. `agents_for_tier` replaces agents only when two of them would be answered by the same model |
| 2026-10-12 | 2.4 | Timeout lookup: exact name, then without `#mode`, then the bare alias. Phase 2 done |
| 2026-10-12 | 3.1 | `engine/costing.py` (`usage_of`, `cost_of`, `charge`), `config.price_of` (number or input/output/cached table; unlisted local Ollama model = 0), agent record carries `prompt_tokens`, `cached_tokens`, `cost_source`; a failed call that reported usage is charged. The approval-gate estimator (`common.py` `_estimate_run_cost`) still has its own table |

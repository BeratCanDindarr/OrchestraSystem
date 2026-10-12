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
- [ ] 1.2 Cancel stops every provider and the runners between phases (E-1, E-4 slot leak)
- [ ] 1.3 Resume keeps what the run had: outputs, stances, follow-up, costs (E-3)
- [ ] 1.4 Per-call usage in a ContextVar for every provider (P-1)
- **Acceptance criteria**: each finding has a test that failed before the change

### Phase 2: Remove what is not wired, wire routing
- [ ] 2.1 Delete unreachable modules: toolmaker chain, `engine/skills/`, `batch_runner`,
      `patterns`, `integration_hooks`, `federation_server/` (CRDT core stays), dead config accessors
- [ ] 2.2 Outcome routing reads the outcomes that finalize writes (R-1)
- [ ] 2.3 Classifier matches whole words; tiers and personas name agents that can run (R-2, R-4)
- [ ] 2.4 `timeout_for_alias` understands `alias#mode` and `alias@node` (R-3)
- **Depends on**: Phase 1
- **Acceptance criteria**: no module without an importer; a routed task records and reads outcomes

### Phase 3: Cost and one run pipeline
- [ ] 3.1 `ProviderResult` with usage (prompt / completion / cached), price table from config,
      local models cost nothing (P-3)
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

# Plan: Live parallel-agent visibility inside Claude Code

## Status
In Progress — Phases 0-3 complete (branch feat/agent-visibility; Phase 0: e11d748..2fbe603,
Phase 1: 0a1d509..3c7cbb8, Phase 2: 80a5d84..cdfc305, Phase 3: 27af016..3ef3cfc).
Core roadmap (boxes + dispatch + approval) done. Remaining: optional Phase 4 (watch --all).

## Goal
Make Orchestra's parallel CLI agents (codex/gemini/claude) visible live inside the Claude
Code session — per-agent rows (alias, state, elapsed, cost, last output line) rendered via
the Claude Code statusline, with MCP-based dispatch/narration and approval gates as follow-ups.
The deleted Textual TUI is NOT coming back; no standalone web UI.

## Context

Research findings (Claude Code docs + code exploration, 2026-07-09):

- Claude Code offers **no custom transcript panels** (no plugin UI API); MCP progress
  notifications are received but not displayed (tool calls render collapsed).
- The **statusline is the only persistent live surface**: a command configured in
  `settings.json` (`statusLine: {type: "command", command, refreshInterval >= 1}`), invoked
  with session JSON on stdin, supporting multi-line ANSI output — one row per agent gives
  the "subagent boxes" feel.
- **MCP elicitation** is supported (server-initiated dialogs — good for approval gates);
  hooks can inject context but not render UI.
- Orchestra already writes `manifest.json` live at agent start/retry/pid-change
  (engine/parallel/__init__.py:110, 319). Genuinely missing live data: per-agent last
  output line (logs written only at completion) and cost (computed at completion).
  Elapsed is derivable from `start_time`.
- `stream_callback` carries no alias; the only place alias+chunk meet is
  `_scrubbed_stream_callback` inside `_run_one_agent`/`_run_one_agent_async`
  (parallel/__init__.py:167-169, 376-378).
- The WebSocket daemon cannot observe runs started in other processes (event bus is
  in-process only) — **file-based live state is the transport**.
- Measured CLI latency: `python -m orchestra version` = 84ms warm → a Python statusline
  renderer is viable without a shell/jq caching layer.

Verified blocker bugs folded into Phase 0:
- `events/bus.py:81` uses undefined `logger` (only `_bus_logger` exists) — bus subscriber
  exceptions become silent NameErrors; would kill the live-status writer invisibly.
- `service.py:262/583` spawn job-worker/daemon with `cwd=str(PACKAGE_ROOT)`; since
  `config.py` walks up from CWD at import time, MCP/async-submitted runs write artifacts
  into the Orchestra package's `.orchestra`, not the caller's project.
- `mcp_server.py:136` `tools/list` advertises only 1 of 9 registered tools; unknown
  methods with an id get no response (client hangs); `initialize` declares an
  unimplemented `resources` capability.

## Phases

### Phase 0: Live-state channel (foundation, ~3-4h)
- [ ] Fix `orchestra/events/bus.py:81` — `logger` → `_bus_logger` (prereq: subscriber errors must be visible)
- [ ] New `orchestra/live_status.py`: `LiveStatusWriter` — subscribes via `bus.subscribe_all()`,
      maintains `.orchestra/live_status.json` (schema: version, updated_at, runs{run_id:
      {mode, status, started_at, heartbeat, agents{alias: {status, model, started_at,
      ended_at, last_line, est_cost_usd}}}}); atomic tmp+`os.replace`; single internal lock
      (bus handlers run in a 4-worker pool); prune runs finished >10min; stale heartbeat
      marking; handles agent_started/agent_retry/agent_failed/agent_completed/
      agent_cancelled/run_* (all already emitted through `engine/artifacts.py:append_event`)
- [ ] Explicit `live_status.install()` at process entry points: `service.run_task()`,
      job-worker path (`service.execute_job`), `server.py` startup — zero executor changes
      for lifecycle events
- [ ] Per-agent last_line: extend `_scrubbed_stream_callback` in BOTH
      `_run_one_agent` and `_run_one_agent_async` (parallel/__init__.py:167-169, 376-378)
      to call `live_status.note_stream(run_id, alias, chunk)` — throttled >=500ms/agent,
      ANSI/control-char stripped (PII already scrubbed upstream); no writes while holding
      the run lock
- [ ] Fix `service.submit_run`/`start_daemon` worker CWD (service.py:262, 583): pass the
      caller's project root (env `ORCHESTRA_PROJECT_ROOT` honored by config, or
      `cwd=<project root>`) so MCP-submitted runs land in the right `.orchestra`
- **Acceptance criteria:** mid-`orchestra run dual`, `.orchestra/live_status.json` shows both
  agents `started` with updating last_line; file is valid JSON at every read (atomicity);
  finished runs pruned; a job-worker run started from a foreign CWD lands in the caller's
  `.orchestra`.
- **Tests:** new `tests/test_live_status.py` — synthetic bus events → expected JSON;
  concurrent emit stress (no torn reads); throttle behavior; prune/stale logic;
  job-worker cwd regression.

### Phase 1: Statusline renderer — "the boxes" (~3h)
- [ ] New `orchestra/statusline.py` (stdlib-only; must NOT import typer/rich/config.py —
      avoids config's import-time CWD walk; target <50ms): read Claude Code session JSON
      from stdin, walk up from `workspace.current_dir` to find `.orchestra/live_status.json`;
      render header line (short run_id, mode, total cost) + one ANSI row per active agent
      (state icon/color, alias, model, elapsed computed from started_at, cost or `—`,
      truncated last_line); cap ~5 agent rows + "+N more"; stale heartbeat → dim "stale";
      no active runs → single dim idle line
- [ ] `python -m orchestra.statusline` entry (`__main__` handling inside the module)
- [ ] `cli.py`: `orchestra statusline` group (added ONCE, near `watch`) — `render` (manual
      test) and `install` (non-destructive merge into `.claude/settings.json`:
      `statusLine: {type: "command", command: "<abs venv python> -m orchestra.statusline",
      refreshInterval: 2}`; back up and require `--force` if a statusLine already exists)
- **Acceptance criteria:** with the statusline installed, `orchestra run dual "x"` shows two
  agent rows updating in the Claude Code footer within ~2s of state changes; renderer
  exits <150ms.
- **Tests:** golden-output tests (fixture live_status.json + fake stdin → expected ANSI
  text); stale/missing/corrupt-file handling; latency assertion; settings.json merge
  (tmp dir).

### Phase 2: MCP dispatch + narration (~3-4h)
- **Depends on:** Phase 0 (cwd fix — else MCP runs are invisible to the statusline)
- [ ] Fix `orchestra/mcp_server.py`: `tools/list` enumerates ALL registered tools with real
      inputSchemas (add schema metadata to `engine/mcp_registry.py` REGISTRY.register);
      respond with JSON-RPC error `-32601` for unknown methods carrying an id (currently
      hangs clients); drop the unimplemented `resources` capability from `initialize`
- [ ] Ensure exposed: orchestra_submit_run, orchestra_get_run, orchestra_get_logs,
      orchestra_wait_for_run, orchestra_list_runs, orchestra_cancel_run
- [ ] Add `.mcp.json` to this repo (self-hosting, based on mcp.json.example)
- [ ] New `.claude/skills/orchestra-dispatch/SKILL.md`: dispatch via orchestra_submit_run
      (non-blocking), poll orchestra_get_run, optionally launch `orchestra watch <run_id>`
      as a background Bash task, summarize results with per-agent outcomes
- **Acceptance criteria:** in Claude Code, "run this task on dual agents via orchestra" →
  Claude dispatches non-blocking, statusline shows agents live, Claude reports completion
  with costs; `tools/list` returns 6+ tools; `resources/list` no longer hangs.
- **Tests:** extend `tests/test_mcp_server_unit.py` — tools/list completeness + schema
  shape, unknown-method error response, initialize capabilities; submit→get_run round-trip
  in a tmp project verifying artifact location.

### Phase 3: Approval gates (~2h fallback + ~4h elicitation)
- **Depends on:** Phase 2
- [ ] 3a (ship first): tool-based approvals — expose `orchestra_continue_run`
      (service.py:226) + surface `waiting_approval` prominently in orchestra_get_run; the
      skill instructs Claude to ask the user in chat and call continue/cancel; statusline
      shows a distinct `⏸ approval` state for waiting runs
- [ ] 3b: migrate mcp_server.py to the official `mcp` SDK (FastMCP) for bidirectional RPC;
      on `approval_requested` (for runs this server dispatched) send `elicitation/create`
      → approve/reject dialog in Claude Code; keep 3a as fallback
- **Acceptance criteria:** a critical-mode run pauses; the user approves from inside Claude
  Code (chat in 3a, dialog in 3b); the run resumes; events recorded.
- **Tests:** reuse `tests/test_approval_*` patterns; e2e pause→approve→resume via the
  service layer; 3b: SDK server handshake test.

### Phase 4 (optional): multi-run watch companion (~2-3h)
- **Depends on:** Phase 0
- [ ] Extract `_watch_plain` internals (cli.py:142-269) into a reusable module; add
      `orchestra watch --all`: iterate active runs from live_status.json, auto-attach new
      ones, one section per run
- [ ] Skill note: Claude may launch `orchestra watch --all` as a background task for a
      full-screen side view
- **Acceptance criteria:** two concurrent runs render in one watch session; a new run
  auto-appears.
- **Tests:** extend `tests/test_cli_live.py` with multi-run fixtures.

## Dependencies
- Phase 1 → Phase 0 (needs live_status.json). Phase 2 → Phase 0 (cwd fix); otherwise
  independent of Phase 1. Phase 3 → Phase 2. Phase 4 → Phase 0.
- Each phase is independently shippable (depth-first; blocks <4h each).

## Risks
- Statusline min refresh 1s + engine throttle 0.5s → worst-case ~2s lag → acceptable; document it.
- last_line garbage (agent CLIs emit ANSI/spinners) → strip ESC sequences + non-printables,
  truncate by display width; PiiScrubber already applied upstream.
- live_status adds I/O per lifecycle event + throttled stream ticks; bus handlers run in a
  4-worker pool → writer serializes internally (single lock) to avoid interleaved os.replace.
- settings.json installer could clobber an existing user statusline → detect, back up,
  require `--force`.
- config.py import-time CWD walk (CLAUDE.md gotcha) → statusline avoids config.py entirely;
  the Phase 0 env-based project-root override must not regress existing tests.
- `engine/mcp_registry.py` `_auto_wrap` drops unknown args silently → Phase 2 schema work
  should surface mismatches instead.
- cli.py historically had duplicated command groups → add the `statusline` group exactly
  once, near `watch`; prefer executing refactor-cli-split P2 first if that area churns.

## Progress Log
| Date | Phase | Notes |
|------|-------|-------|
| 2026-07-09 | — | Plan drafted from 3-agent research (Claude Code docs + 2 code sweeps); latency measured (84ms warm `python -m orchestra version`); bus/cwd/MCP bugs spot-check verified (bus.py:81, service.py:262, mcp_server.py:136, parallel/__init__.py:167/376) |
| 2026-07-13 | 3 | COMPLETE (3a: 27af016+58b9c10; 3b: 3430512..3ef3cfc). 3a — orchestra_continue_run + approval_state/waiting_approval in structured results + chat approval flow. 3b — orchestra/mcp_fastmcp.py: official-SDK low-level Server bridging REGISTRY (12 tools, policy-gate + PII-scrub parity), optional [mcp] extra; orchestra_request_approval does server-initiated elicitation (approve→continue_run, reject/decline/cancel→cancel_run, fail-closed on malformed/error — verified exhaustively by final review); .mcp.json switched to SDK server (hand-rolled kept as no-SDK fallback). SDK API validated by spikes; 15 async in-memory tests; branch sweep 146/146. NOTE: one implementer stalled mid-GREEN (Task 28) — controller finished it. Ride: build_server 95 lines (nested handlers); mcp.json.example still documents fallback. |
| 2026-07-10 | 2 | COMPLETE (6 commits, SDD: 2 tasks + 3 review-fix waves, final whole-phase review Approved). MCP server repaired: tools/list advertises 11 tools with real schemas (PEP563-safe derivation), -32601 for unknown methods (the old client-hang bug), robust stdio loop (malformed JSON / non-dict lines survived), registry single-param wrap fix (CRITICAL: get_run/cancel_run were broken — caught by final review, reproduced live, fixed + regression-tested at registry AND e2e level). Self-hosting .mcp.json + orchestra-dispatch skill (field names verified against models.py). Bonus: mcp_fastlane 0/6→6/6, benchmark_mcp orphan call fixed. 40 MCP tests green; branch sweep 117/117. |
| 2026-07-10 | 1 | COMPLETE (4 commits, SDD: 2 tasks + 2 review-fix waves, final whole-phase review Approved). orchestra/statusline.py (stdlib-only, never-crash, ~23ms; naive-ts coercion, control-char sanitization, active runs hidden after 600s-stale heartbeat) + `orchestra statusline render|install` (pure merge helpers, shlex-quoted interpreter, timestamped backup on --force). 60+ tests; live-verified: mid-run dual render shows both agent rows with elapsed/cost/last_line; stale runs auto-hide. |
| 2026-07-09 | 0 | COMPLETE (5 commits, SDD: 3 tasks + 2 review-fix waves, final whole-branch review Approved). live_status.py (merge-on-write for multi-process safety: own-runs-win, stale-foreign evicted at 600s), path = orchestra_root()/live_status.json, install() at 3 entry points, note_stream in both executors, worker cwd fix, bus logger fix. 35 new tests; covering sweep green (only 4 known pre-existing failures). Known limitations: foreign entries may lag one write cycle; file read happens under the writer's internal lock (fine on local FS); `_owned_run_ids` grows per process lifetime (strings only). |

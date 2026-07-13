---
name: orchestra-dispatch
description: Dispatch tasks to Orchestra's multi-agent engine via MCP and narrate progress. Use when the user asks to run a task "with orchestra", "on dual/critical agents", or wants parallel AI agents working on something.
---

# Orchestra Dispatch Skill

> **Setup note**: The MCP server uses the SDK bridge (`orchestra.mcp_fastmcp`).
> If it fails to start, run `pip install -e '.[mcp]'` inside the repo's `.venv`.

## Step 1 — Submit (non-blocking)

Call `orchestra_submit_run` with `mode` and `task`. Never use `orchestra_run` for
long tasks — it blocks until completion.

```
orchestra_submit_run(
  task="<user task text>",
  mode="dual"           # ask | dual | critical | auto
  # alias="cdx-fast"          # required only when mode="ask"
  # pause_after_round1=true   # optional: gate before round 2 (critical mode)
)
```

The response contains `response["job"]["job_id"]`. Save it. `run_id` is not available
at submit time — the worker fills it in later. Tell the user:
> "Submitted to Orchestra (job `<job_id>`). Polling for progress…"

If the MCP server is connected and the user has run `orchestra statusline install`,
the Claude Code statusline shows per-agent progress in real time.

## Step 2 — Poll until terminal

Prefer `orchestra_wait_for_run` (blocks server-side). Use `job_id` at first; switch to
`run_id` once the manifest exposes it. `orchestra_get_run(run_id=...)` is a manual alternative.

```
orchestra_wait_for_run(job_id="<job_id>", timeout_seconds=30)   # then run_id=... when known
```

Terminal statuses: `completed`, `failed`, `cancelled`, `waiting_approval`. Repeat until
terminal. Optionally launch `orchestra watch <run_id>` (background) for a live side pane.

## Step 3 — Handle `waiting_approval`

If `response["run"]["status"] == "waiting_approval"`, the run is paused at an approval
gate after round 1.

**Preferred — elicitation dialog:** call `orchestra_request_approval(run_id=<run_id>)`.
It triggers an in-editor popup showing the run summary and lets the user approve or reject
without leaving Claude Code; the tool maps approve → `continue_run` and
reject/decline/cancel → `cancel_run` automatically. Wait for its result.

**Fallback — chat-based** (clients without elicitation, or the hand-rolled server):

1. Summarize round-1 from `response["run"]`: `run["agents"][]` (alias, status, soft_failed,
   estimated_cost_usd) and `run["summary"]` if present.
2. Ask the user explicitly: "approve (continue round 2) or reject (cancel)?" Never auto-approve.
3. Approve → `orchestra_continue_run(run_id=<run_id>)` (warn: blocks until round 2 finishes).
4. Reject → `orchestra_cancel_run(run_id=<run_id>)`.

## Step 4 — Report results on completion

`orchestra_get_run` returns the manifest under the `run` key. Extract per-agent outcomes
(`run["agents"][]` → alias, status, soft_failed, estimated_cost_usd), `run["summary"]`, and
`run["total_cost_usd"]`. Present a compact recap, e.g.:

```
Orchestra run completed (dual mode)
  cdx-fast  → completed   $0.004
  gmn-pro   → completed   $0.011
  Total cost: $0.015
```

## Step 5 — Handle failures

Real `AgentStatus` values: `queued`, `started`, `retrying`, `completed`, `failed`, `cancelled`.
`soft_failed` is a **boolean** on each agent (`run["agents"][i]["soft_failed"]`), NOT a status.

If any agent has `soft_failed == true` or the run `status == "failed"`, fetch logs and surface
the relevant excerpts:

```
orchestra_get_logs(run_id="<id>")                  # or agent="cdx-fast" to filter
```

## Tool reference (actual schemas)

| Tool | Required params | Optional params |
|---|---|---|
| `orchestra_submit_run` | `task`, `mode` | `alias`, `pause_after_round1` |
| `orchestra_get_run` | `run_id` | — |
| `orchestra_wait_for_run` | — | `run_id`, `job_id`, `timeout_seconds`, `poll_interval_seconds` |
| `orchestra_list_runs` | — | `limit`, `mode`, `status` |
| `orchestra_get_logs` | `run_id` | `agent`, `normalized` |
| `orchestra_cancel_run` | `run_id` | — |
| `orchestra_continue_run` | `run_id` | — |
| `orchestra_request_approval` | `run_id` | — |
| `orchestra_run` | `task`, `mode` | `alias`, `previous_alias`, `pause_after_round1`, `intent_hint` |

Use `orchestra_run` only for short synchronous calls (e.g. `mode="ask"`) expecting a reply
under 30 s. `orchestra_request_approval` is SDK-server-only (`orchestra.mcp_fastmcp`, the
default in `.mcp.json`); the hand-rolled fallback does not support elicitation.

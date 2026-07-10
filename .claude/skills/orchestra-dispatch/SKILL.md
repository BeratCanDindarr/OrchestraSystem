---
name: orchestra-dispatch
description: Dispatch tasks to Orchestra's multi-agent engine via MCP and narrate progress. Use when the user asks to run a task "with orchestra", "on dual/critical agents", or wants parallel AI agents working on something.
---

# Orchestra Dispatch Skill

## Step 1 — Submit (non-blocking)

Call `orchestra_submit_run` with `mode` and `task`. Never use `orchestra_run` for
long tasks — it blocks until completion.

```
orchestra_submit_run(
  task="<user task text>",
  mode="dual"           # ask | dual | critical | auto
  # alias="cdx-fast"   # required only when mode="ask"
  # pause_after_round1=true  # optional: gate before round 2 (critical mode)
)
```

The response contains `response["job"]["job_id"]`. Save it. Note: `run_id` is not available at submit time — the worker fills it in later. Tell the user:
> "Submitted to Orchestra (job `<job_id>`). Polling for progress…"

If the MCP server is connected, the statusline at the bottom of Claude Code shows
per-agent progress in real time once the user has run `orchestra statusline install`.

## Step 2 — Poll until terminal

Poll with `orchestra_wait_for_run` (preferred — it blocks server-side) or call
`orchestra_get_run` manually every 15–30 s. Use `job_id` initially; switch to `run_id` once available.

```
# Preferred: server-side poll with short timeout (use job_id at submit time)
orchestra_wait_for_run(job_id="<job_id>", timeout_seconds=30)

# Once run_id is available (from manifest), use it instead
orchestra_wait_for_run(run_id="<run_id>", timeout_seconds=30)

# Manual poll alternative
orchestra_get_run(run_id="<run_id>")
```

Terminal statuses: `completed`, `failed`, `cancelled`, `waiting_approval`.

While waiting, optionally launch a background watcher:
```bash
orchestra watch <run_id>   # live per-agent stream in a side pane
```

Repeat polling until a terminal status is reached.

## Step 3 — Handle `waiting_approval`

If `orchestra_get_run` returns `status: waiting_approval`, the run has paused at a
human-approval gate. Inform the user:

> "The run is paused for your approval (gate after round 1). Resume it with:
> `orchestra continue <run_id>`"

Approval gate continuation is a CLI-only flow in the current phase. Do not attempt
to resume via MCP tools.

## Step 4 — Report results on completion

Extract from the `orchestra_get_run` manifest:

- **Per-agent outcomes** — `manifest.agents[]` → alias, status, estimated_cost_usd
- **Summary** — `manifest.summary`
- **Total cost** — `manifest.total_cost_usd`

Present a compact summary, e.g.:

```
Orchestra run completed (dual mode)
  cdx-fast  → passed   $0.004
  gmn-pro   → passed   $0.011
  Total cost: $0.015
```

## Step 5 — Handle failures

If any agent `status == "soft_failed"` or the run `status == "failed"`, call:

```
orchestra_get_logs(run_id="<id>")
# or filter to a specific agent:
orchestra_get_logs(run_id="<id>", agent="cdx-fast")
```

Surface the relevant excerpts to the user and explain what went wrong.

## Tool reference (actual schemas)

| Tool | Required params | Optional params |
|---|---|---|
| `orchestra_submit_run` | `task`, `mode` | `alias`, `pause_after_round1` |
| `orchestra_get_run` | `run_id` | — |
| `orchestra_wait_for_run` | — | `run_id`, `job_id`, `timeout_seconds`, `poll_interval_seconds` |
| `orchestra_list_runs` | — | `limit`, `mode`, `status` |
| `orchestra_get_logs` | `run_id` | `agent`, `normalized` |
| `orchestra_cancel_run` | `run_id` | — |
| `orchestra_run` | `task`, `mode` | `alias`, `previous_alias`, `pause_after_round1`, `intent_hint` |

Use `orchestra_run` only for short synchronous calls (e.g. `mode="ask"` single-agent
queries) where you expect a response in under 30 s.

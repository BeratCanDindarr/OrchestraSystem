---
name: orchestra-dispatch
description: Dispatch tasks to Orchestra's multi-agent engine via MCP and narrate progress. Use when the user asks to run a task "with orchestra", "on dual/critical agents", or wants parallel AI agents working on something.
---

# Orchestra Dispatch Skill

> **Setup note**: The MCP server now uses the SDK bridge (`orchestra.mcp_fastmcp`).
> If it fails to start, run `pip install -e '.[mcp]'` inside the repo's `.venv`.

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

If `response["run"]["status"] == "waiting_approval"`, the run is paused at an
approval gate after round 1.

### Preferred: elicitation dialog (Claude Code SDK server)

Call `orchestra_request_approval(run_id=<run_id>)`. This triggers an in-editor
popup that shows the run summary and lets the user approve or reject without
leaving Claude Code. The tool handles approve → `continue_run` and
reject/decline/cancel → `cancel_run` automatically.

```
orchestra_request_approval(run_id="<run_id>")
```

Wait for the tool result. It will report whether the run was resumed or cancelled.

### Fallback: chat-based approval (clients without elicitation support)

If the elicitation dialog is not available (e.g. older clients or the hand-rolled
MCP fallback server), handle approval manually:

1. **Summarize round-1 results** for the user. Extract from `response["run"]`:
   - `run["agents"][]` — each agent's alias, status, soft_failed, estimated_cost_usd
   - `run["summary"]` — overall round-1 summary (if present)

   Present a brief recap, e.g.:
   > "Round 1 is complete. Here's what the agents found: …"

2. **Ask the user** explicitly:
   > "Do you want to approve (continue round 2) or reject (cancel) this run?"
   Never auto-approve — always wait for the user's answer.

3. **Approve** → call `orchestra_continue_run(run_id=<run_id>)`.
   Warn the user first:
   > "Resuming — this will block until round 2 finishes (may take minutes)."
   Then call the tool and report the final result.

4. **Reject** → call `orchestra_cancel_run(run_id=<run_id>)`.

## Step 4 — Report results on completion

`orchestra_get_run` returns the manifest under the `run` key (`response["run"]`).
Extract from it:

- **Per-agent outcomes** — `response["run"]["agents"][]` → alias, status, soft_failed, estimated_cost_usd
- **Summary** — `response["run"]["summary"]`
- **Total cost** — `response["run"]["total_cost_usd"]`

Present a compact summary, e.g.:

```
Orchestra run completed (dual mode)
  cdx-fast  → completed   $0.004
  gmn-pro   → completed   $0.011
  Total cost: $0.015
```

## Step 5 — Handle failures

Real `AgentStatus` values: `queued`, `started`, `retrying`, `completed`, `failed`, `cancelled`.

`soft_failed` is a **boolean field** on each agent (`response["run"]["agents"][i]["soft_failed"] == true`),
NOT a status string. To detect soft failures, check each agent entry:

```python
for agent in response["run"]["agents"]:
    if agent.get("soft_failed"):
        # this agent soft-failed
```

If any agent has `soft_failed == true` or the run `status == "failed"`, call:

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
| `orchestra_continue_run` | `run_id` | — |
| `orchestra_request_approval` | `run_id` | — |
| `orchestra_run` | `task`, `mode` | `alias`, `previous_alias`, `pause_after_round1`, `intent_hint` |

Use `orchestra_run` only for short synchronous calls (e.g. `mode="ask"` single-agent queries) where you expect a response in under 30 s.

`orchestra_request_approval` is SDK-server-only: it requires the `mcp` SDK bridge
(`orchestra.mcp_fastmcp`, now the default in `.mcp.json`). The hand-rolled fallback
(`orchestra.mcp_server`) does not support elicitation — use the chat-based fallback instead.

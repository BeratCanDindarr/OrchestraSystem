---
name: orchestra-dispatch
description: Dispatch tasks to Orchestra's multi-agent engine via MCP and narrate progress. Use when the user asks to run a task "with orchestra", wants a second opinion, a decision argued from both sides, a code review of the current changes, or parallel AI agents working on something.
---

# Orchestra Dispatch Skill

> **Setup note**: The MCP server uses the SDK bridge (`orchestra.mcp_fastmcp`).
> If it fails to start, run `pip install -e '.[mcp]'` inside the repo's `.venv`.

## Step 0 — Pick the tool

| The user wants | Call |
|---|---|
| A quick answer from one model | `orchestra_run(mode="ask")`: leave `alias` out and Orchestra picks by the task (web research → `agy-argon`, else the task's tier); name one only when you need that model |
| Two independent answers, checked and merged | `orchestra_submit_run(mode="dual")` |
| A decision, or a second opinion that must not just agree | `orchestra_submit_run(mode="consensus")`: one agent argues for the proposal, one against, a third assesses it impartially, and the synthesis decides and says what would reverse it. Phrase the task as a proposal. |
| A review of code changes | `orchestra_review_changes()`: uncommitted changes by default; `base="main"` for a branch, `paths=[...]` to narrow, `focus="..."`. Blocking, one to four minutes. Findings come back as data and each was checked against the diff. |
| To continue an earlier answer | the same call plus `follow_up_to="<run_id>"`: the agents are shown that discussion, so the task can be a short follow-up. In ask mode the alias may be left out. |

Agents can read the project but cannot change files or run commands. In a Unity project with
`[unity] enabled = true` the Claude and Ollama agents can also *read* the open editor: scene
hierarchy, GameObjects and components, assets, prefabs, scripts and the console (through a
read-only proxy; every write is refused). Ask them about a NullReference or a scene's setup.
Aliases and which model does what: `docs/model-routing.md` (`cld-deep`, `cld-fast`, `kimi-deep`,
`kimi-code`, `oll-default`, `oll-glm`, ...). Add `fresh=true` to an ask to get a new answer
instead of a stored one.

## Step 1 — Submit (non-blocking)

Call `orchestra_submit_run` with `mode` and `task`. Never use `orchestra_run` for
long tasks — it blocks until completion.

```
orchestra_submit_run(
  task="<user task text>",
  mode="dual"           # ask | dual | consensus | auto
  # alias="kimi-deep"         # optional: without it Orchestra picks the agent for the task
  # follow_up_to="<run_id>"   # continue an earlier run
)
```

The response contains `response["job"]["job_id"]` and `response["job"]["run_id"]`: the run
exists from submit on (status `queued`), so it can be waited on or cancelled right away. Tell the user:
> "Submitted to Orchestra (run `<run_id>`). Polling for progress…"

If the MCP server is connected and the user has run `orchestra statusline install`,
the Claude Code statusline shows per-agent progress in real time.

## Step 2 — Poll until terminal

Prefer `orchestra_wait_for_run` (blocks server-side) with the `run_id` (or the `job_id`). One wait (default 240 s) usually covers a whole dual run.

```
orchestra_wait_for_run(run_id="<run_id>")
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

The wait/run tools already return what you need under `run`: `summary` (the answer or the
synthesis, in full), `agents[]` (alias, model, status, soft_failed, estimated_cost_usd),
`latest_review_status` (the verifier's verdict), `total_cost_usd`, and `run_id` (keep it: it
is what `follow_up_to` takes). `orchestra_get_run(run_id, detail=true)` returns the whole
manifest and every event (large); call it only when you need more. Lead with the answer, then a one-line recap, e.g.:

```
dual, verifier passed: cld-deep (claude/opus) + kimi-deep (kimi/k3), $0.015, run 9a94ca39
```

`fallbacks` is present when an alias could not answer (quota, timeout, logged out) and another
model answered in its place. Say so: a `dual` or a review whose two agents ended up the same
model is one opinion, not two.

## Step 5 — Handle failures

Real `AgentStatus` values: `queued`, `started`, `retrying`, `completed`, `failed`, `cancelled`.
`soft_failed` is a **boolean** on each agent (`run["agents"][i]["soft_failed"]`), NOT a status.

If any agent has `soft_failed == true` or the run `status == "failed"`, fetch logs and surface
the relevant excerpts:

```
orchestra_get_logs(run_id="<id>")   # last 2000 chars of each failed agent; agent="kimi-deep" for one, full=true for all
```

## Tool reference (actual schemas)

| Tool | Required params | Optional params |
|---|---|---|
| `orchestra_submit_run` | `task`, `mode` | `alias`, `follow_up_to`, `fresh` |
| `orchestra_review_changes` | — | `base`, `paths`, `focus`, `diff` |
| `orchestra_get_run` | `run_id` | `detail` |
| `orchestra_wait_for_run` | — | `run_id`, `job_id`, `timeout_seconds`, `poll_interval_seconds` |
| `orchestra_list_runs` | — | `limit`, `mode`, `status`, `include_steps` |
| `orchestra_memory_search` | `query` | `limit`, `full` |
| `orchestra_get_logs` | `run_id` | `agent`, `full` |
| `orchestra_cancel_run` | `run_id` | — |
| `orchestra_continue_run` | `run_id` | — |
| `orchestra_request_approval` | `run_id` | — |
| `orchestra_run` | `task`, `mode` | `alias`, `follow_up_to`, `fresh`, `previous_alias`, `intent_hint` |

Use `orchestra_run` only for short synchronous calls (e.g. `mode="ask"`) expecting a reply
under 30 s. `orchestra_request_approval` is SDK-server-only (`orchestra.mcp_fastmcp`, the
default in `.mcp.json`); the hand-rolled fallback does not support elicitation.

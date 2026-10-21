# Orchestra

Orchestra sends one task to several AI agents, checks what they say against each other, and
gives back one answer. It runs on your machine and drives the agent CLIs you are already
logged in to (Claude Code, Kimi, Codex, Antigravity for Gemini) and Ollama models. It is built
to sit next to a coding agent such as Claude Code: the coding agent edits, Orchestra gives
second opinions, argued decisions, code reviews and research.

Every alias is an agent that can read the project. An agent gets the task and a pointer to
`CLAUDE.md`, `AGENTS.md` and `docs/`, and reads what it needs; files the task mentions are not
pasted in. Agents do not change files unless you route a task to an alias you have allowed to
(`cld-write`).

## What it does

| You want | Use | What happens |
|---|---|---|
| A quick answer from one model | `ask` | One agent answers. |
| Two independent answers | `dual` | Two agents of different model families answer, a verifier checks them, a synthesis merges them and keeps the disagreement. |
| A decision | `consensus` | One agent argues for the proposal, one against, a third assesses it without seeing either. The synthesis decides and says what would reverse the decision. |
| A code review | `review` | Two agents review the diff. A finding counts only if the line it quotes is really in the diff; findings both report are merged. |
| A larger task broken up | `planned`, `supervise` | A planner splits the task into steps or subtasks, agents work through them, the results are combined. |
| To continue an earlier answer | `--follow <run_id>` | The agents are shown that discussion, so the new task can be a short follow-up. |

Every run is recorded under `.orchestra/runs/` (task, each agent's answer, events, cost) and
can be listed, inspected, compared and re-synthesised later.

## Requirements

- Python 3.11 or newer
- At least one agent: the `claude`, `kimi`, `codex` or Antigravity CLI (logged in), or Ollama
  (its models run through the `claude` CLI, so that has to be installed too)
- Ollama with an embedding model if you want semantic memory

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"        # add ",mcp" for the MCP server: ".[dev,mcp]"
mkdir -p .orchestra && cp config.toml.example .orchestra/config.toml
.venv/bin/orchestra doctor               # which providers can run, and whether every role has one
```

Orchestra finds `.orchestra/config.toml` by walking up from the directory you run it in, and
keeps its runs and databases beside that file. To use one configuration from another project,
link it: `ln -s /path/to/OrchestraSystem/.orchestra/config.toml yourproject/.orchestra/config.toml`.

## Use it from Claude Code (MCP)

```bash
claude mcp add orchestra --scope local -- /path/to/OrchestraSystem/.venv/bin/python -m orchestra.mcp_fastmcp
orchestra statusline install             # optional: show running agents in the status line
```

Restart Claude Code after changing Orchestra's code: the MCP server is a long-lived process.

| Tool | For |
|---|---|
| `orchestra_run` | Run a task and wait (`mode`: `ask`, `dual`, `consensus`, `auto`). |
| `orchestra_submit_run`, `orchestra_wait_for_run` | The same in the background: submit, then wait for the job. |
| `orchestra_review_changes` | Code review of uncommitted changes, or of `base`, `paths`, a given `diff`. |
| `orchestra_get_run` | A run's summary; `detail: true` for the full manifest and events. |
| `orchestra_get_logs` | The tail of each failed agent's log; `full: true` for every agent. |
| `orchestra_list_runs` | Recent runs. |
| `orchestra_cancel_run`, `orchestra_continue_run`, `orchestra_request_approval` | Stop a run; approve one that is waiting. |
| `orchestra_memory_search` | Search the notes that are given to agents as background. |

Run tools take `follow_up_to: <run_id>` to continue an earlier run and `fresh: true` to ask a
model again instead of returning a stored answer. They return the answer, the agents and
models that gave it, the verifier's verdict and the cost, in a compact payload;
`orchestra_get_run` returns the rest.

## Use it from the terminal

```bash
orchestra ask kimi-deep "Why would a pooled object keep stale state?"
orchestra run dual "Compare tweens and Animator controllers for HUD motion"
orchestra run consensus "Proposal: load every level through Addressables"
orchestra run auto "..."                 # the router picks the mode and the agent
orchestra ask kimi-deep "And on low-end devices?" --follow <run_id>

orchestra review                         # uncommitted changes
orchestra review --base main Assets/Scripts --focus "thread safety"

orchestra ps                             # recent runs (--all also lists their internal steps)
orchestra logs <run_id>                  # each agent's answer
orchestra timeline <run_id>              # who ran when
orchestra diff <run_a> <run_b>           # compare two runs
orchestra replay <run_id> -s oll-flash   # synthesise a finished run again with another model
orchestra resume <run_id> [--reject]     # approve or reject a run waiting for approval
orchestra cancel <run_id>
orchestra forget <run_id>...             # erase runs for good: folder, database rows, steps, memory row

orchestra eval "task"                    # run ask, dual and consensus on a task and judge them
orchestra supervise "big task"           # decompose, dispatch, synthesise
orchestra memory add "The board core is pure C#."
orchestra report cost                    # spend per model
orchestra --help                         # everything else
```

## Agents and routing

An alias names a model: `cld-deep` and `cld-fast` (Claude), `kimi-deep` and `kimi-code` (Kimi
CLI), `cdx-*` (Codex), `agy-*` (Gemini through the Antigravity CLI; the older `gmn-*` names map
there too) and `oll-*` (Ollama). Ollama models run as agents through the Claude CLI pointed at
Ollama's Anthropic API, with read-only tools; `model@backend` addresses a model on a named Ollama
backend (`orchestra backends`, `orchestra models`). `orchestra aliases` lists them with a mark
for whether each can run right now.

Which alias plays which role (implementer, researcher, reviewer, verifier, synthesiser, the
neutral agent of consensus mode) is set in `.orchestra/config.toml`. The choices in this
repository were measured, not guessed: `tools/model_bench/bench.py` runs a set of tasks on every
model, and `docs/model-routing.md` records the results and the routing that follows from them.
When a new kind of work comes up, benchmark it first.

If an alias's provider cannot run (not installed, switched off, logged out), the next alias in
its fallback chain answers and the run says so.

## What keeps the answers honest

- **Different families.** The two agents of a `dual` run come from different model providers.
- **Verifier.** A third model checks the answers for invented paths, contradictions and unsafe
  suggestions before anything is merged.
- **Stances.** In `consensus` mode disagreement is assigned, not hoped for, and honesty comes
  before the stance.
- **Checked findings.** A code-review finding whose quoted line is not in the diff is dropped.
- **Approval gate.** A task that mentions a privileged action (`git push`, `deploy`, ...; see
  `[autonomy_policy]`) stops and waits for you.
- **Stored answers.** A single-agent answer is reused only for the same prompt to the same
  alias; follow-ups are never answered from a store.
- **Memory.** Only notes you added are shown to agents, as background, never as tasks.
- **Stated confidence.** Each agent ends its answer with how sure it is; the synthesis weighs
  answers by that, and is told when an agent gave none.
- **Honest labels.** When an alias cannot answer and its fallback does, the run and a review's
  findings name the model that really answered.

Costs: a Claude call is recorded at what the CLI reports (API prices; a subscription is not
billed that way), an Ollama call from its real token counts, everything else from estimates.

## Configuration

`config.toml.example` documents every section. The ones you will touch first:

| Section | Sets |
|---|---|
| `[personas]`, `[tiers.*]` | Which alias plays which role, and the agents of each weight class. |
| `[providers.claude]`, `[providers.claude_write]` | What Claude agents may do (read-only profile, tools, permissions). |
| `[providers.<name>] enabled = false` | Switch a provider off. |
| `[providers.ollama.think_by_model]`, `[timeouts]` | How long each model may think and run. |
| `[consensus]`, `[follow_up]`, `[review_changes]`, `[planned]` | The neutral agent; how much history a follow-up sees; reviewers and limits; a planned run's time budget. |
| `[autonomy_policy]` | What stops a run for approval. |
| `[fts_cache]` | The store answers are reused from. |

## Project layout

```
orchestra/
  cli.py, service.py        entry points: terminal, and the functions MCP calls
  mcp_server.py             the MCP tool registry (served by mcp_fastmcp.py)
  engine/runners/           ask, dual (and consensus), planned, resume
  engine/                   code review, follow-ups, stances, synthesis, verification, policies
  providers/                Claude, Kimi, Codex, Antigravity, Ollama; aliases and fallbacks
  router/                   which mode and agent a task goes to
  storage/                  runs, events, memory, answer cache (SQLite)
  statusline.py             the Claude Code status line
  monitoring/               cost and analytics
tools/model_bench/          benchmark that decides the routing
docs/                       model routing, architecture, plans
tests/                      pytest suite
```

## Development

```bash
.venv/bin/pytest -m "not slow"           # the fast suite
.venv/bin/pytest --cov=orchestra         # with coverage
```

`CLAUDE.md` holds the conventions for working on the code.

## License

MIT

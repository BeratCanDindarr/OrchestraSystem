# Orchestra

**High-Performance Multi-Agent Orchestration System**

Orchestra is a powerful, production-grade engine that dispatches complex tasks to multiple AI agents concurrently. It features an event-driven automation system, automated playbooks, isolated git environments, and comprehensive observability dashboards.

## 🚀 Key Maestro Features

Orchestra has been enhanced with advanced capabilities to bridge the gap between simple chat and autonomous execution:

*   **Playbook Automation:** Execute Markdown-based task lists (`- [ ]`) sequentially. Orchestra maintains state directly in the file, checkmarking completed tasks and injecting error context upon failure.
*   **Orchestra Cue (Event-Driven):** An autonomous background watcher that triggers agent workflows based on file changes, cron-like schedules, GitHub activity, or manual triggers.
*   **Git Worktree Isolation:** Run agents in dedicated, ephemeral git branches. Prevent working-tree contamination and automatically create GitHub Pull Requests upon task completion.
*   **Surgical Context:** Drastically reduce token usage and cost by resetting context per sub-task using tiered injection strategies (`full`, `surgical`, `minimal`).
*   **Dynamic Moderator & Debate:** A multi-role consensus loop where specialized agents (Architect, QA, Lead) debate solutions until the Moderator reaches a consensus.
*   **Session Discovery:** Scan, search, and resume legacy provider sessions (e.g., Claude Code states) as native Orchestra runs.

## 📊 Dashboards & Analytics

*   **Web Dashboard:** A lightweight HTML interface accessible over the network to track your agent fleet's performance.
*   **Usage Analytics:** Generate cost/token heatmaps and CSV exports to profile your usage over daily, weekly, or monthly periods.

## 📦 Requirements

- Python 3.11+
- At least one AI CLI: `codex`, `gemini-cli`, or `claude-code`.
- GitHub CLI (`gh`) for autonomous PR creation.
- Python packages: `watchdog`, `pyyaml`, `apscheduler`, `rich`.

## 🛠️ Installation

```bash
# 1. Clone or copy into your project
cp -r orchestra/ .orchestra/ /your/project/

# 2. Install dependencies
pip install -r orchestra/requirements.txt

# 3. Configure
cp .orchestra/config.toml.example .orchestra/config.toml
```

## ⌨️ CLI Usage

### Core Runners
- **Single Agent:** `python3 -m orchestra ask cdx-deep "Refactor this"`
- **Parallel Mode:** `python3 -m orchestra run dual "Design API"`
- **Critical Loop:** `python3 -m orchestra run critical "Verify auth logic"`

### Advanced Automation
- **Playbook:** `python3 -m orchestra playbook run task_list.md`
- **Cue Watcher:** `python3 -m orchestra cue start`
- **Worktree Run:** `python3 -m orchestra run dual "task" --worktree fix-bug --pr`
- **Debate:** `python3 -m orchestra debate start "Should we use GraphQL?"`

### Observability
- **Stats:** `python3 -m orchestra report stats --format heatmap`
- **Sessions:** `python3 -m orchestra sessions list`

## 🧩 MCP Server (Claude Code Integration)

Orchestra exposes a rich set of tools to Claude Code via the Model Context Protocol (MCP).

Register it for a project with:

```
claude mcp add orchestra --scope local -- /path/to/OrchestraSystem/.venv/bin/python -m orchestra.mcp_fastmcp
```

Tools:
*   Runs: `orchestra_run` (blocking), `orchestra_submit_run` + `orchestra_wait_for_run` (background),
    `orchestra_get_run`, `orchestra_list_runs`, `orchestra_get_logs`, `orchestra_cancel_run`,
    `orchestra_continue_run`, `orchestra_request_approval`. Pass `fresh: true` to ask a model again
    instead of getting a stored answer to the same task.
*   Playbooks and cues: `orchestra_playbook_status`, `orchestra_playbook_run`, `orchestra_cue_list`,
    `orchestra_cue_trigger`.
*   Sessions: `orchestra_sessions_list`, `orchestra_sessions_search`.
*   Config and memory: `orchestra_config_get`, `orchestra_config_set`, `orchestra_memory_add`,
    `orchestra_memory_search`.

Debate, supervise and planned graphs are CLI commands only (`orchestra debate start`, `orchestra supervise`).

Providers: Claude CLI (`cld-*`), Kimi CLI (`kimi-*`), Ollama cloud models (`oll-*`), Codex CLI (`cdx-*`);
any of them can be switched off with `[providers.<name>] enabled = false`. Which model does which kind
of work, and the measurements behind it, is in `docs/model-routing.md`.

## 🏗️ Project Structure

```
orchestra/           # Core Engine
  engine/            # Playbook, Cue, Worktree, Moderator, Context
  providers/         # Azure, OpenAI, Claude, Gemini, DeepSeek, Perplexity
  monitoring/        # Analytics, Web UI
  storage/           # SQLite EventLog, FTS5 Cache, Blackboard
  cli.py             # Typer-based entry point
  mcp_server.py      # Standardized MCP integration
.orchestra/          # Runtime Configuration & Data
  config.toml        # Main configuration
  cue.yaml           # Automation rules
```

## ⚖️ License

MIT

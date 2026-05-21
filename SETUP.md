# Orchestra — Setup Guide

Multi-agent orchestration system. Runs Codex, Gemini, Claude, DeepSeek, and Perplexity CLIs in parallel. Includes an artifact store (SQLite + JSONL), automated playbooks, event-driven triggers (Cue), and an MCP server.

## Requirements

- Python 3.11+
- `codex` CLI installed (`npm install -g @openai/codex`)
- `gemini` CLI installed (`npm install -g @google/gemini-cli`)
- `watchdog`, `pyyaml`, `apscheduler` (Python packages)

## Installation

1. **Copy this folder into your project root:**
   ```bash
   cp -r orchestra/      /your/project/orchestra/
   cp -r .orchestra/     /your/project/.orchestra/
   ```

2. **Install dependencies:**
   ```bash
   pip install -r orchestra/requirements.txt
   ```

3. **Configure settings:**
   Edit `.orchestra/config.toml` (timeouts, token budgets, model aliases).

4. **MCP Setup:**
   Place the `.mcp.json` file in the project root, replacing `/YOUR/PROJECT/ROOT` with the actual absolute path.

## Usage

### CLI Commands

- **Single Agent:** `python3 -m orchestra ask cdx-deep "task"`
- **Parallel Run:** `python3 -m orchestra run dual "task"`
- **Playbook Automation:** `python3 -m orchestra playbook run path/to/plan.md`
- **Event Watcher:** `python3 -m orchestra cue start`
- **Dynamic Debate:** `python3 -m orchestra debate start "topic"`

### Isolated Environments (Worktrees)

Run tasks in a dedicated git branch without affecting your main working tree:
```bash
python3 -m orchestra run dual "refactor auth" --worktree fix-auth --pr
```

## Event-Driven Automation (Cue)

Define rules in `.orchestra/cue.yaml`:
```yaml
rules:
  lint_on_save:
    on: file.changed
    match:
      CUE_FILE_PATH: "src/.*\.py"
    do: run_agent
    with:
      alias: cdx-fast
      task: "Lint this file: {{CUE_FILE_PATH}}"
```

## Monitoring & Analytics

- **Real-time TUI:** `python3 -m orchestra dashboard`
- **Web Dashboard:** Start the daemon (`python3 -m orchestra daemon`) and visit `http://localhost:8080`.
- **Usage Stats:** `python3 -m orchestra report stats --period weekly --format heatmap`

## Testing

Run the test suite to ensure stability:
```bash
pytest tests/
```

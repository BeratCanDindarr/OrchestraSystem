# Orchestra

Multi-agent orchestration system that runs Codex, Gemini, and Claude CLI tools in parallel, synthesizes their outputs, and exposes results via a CLI, TUI, WebSocket daemon, and MCP server.

## Overview

Orchestra dispatches a task to one or more AI agents concurrently, collects their outputs into a structured artifact store (SQLite + JSONL), cross-reviews them, and optionally synthesizes a final answer. It is designed to be dropped into any project directory.

```
orchestra run dual "analyze this architecture"
  -> cdx-deep (Codex)  -+
  -> gmn-pro  (Gemini) -+-> cross-review -> synthesis -> stored run
```

## Requirements

- Python 3.11+
- At least one of the following AI CLIs installed:
  - `codex` — `npm install -g @openai/codex`
  - `gemini` — `npm install -g @google/gemini-cli`
  - `claude` — `npm install -g @anthropic-ai/claude-code`
- Optional: [Ollama](https://ollama.com) for local model support

## Installation

```bash
# 1. Copy into your project root
cp -r orchestra/      /your/project/orchestra/
cp -r .orchestra/     /your/project/.orchestra/

# 2. Copy and edit config
cp .orchestra/config.toml.example .orchestra/config.toml

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set up MCP for Claude Code
cp mcp.json.example .mcp.json
# Edit .mcp.json and replace /path/to/your/project with the actual absolute path
```

## Usage

```bash
# Single agent
python3 -m orchestra ask cdx-deep "what is wrong with this code?"

# Parallel (Codex + Gemini)
python3 -m orchestra run dual "design a caching layer"

# Full loop (parallel + cross-review + synthesis)
python3 -m orchestra run critical "decide: approach A or B?"

# Auto-route based on task complexity
python3 -m orchestra run auto "summarize this file"

# List recent runs
python3 -m orchestra ps

# Print agent logs for a run
python3 -m orchestra logs <run_id>

# Live TUI monitor
python3 -m orchestra watch <run_id>
```

## Execution Modes

| Mode | Description |
|------|-------------|
| `ask <alias>` | Single agent, fast |
| `dual` | cdx-deep + gmn-pro in parallel |
| `critical` | dual + cross-review + synthesis |
| `auto` | Router selects mode by task keywords |
| `planned` | Graph-based multi-step execution |

## Agent Aliases

| Alias | Provider | Effort/Model |
|-------|----------|--------------|
| `cdx-fast` | Codex | low |
| `cdx-deep` | Codex | xhigh |
| `gmn-fast` | Gemini | flash |
| `gmn-pro` | Gemini | pro |
| `cld-fast` | Claude | sonnet |
| `cld-deep` | Claude | opus |
| `oll-coder` | Ollama | qwen2.5-coder:7b |
| `oll-analyst` | Ollama | deepseek-r1:8b |

## MCP Server (Claude Code integration)

Orchestra exposes an MCP server so Claude Code can dispatch tasks to it as tool calls.

Copy `mcp.json.example` to `.mcp.json`, set `cwd` and `PYTHONPATH` to your project root, then restart Claude Code. Available MCP tools: `orchestra_submit_run`, `orchestra_get_job`, `orchestra_list_runs`, `orchestra_get_logs`, `orchestra_get_run`, `orchestra_cancel`, `orchestra_continue`.

## WebSocket Daemon (editor integration)

```bash
python3 -m orchestra daemon --host 127.0.0.1 --port 8765
```

## Configuration

Copy `.orchestra/config.toml.example` to `.orchestra/config.toml` and edit as needed.

Key settings:

```toml
[orchestra]
timeout = 180            # seconds per agent run
auto_synthesize = false  # auto-run synthesis after dual mode

[limits]
max_turns = 10
max_budget_usd = 5.0

[redaction]
# Patterns masked in stored artifacts
patterns = [
    "Bearer [A-Za-z0-9\\-._~+/]+=*",
    "sk-[A-Za-z0-9]{32,}",
]
```

## Project Structure

```
orchestra/           # Python package
  cli.py             # Typer CLI entry point
  service.py         # Headless service API (used by MCP)
  mcp_server.py      # stdio MCP server
  server.py          # WebSocket daemon
  engine/            # Run execution: parallel, runner, reviewer, synthesizer
  providers/         # Codex, Gemini, Claude, Ollama adapters
  router/            # Task classifier and mode router
  storage/           # SQLite + JSONL artifact store
  redaction/         # Output redaction engine
  tui/               # Textual TUI for live monitoring
  prompts/           # Prompt templates
.orchestra/          # Runtime data directory (gitignored except config)
  config.toml        # Local config (copy from config.toml.example)
  runs/              # Per-run artifacts (gitignored)
tests/               # Test suite
```

## License

MIT

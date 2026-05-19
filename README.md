# Orchestra

Multi-agent orchestration system that runs Codex, Gemini, and Claude CLI tools in parallel, synthesizes their outputs, and exposes results via a CLI, TUI, WebSocket daemon, MCP server, and federation HTTP API.

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

### Running tasks

```bash
# Single agent
python3 -m orchestra ask cdx-deep "what is wrong with this code?"

# Parallel (Codex + Gemini)
python3 -m orchestra run dual "design a caching layer"

# Full loop (parallel + cross-review + synthesis)
python3 -m orchestra run critical "decide: approach A or B?"

# Auto-route based on task complexity
python3 -m orchestra run auto "summarize this file"

# Large task — auto-decomposition into planned graph
python3 -m orchestra run planned "refactor the entire auth module"

# List recent runs
python3 -m orchestra ps

# Print agent logs for a run
python3 -m orchestra logs <run_id>

# Live TUI monitor
python3 -m orchestra watch <run_id>
```

### Skills

```bash
# Install a skill from the marketplace
python3 -m orchestra skill install rust-error-patterns

# Run a skill directly
python3 -m orchestra skill run rust-error-patterns

# List installed skills
python3 -m orchestra skill list

# Uninstall a user-tier skill
python3 -m orchestra skill uninstall <name>
```

### Federation (cross-instance memory sharing)

```bash
# Push a run summary to federation outbox
python3 -m orchestra federation push <run_id>

# Search remote peers for similar summaries
python3 -m orchestra federation search "caching strategy"

# Sync outbox with all configured peers
python3 -m orchestra federation sync

# Rotate auth token
python3 -m orchestra federation rotate-token --ttl 720

# Start federation HTTP server
python3 -m orchestra federation serve --host 127.0.0.1 --port 8766
```

### Consensus & Quality

```bash
# Export consensus graph as inline SVG HTML
python3 -m orchestra graph --format html --out consensus.html

# View synthesis quality report
python3 -m orchestra report synthesis
```

### Time Machine & Branching

```bash
# Branch a run from a specific checkpoint
python3 -m orchestra branch <run_id> --from-version 3
```

## Execution Modes

| Mode | Description |
|------|-------------|
| `ask <alias>` | Single agent, fast |
| `dual` | cdx-deep + gmn-pro in parallel |
| `critical` | dual + cross-review + synthesis + approval gate |
| `auto` | Router selects mode by task keywords |
| `planned` | Graph-based multi-step execution |
| `extreme` | Auto-decomposition for very large tasks |

## Agent Aliases

| Alias | Provider | Effort/Model |
|-------|----------|--------------|
| `cdx-fast` | Codex | low |
| `cdx-deep` | Codex | xhigh |
| `gmn-fast` | Gemini | flash |
| `gmn-pro` | Gemini | pro |
| `cld-fast` | Claude | sonnet |
| `cld-deep` | Claude | opus |
| `oll-coder` | Ollama | coder |
| `oll-analyst` | Ollama | analyst |
| `oll-fast` | Ollama | fast |
| `oll-deep` | Ollama | llama3 |
| `oll-mini` | Ollama | mini |
| `oll-kimi` | Ollama | kimi-k2.6:cloud |
| `oll-minimax` | Ollama | minimax-m2.7:cloud |
| `oll-dev` | Ollama | minimax-m2.7:cloud |
| `oll-embed` | Ollama | embed |

## Built-in Safety & Quality Layers

- **Approval Gates** — Policy-based (`policy.yaml`) or cost/confidence/mode-based automatic escalation to human approval.
- **Context Compression** — 3-tier prompt degradation with AST-based Python signature extraction to stay within token budgets.
- **PII Scrubber** — Redacts API keys, emails, JWTs, phone numbers, SSNs, credit cards, OAuth tokens, AWS keys, GitHub/Slack/Discord tokens, private keys, and DB connection strings before caching or logging.
- **Workspace Guard** — Blocks edits to critical files during execution.
- **Red-Team / Adversarial** — `ensure_diversity`, `inject_red_team`, and `search_counter_example` for dissent injection and counter-example search.
- **Synthesis Quality Measurement** — Structural scoring (sections, length, divergence penalty) persisted to SQLite.
- **Consensus Graph** — Interactive Textual TUI + inline SVG export for multi-agent reasoning visualization.
- **Replay System** — Deterministic replay with SHA256-seeded RNG and append-only JSONL logs.
- **FTS5 Cache** — Semantic prompt/response caching with BM25 similarity search.
- **Idempotency** — `@idempotent` decorator with seq-ordered event log replay.
- **WASM Sandbox** — Stub for future WebAssembly tool execution (`orchestra.toolmaker.wasm_sandbox`).

## MCP Server (Claude Code integration)

Orchestra exposes an MCP server so Claude Code can dispatch tasks to it as tool calls.

Copy `mcp.json.example` to `.mcp.json`, set `cwd` and `PYTHONPATH` to your project root, then restart Claude Code. Available MCP tools include `orchestra_submit_run`, `orchestra_get_job`, `orchestra_wait_for_run`, `orchestra_list_runs`, `orchestra_get_logs`, `orchestra_get_run`, `orchestra_cancel`, `orchestra_continue`.

`orchestra_wait_for_run` is a short-wait convenience wrapper. For long-running work, prefer `orchestra_submit_run` and poll with `orchestra_get_job` or `orchestra_get_run` instead of holding one blocking MCP call open.

The MCP surface also includes operational tools for config management, semantic memory, routing inspection, checkpoints, trace/timeline inspection, run replay, batch execution, evaluation summaries, and backtracking.

### Recommended Flows

- Short synchronous wait: use `orchestra_run` or `orchestra_wait_for_run` when the task is expected to finish quickly.
- Long-running task: use `orchestra_submit_run`, then poll with `orchestra_get_job` until the job or run reaches a terminal state.
- Approval flow: if `orchestra_get_job` or `orchestra_get_run` reports `waiting_approval`, call `orchestra_continue`.
- Manual stop: call `orchestra_cancel` only when the user wants to stop an active run.

## WebSocket Daemon (editor integration)

```bash
python3 -m orchestra daemon --host 127.0.0.1 --port 8765
```

## Federation HTTP Server (cross-instance)

```bash
python3 -m orchestra federation serve --host 127.0.0.1 --port 8766
```

Mounts endpoints:
- `GET /federation/health` — health check
- `POST /federation/push` — receive summaries from peers
- `GET /federation/query` — keyword search over inbox
- `GET /federation/peers` — list configured peers
- `GET /federation/status` — outbox/inbox stats

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

[federation]
enabled = false
peer_urls = []
auth_token = ""
# auth_token_ttl_hours = 720
# auth_token_issued_at = 0.0
```

## Project Structure

```
orchestra/           # Python package
  cli.py             # Typer CLI entry point
  service.py         # Headless service API (used by MCP)
  mcp_server.py      # stdio MCP server
  server.py          # WebSocket daemon
  engine/            # Run execution: parallel, runner, reviewer, synthesizer
    skills/          # Tiered skill system with marketplace + reputation
    federation*.py   # CRDT + summarizer for cross-instance memory
    replay.py        # Deterministic replay recorder/runner
    policy_engine.py # Declarative policy gates
    pii_scrubber.py  # Multi-pattern PII redaction
    context_compression.py # Token-aware prompt compression
    adversarial.py   # Red-team dissent injection
    consensus*.py    # Consensus graph builder + visualization
  providers/         # Codex, Gemini, Claude, Ollama adapters
  router/            # Task classifier and mode router
  storage/           # SQLite + JSONL artifact store + migrations
  redaction/         # Output redaction engine
  tui/               # Textual TUI for live monitoring + consensus screen
  prompts/           # Prompt templates
  toolmaker/         # Dynamic tool sandbox (Python + WASM stub)
  federation_server/ # aiohttp peer-to-peer HTTP client + server
  events/            # Async pub-sub event bus
.orchestra/          # Runtime data directory (gitignored except config)
  config.toml        # Local config (copy from config.toml.example)
  runs/              # Per-run artifacts (gitignored)
tests/               # Test suite (1000+)
```

## License

MIT

# Plan: Slim what goes to agents and back to Claude Code

## Status
In Progress

## Goal
Orchestra is used only by Claude Code (MCP + statusline): cut the bytes sent to agents and
returned to Claude, run only agents that can read the project, and delete what nothing uses.

## Measured (dual run 35b5ca4d, 2026-10-20)
- Round-1 agent prompt ~14 KB each; synthesis prompt 46.8 KB (16.3 KB of it inlined file headers).
- `orchestra_list_runs` ~66 KB per call, `orchestra_get_run` 22-62 KB, `orchestra_get_logs` ~27 KB.
- MCP tool list ~2.2K tokens (19 tools, 9 unused by the orchestra-dispatch skill).
- Claude CLI pointed at Ollama (`ANTHROPIC_BASE_URL=http://localhost:11434`, Ollama 0.40.3)
  reads files with Read/Grep/Glob: Ollama models can be agents without the gateway.

## Phases

### Phase 1: What Claude gets back
- [ ] list_runs: short task, small default limit
- [ ] get_run: compact by default, `detail=true` for everything
- [ ] get_logs: tail of failed agents by default, `full=true` for everything
- [ ] review_changes: findings sent once
- [ ] compact JSON, no repeated `result` block, no Provenance table in the answer
- **Acceptance criteria**: fast suite passes; the real run 35b5ca4d payloads shrink

### Phase 2: What agents get
- [ ] no [CODE CONTEXT] file headers (agents read files themselves)
- [ ] follow-up history only to round-1 agents, not to verifier/synthesiser
- [ ] verifier output not a synthesis candidate
- [ ] memory notes only to round-1 agents
- [ ] Unity rule only on Unity/C# tasks
- **Acceptance criteria**: fast suite passes; synthesis prompt carries only the candidates

### Phase 3: Only agents
- [ ] Ollama aliases run through the Claude CLI against Ollama's Anthropic API, read-only tools
- [ ] remove text-only API providers (openai_sdk, deepseek, perplexity, azure_openai)
- **Depends on**: Phase 2
- **Acceptance criteria**: an oll-* ask reads a file; fast suite passes

### Phase 4: Delete what nothing uses
- [ ] web dashboard, WebSocket server, chat, gateway + compat APIs, claude_launcher
- [ ] federation_server, batch_runner, mcp_client, toolmaker/speculation storage, ForestBuildingSystem
- [ ] the 9 MCP tools the skill never calls (cue, playbook, sessions, config, memory_add)
- [ ] dependencies only they used; CLAUDE.md / docs updated
- **Kept**: MCP server, statusline + live_status, remote_nodes for alias@node
- **Acceptance criteria**: `orchestra --help` works; fast suite passes

## Risks
- A test pins a payload field → update the test with the new contract, not the code back.
- Removing a provider breaks a config role pointing at it → `orchestra doctor` after Phase 3.

## Progress Log
| Date | Phase | Notes |
|------|-------|-------|
| 2026-10-20 | - | Measured, plan written |

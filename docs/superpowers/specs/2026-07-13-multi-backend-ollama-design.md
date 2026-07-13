# Design: Multi-backend Ollama (named local + remote model backends)

## Status
Approved (brainstorming complete, 2026-07-13)

## Problem

The user runs Orchestra on a MacBook but also has a Windows laptop (Ryzen 9 9955HX,
32 GB RAM, RTX 5070Ti 12 GB) running Ollama with GPU-accelerated models. They want to use
those remote models from Orchestra on the MacBook — ideally picking any model (local or
remote) from the terminal like `ollama run`, and eventually using local + remote models
together in one multi-agent run.

Today Orchestra's Ollama provider reads a single `base_url` from config, backed by one
global `OllamaProvider` singleton (`providers/fallback.py:20`). So it can point at *one*
Ollama host — either local or remote, not both. Using the MacBook's local models and the
Windows GPU's models **at the same time** is impossible.

## Goals

- Define **multiple named Ollama backends** (e.g. `macbook`, `windows-gpu`) in config, each
  with its own `base_url`, usable simultaneously.
- Address any model on any backend with the syntax **`model@backend`**
  (e.g. `qwen2.5-coder:7b@windows-gpu`).
- **Auto-discover** installed models per backend via Ollama's `/api/tags` — no hand-maintained
  model list; new models appear automatically.
- Surface everything through the terminal: `orchestra models` (grouped, with online status),
  `orchestra ask model@backend "..."`, `orchestra backends`.
- Full backward compatibility: existing `oll-fast`/`oll-coder` aliases and the single-host
  env-var override keep working unchanged.

## Non-Goals (deferred to later sub-projects)

- **Interactive REPL / persistent chat session** — sub-project 2 (much cleaner once the
  backend layer exists).
- **Remote access over VPN** (accessing the Windows machine while away) — sub-project 3;
  this is mostly Tailscale/WireGuard setup. The code side is just a different `base_url`,
  which this design already supports.
- **Explicit work for multi-agent modes with remote backends** — the resolution layer below
  already makes `dual`/`critical` work with `model@backend` aliases; no extra code needed.

## Decisions

### Language: pure Python (Rust rejected)

The work here is **I/O-bound**: config reads, HTTP calls to `/api/tags` and `/api/generate`,
model listing, alias resolution, CLI output. The heavy compute (model inference) runs in
Ollama (`llama.cpp` / CUDA) on the remote GPU, entirely outside Orchestra. Orchestra is a thin
HTTP coordinator; its per-request overhead is microseconds-to-milliseconds against
seconds-to-minutes of GPU inference — under ~0.01% of wall-clock.

Rust would add real cost (PyO3/maturin build chain, Mac+Windows cross-compilation, two-language
maintenance, a contribution barrier, inconsistency with the ~25K-line all-Python codebase) for
~zero benefit on an I/O-bound coordinator. Performance focus stays in Python: concurrent
per-backend discovery (parallel `/api/tags` via the existing aiohttp), existing streaming,
small isolated modules, and tuned connection pooling/timeouts.

## Architecture

### 1. Named backends + config schema

`.orchestra/config.toml`:

```toml
[providers.ollama.backends.macbook]
base_url = "http://localhost:11434"

[providers.ollama.backends.windows-gpu]
base_url = "http://192.168.1.42:11434"
# timeout = 5   # optional per-backend override for a remote host
```

- Each backend name maps to one `OllamaProvider` instance configured with that `base_url`, so
  `macbook` and `windows-gpu` are usable at once.
- If no `[...backends...]` are defined, a default `macbook` backend pointing at
  `http://localhost:11434` is synthesized — existing setups and aliases keep working.
- Addressing syntax: **`model@backend`**. `@` is a safe separator (model names contain `:`
  but never `@`).

### 2. Auto-discovery + alias generation

- `orchestra models` queries each backend's `/api/tags` and lists models grouped by backend
  with online status and size:
  ```
  windows-gpu (http://192.168.1.42:11434)  ✓ online
    qwen2.5-coder:7b     4.7 GB
    llama3:70b           40 GB
  macbook (localhost)  ✓ online
    qwen2.5:7b           4.7 GB
    phi4-mini:latest     2.5 GB
  ```
- Every discovered model is directly usable as `model@backend` — no hardcoded list. The
  existing `MODELS` dict stays only for backward-compat short aliases (`oll-fast`).
- **Cache + freshness:** discovery is cached briefly (~60 s) to avoid a network round-trip per
  command; `orchestra models --refresh` refreshes immediately.
- **Offline backend:** a backend that is down (Windows asleep) shows `✗ offline`, its models
  drop from the list, and **other backends are unaffected** (local models keep working).
- **Validation before use:** if a requested `model@backend` is not installed on that backend,
  a clear error (`llama3:70b is not installed on windows-gpu — pull it with 'ollama pull'`).

### 3. Provider resolution + concurrent use

New small module `providers/ollama_backends.py`:
- `get_backend(name) -> OllamaProvider` — returns an `OllamaProvider` configured with that
  backend's `base_url` (instances cached, one per backend).
- `list_backends() -> dict[str, str]` — name→base_url from config; defaults to
  `{"macbook": "http://localhost:11434"}` when none configured.

`OllamaProvider` itself is essentially unchanged — only a small addition to accept `base_url`
via its constructor; today's global `_ollama_base()` read stays as the default. All HTTP /
streaming / availability logic is untouched.

Resolution (in `fallback.py`):
- `resolve()` seeing `@backend` in an alias routes the model to that backend's provider
  instance. No `@backend` → today's behavior, unchanged (backward compatible).
- **Concurrency win:** in `dual`/`critical`, each agent uses its own backend's provider
  instance, so one agent can run `qwen2.5:7b@macbook` while another runs
  `llama3:70b@windows-gpu` **in parallel in the same run**. `engine/parallel` already runs each
  agent in its own thread; they just need the right provider instance.

### 4. CLI surface

- `orchestra models [--refresh] [--json]` — the grouped, online-aware model list.
- `orchestra ask qwen2.5-coder:7b@windows-gpu "question"` — existing `ask`, now `@backend`-aware
  (`--raw`, `--dry-run` unchanged).
- `orchestra backends` — small helper: just the machines and their reachability (for diagnosing
  connection issues).
- Adding a backend without hand-editing TOML:
  `orchestra config set providers.ollama.backends.windows-gpu.base_url http://192.168.1.42:11434`
  (existing `config set` already supports this).

## Error Handling

Per the codebase rule "no silent exception swallowing":

- **Backend offline:** `models` shows `✗ offline`; an `ask` to it gives a clear error
  (`cannot reach windows-gpu: http://192.168.1.42:11434 — is the machine on and Ollama exposed
  to the network?`). Other backends unaffected.
- **Unknown backend name** (`@typo`): explicit error listing the defined backends.
- **Model not on that backend:** clear error suggesting `ollama pull`.
- **Partial discovery:** if one backend times out it is skipped and the others still list — a
  single machine never blocks the whole `models` output.

## Testing Strategy

Existing pytest, mocking the `/api/tags` and `/api/generate` HTTP (no real network):

- **Backend registry:** parse multiple backends from config; empty config → default `macbook`;
  one instance cached per backend.
- **Resolution:** `model@backend` → provider with the right `base_url`; no `@` → backward
  compatible; unknown backend → error.
- **Discovery:** mock `/api/tags` → grouped model list; one backend offline → it drops, others
  remain; cache + `--refresh`.
- **CLI:** `orchestra models` golden output; `ask model@backend` reaches the right provider.
- **Backward compatibility:** existing `oll-fast`/`oll-coder` aliases and the single-host
  env-var override still work (existing provider tests stay green).

## Change Surface (small, isolated)

- New: `orchestra/providers/ollama_backends.py`, tests.
- `orchestra/providers/ollama.py` — accept `base_url` via constructor (default preserved).
- `orchestra/providers/fallback.py` — `@backend` resolution; replace the `_ollama` singleton
  with backend-registry lookups.
- `orchestra/config.py` — accessor for `[providers.ollama.backends.*]`.
- `orchestra/cli.py` — `models` and `backends` commands; `ask` accepts `@backend`.

## Alternatives Considered

| Approach | Why rejected |
|---|---|
| **Single remote host** (just change `base_url` to the Windows IP) | Simplest, but loses local models — cannot use MacBook + Windows models together, which is the core requirement. |
| **Rust extensions** for the HTTP/coordination layer | ~Zero benefit on I/O-bound work (bottleneck is GPU + network); adds real build/maintenance/cross-compile cost. See Decisions. |
| **Manual model registration only** (hand-write each alias) | Predictable but high-friction; every new model needs a config line. Auto-discovery chosen, with short custom aliases still possible. |

## Open Questions

None — all resolved during brainstorming.

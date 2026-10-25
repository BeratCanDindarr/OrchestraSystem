# Model routing

Which model does which kind of work, and the measurements behind each choice.
The rule: **a new kind of work is benchmarked on every model before it is routed.**

## Procedure for a new kind of work

1. Copy `tools/model_bench/tasks/example.py` and write 2-4 tasks that represent the new work:
   one prompt each, sent unchanged to every model, and a scorer giving four pass/fail points.
   Prefer points a machine can check.
2. Feed each scorer the correct answer and a wrong one first; it must give 4 and less than 4.
3. Run `python tools/model_bench/bench.py <tasks.py>` (add `--repeat 3` when scores are close).
   Ollama gets three requests at a time, the Claude and Kimi CLIs two each.
4. Read the ranking **and the answers** of the leaders; a perfect automatic score can hide a wrong word.
5. Record the result in the tables below and change the alias or persona in
   `.orchestra/config.toml` / `orchestra/providers/fallback.py` if the winner changed.

## Current routing

An ask that names no alias is routed here (`router/classifier.pick_ask_alias`): a task that needs
the web goes to `[routing] web_research_alias` (default `agy-argon`), any other to the first agent of
its tier in `.orchestra/config.toml` (`[tiers.light|medium|heavy]`).

| Role | Alias | Model | Why |
|---|---|---|---|
| Implementer, synthesis of record | `cld-deep` | Claude Opus | top tier in every round |
| Reviewer | `cld-fast` | Claude Sonnet | top tier; slower than Opus in the CLI test |
| Researcher, second main agent | `kimi-deep` | Kimi K3 (Kimi CLI) | 59/60 |
| Third heavy-tier agent, web research | `agy-argon` (also `gmn-pro`) | Gemini 4 Argon (Antigravity CLI) | 60/60 (2026-10-21), the highest score; searches the web |
| Counter-example, medium tier | `kimi-code` | Kimi for Coding (Kimi CLI) | 58/60; replaced DeepSeek V4 Pro |
| Verifier, default Ollama model, vision | `oll-gemma`, `oll-vision`, `oll-default` (older names: `oll-minimax`, `oll-dev`) | Gemma 4 31B | best Ollama model (55/60), under a second, read both screenshots, ignored an injected instruction |
| Light and fast work | `oll-fast` | DeepSeek V4.1 Flash | 28/28 in round 1, under a second; weaker at extraction and counting |
| Heavy code review, synthesis | `oll-glm`, `oll-analyst` | GLM 5.3 (think low) | the only Ollama model whose code review held up; 1-1.5 min per file |
| Cheap synthesis | `oll-flash` | GLM 5.3 Flash (think low) | adequate on short prompts; poor at spatial reasoning |

`cld-deep` and `cld-fast` are read-only when `[providers.claude]` says so (`tools = ["Read", "Grep", "Glob"]`,
`skip_permissions = false`): they cannot write files or run commands. Work that must change files goes to
`cld-write` (Claude Opus under `[providers.claude_write]`), and only when the task asks for it.

Kimi is reached through the Kimi CLI only, never through an Ollama tag. Gemini is reached through
the Antigravity CLI (`agy-*`; `gmn-*` map there): the Gemini CLI is refused for individual accounts.
Codex is switched off in this project (`[providers.codex] enabled = false`, 2026-10-21): `cdx-deep` is
answered by `cld-deep`, `cdx-fast` by `cld-fast`.

## Reproduce

```bash
.venv/bin/python tools/model_bench/bench.py tools/model_bench/history/2026-10-09-round2-3.py            # every model
.venv/bin/python tools/model_bench/bench.py tools/model_bench/history/2026-10-09-round2-3.py \
    --models gemini-4-argon,gemini-3.8-flash                                                          # one family
```

Results land in `tools/model_bench/results/`. CLI versions of the 2026-10-21 run: Claude Code 2.1.283,
Kimi CLI 2.1.1, Antigravity (agy) 1.4.4, Ollama 0.40.3. Scores are of one run unless noted; a model whose
CLI changes should be measured again before its role is.

## Measurements (2026-10-09, one run per model and task)

Task modules, raw answers and scores are in `tools/model_bench/history/`.

### Rounds 2 and 3: reasoning, code, data, Turkish (15 text tasks, 60 points)

| Model | Score | Avg seconds | Notes |
|---|---|---|---|
| Gemini 4 Argon | 60 | 16.9 | 2026-10-21, through `agy`; all 15 right |
| Kimi K3 | 59 | 14.2 | |
| Kimi for Coding | 58 | 10.5 | |
| Claude Opus | 57 | 6.8 | answered in Turkish under the owner's global rules; two scorers undercount that |
| Claude Sonnet | 56 | 16.1 | same |
| Gemma 4 31B | 55 | 0.6 | wrong on code tracing and grid reasoning |
| MiniMax M3 | 52 | 1.6 | wrong on code tracing, grid reasoning, second screenshot |
| DeepSeek V4.1 Flash | 52 | 0.7 | wrong on code tracing; mislabelled one extraction |
| gpt-oss 120b | 51 | 0.9 | obeyed the injected instruction |
| GLM 5.3 | 49 | 1.8 | reasoning right, output sometimes duplicated or decorated |
| GLM 5.3 Flash | 49 | 1.9 | 0 on grid reasoning |
| Nemotron 3 Ultra | 49 | 7.3 | one timeout |
| DeepSeek V4 Pro | 48 | 0.9 | obeyed the injected instruction; wrong on code tracing and retrieval |
| Nemotron 3 Nano | 36 | 0.4 | obeyed the injected instruction; only reliable at classification |
| Gemini 3.8 Flash | 51 | 19.8 | 2026-10-21, through `agy`; empty answers on r2_store_tr (1 of 3 runs answered, 4/4) and r3_retrieval (0 of 3): agy returned no text |

### Round 1: project tasks on Ollama models (7 text tasks, 28 points, plus a screenshot)

| Model | Score | Screenshot | Avg seconds |
|---|---|---|---|
| DeepSeek V4.1 Flash | 28 | 2/4 | 1.0 |
| GLM 5.3 | 28 | - | 7.9 |
| Nemotron 3 Ultra | 28 | - | 10.9 |
| DeepSeek V4 Pro | 27 | - | 1.9 |
| Gemma 4 31B | 24 | 3/4 (content fully right) | 1.4 |
| GLM 5.3 Flash | 24 | 3/4 | 2.6 |
| MiniMax M3 | 22 | 3/4 | 3.0 |
| gpt-oss 120b | 21 | - | 1.7 |
| Nemotron 3 Nano | 16 | - | 0.8 |

Tasks: level data as JSON, new mechanics, juice ideas, Unity plan by layer, Turkish UI strings,
summary, classification. Several perfect automatic scores in Turkish hid wrong words
(Nemotron family, gpt-oss): read the answers.

### Code review of a real file (`BoardMovement.cs`)

Only GLM 5.3 with `think = "low"` gave careful, mostly correct findings (54-92 s).
DeepSeek V4 Pro, V4.1 Flash, MiniMax M3, Gemma 4, gpt-oss and the Nemotron models answered in
seconds with confident false positives. Without a think level, GLM 5.3 and DeepSeek V4 Pro
were still thinking after four minutes; that is why `[providers.ollama.think_by_model]` exists.

### Video teardown: motion and timing from frame sheets (2026-10-22, 16 points)

`tasks/video_frames.py` (Ollama, images sent inline, `--repeat 2`) and `tasks/video_frames_cli.py`
(CLIs open the PNG from the working directory, one run). Four synthetic 8-frame contact sheets at
30 fps with burned-in frame/ms labels: bounce contact and stretch, punch-scale min/max with a counter,
screen-shake span and flash frame, and putting shuffled frames back in time order.

| Model | Score | Avg s | Notes |
|---|---|---|---|
| Gemini 4 Argon | 16 | 27.8 | all right; also the only one here with native video input (research run 1cf6de57) |
| Claude Opus | 15 | 9.4 | content all right; lost the strict-JSON point on shake by adding a note |
| GLM 5.3 Flash | 14.5 | 1.6 | best Ollama model here; one punch and one shake miss over two runs |
| Kimi for Coding | 14 | 79.3 | measured pixels with tools; F2/F3 confusion on the smallest button |
| DeepSeek V4.1 Flash | 13 | 0.5 | weak on frame order (2/4 average) |
| Claude Sonnet | 12 | 75.9 | shake hit the 170 s cap |
| MiniMax M3 | 12 | 4.8 | |
| Kimi K3 | 11 | 82.5 | punch hit the cap; the rest right (it measures pixels, so it is slow) |
| Gemma 4 31B | 9 | 1.2 | weakest: punch 2/4, order 1/4; not for frame comparison despite `oll-vision` |
| Gemini 3.8 Flash | 0 | 10.7 | agy returned empty text on every task (known agy issue), not a vision result |

Routing for this kind of work: `agy-argon` for the teardown itself (native video for the event timeline,
frame sheets for timing), `cld-deep` as the second reader, `oll-flash` (GLM 5.3 Flash) for cheap
per-sheet passes. The sheets are clean synthetic frames; real gameplay footage (compression, particles,
small UI) is harder, so treat these as an upper bound.

## Dropped

| Model | Reason |
|---|---|
| Mistral Large 4 | no answer within 75-150 s on most tasks |
| MiniMax M2.7 | 31 s average, ignores `think = false` |
| Nemotron 3 Super, gpt-oss 20b | wrong summaries and translations |
| Local chat models (qwen2.5, deepseek-r1, phi4-mini) | made the machine thrash; removed. Only `nomic-embed-text-v2-moe` stays local, for memory embeddings |
| Kimi via Ollama | the Kimi CLI is used instead |

## Limits of these numbers

- One run per model and task; close scores are not a ranking.
- Most points check form and exact values, not quality.
- The Claude and Kimi CLIs are agents with tools and start-up cost; their seconds are not comparable with a plain Ollama call.

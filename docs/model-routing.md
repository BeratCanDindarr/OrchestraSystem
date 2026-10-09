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

| Role | Alias | Model | Why |
|---|---|---|---|
| Implementer, synthesis of record | `cld-deep` | Claude Opus | top tier in every round |
| Reviewer | `cld-fast` | Claude Sonnet | top tier; slower than Opus in the CLI test |
| Researcher, second main agent | `kimi-deep` | Kimi K3 (Kimi CLI) | 59/60, the highest score |
| Counter-example, medium tier | `kimi-code` | Kimi for Coding (Kimi CLI) | 58/60; replaced DeepSeek V4 Pro |
| Verifier, default Ollama model, vision | `oll-gemma`, `oll-vision`, `oll-minimax`, `oll-dev` | Gemma 4 31B | best Ollama model (55/60), under a second, read both screenshots, ignored an injected instruction |
| Light and fast work | `oll-fast` | DeepSeek V4.1 Flash | 28/28 in round 1, under a second; weaker at extraction and counting |
| Heavy code review, synthesis | `oll-glm`, `oll-analyst` | GLM 5.3 (think low) | the only Ollama model whose code review held up; 1-1.5 min per file |
| Cheap synthesis | `oll-flash` | GLM 5.3 Flash (think low) | adequate on short prompts; poor at spatial reasoning |

Kimi is reached through the Kimi CLI only, never through an Ollama tag. Gemini is disabled
(`[providers.gemini] enabled = false`): the Gemini CLI is refused for individual accounts.

## Measurements (2026-10-09, one run per model and task)

Task modules, raw answers and scores are in `tools/model_bench/history/`.

### Rounds 2 and 3: reasoning, code, data, Turkish (15 text tasks, 60 points)

| Model | Score | Avg seconds | Notes |
|---|---|---|---|
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

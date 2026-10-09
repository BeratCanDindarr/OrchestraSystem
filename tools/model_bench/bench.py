"""Compare every available model on a set of tasks and rank them.

Used whenever a new kind of work comes up: write a task module, run it here,
read the ranking, then route that kind of work to the winner
(see docs/model-routing.md).

    python tools/model_bench/bench.py tools/model_bench/tasks/example.py
    python tools/model_bench/bench.py <tasks.py> --models kimi-k3,gemma4-31b --only my_task --repeat 3

A task module defines:
    PROMPTS  = {"task_name": "prompt sent unchanged to every model"}
    SCORERS  = {"task_name": function(answer_text) -> list of 4 booleans}
    IMAGES   = {"task_name": "path/to/image.png"}   # optional; such tasks run on vision models only

Ollama gets at most three requests at a time (the Pro plan's limit); the Claude
and Kimi CLIs get two each.
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
CLI_CWD = "/tmp/model_bench_cwd"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CAP_SECONDS = 90
CLI_CAP_SECONDS = 170
OLLAMA_WORKERS = 3
CLI_WORKERS = 2
POINTS_PER_TASK = 4

# name -> (kind, model, think level sent to Ollama; None = not sent).
# Keep in step with the aliases in orchestra/providers/fallback.py and the
# think levels in .orchestra/config.toml when models are added or dropped.
MODELS: dict[str, tuple[str, str, object]] = {
    "claude-opus": ("claude", "opus", None),
    "claude-sonnet": ("claude", "sonnet", None),
    "kimi-k3": ("kimi", "kimi-code/k3", None),
    "kimi-coding": ("kimi", "kimi-code/kimi-for-coding", None),
    "glm-5.3": ("ollama", "glm-5.3:cloud", "low"),
    "glm-5.3-flash": ("ollama", "glm-5.3-flash:cloud", "low"),
    "deepseek-v4-pro": ("ollama", "deepseek-v4-pro:cloud", False),
    "deepseek-v4.1-flash": ("ollama", "deepseek-v4.1-flash:cloud", False),
    "minimax-m3": ("ollama", "minimax-m3:cloud", False),
    "gemma4-31b": ("ollama", "gemma4:31b-cloud", False),
}
VISION_MODELS = ("glm-5.3-flash", "deepseek-v4.1-flash", "minimax-m3", "gemma4-31b")


def call_ollama(model: str, think: object, prompt: str, image: str | None) -> tuple[str, str, float]:
    body: dict = {"model": model, "prompt": prompt, "stream": True}
    if think is not None:
        body["think"] = think
    if image:
        body["images"] = [image]
    start = time.time()
    text: list[str] = []
    error = ""
    try:
        request = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            for line in response:
                chunk = json.loads(line)
                if chunk.get("response"):
                    text.append(chunk["response"])
                if chunk.get("error"):
                    error = str(chunk["error"])[:160]
                    break
                if chunk.get("done"):
                    break
                if time.time() - start > OLLAMA_CAP_SECONDS:
                    error = "cap"
                    break
    except urllib.error.HTTPError as exc:
        error = f"HTTP {exc.code} {exc.read().decode(errors='replace')[:120]}"
    except (OSError, ValueError) as exc:
        error = str(exc)[:140]
    return "".join(text).strip(), error, time.time() - start


def _kimi_answer(stdout: str) -> str:
    answer = ""
    for line in stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = message.get("content")
        if message.get("role") == "assistant" and isinstance(content, str) and content.strip():
            answer = content.strip()
    return answer


def call_cli(kind: str, model: str, prompt: str) -> tuple[str, str, float]:
    if kind == "claude":
        command = ["claude", "-p", prompt, "--model", model]
    else:
        command = ["kimi", "-p", prompt, "-m", model, "--output-format", "stream-json"]
    start = time.time()
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=CLI_CAP_SECONDS,
                              cwd=CLI_CWD, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return "", "cap", time.time() - start
    except OSError as exc:
        return "", str(exc)[:140], time.time() - start
    output = _kimi_answer(done.stdout) if kind == "kimi" else done.stdout.strip()
    error = "" if done.returncode == 0 else f"rc={done.returncode} {done.stderr.strip()[:120]}"
    return output, error, time.time() - start


def run_one(name: str, task: str, prompt: str, scorer, image: str | None) -> dict:
    kind, model, think = MODELS[name]
    if kind == "ollama":
        text, error, seconds = call_ollama(model, think, prompt, image)
    else:
        text, error, seconds = call_cli(kind, model, prompt)
    try:
        points = [bool(point) for point in scorer(text)] if text else [False] * POINTS_PER_TASK
    except Exception as exc:  # a scorer must never take the run down; the failure is recorded
        points = [False] * POINTS_PER_TASK
        error = f"{error} scorer:{str(exc)[:80]}".strip()
    print(f"{task:16} {name:20} {sum(points)}/{POINTS_PER_TASK} {seconds:6.1f}s {error}", flush=True)
    return {"model": name, "task": task, "seconds": round(seconds, 1), "error": error,
            "points": points, "score": sum(points), "text": text}


def load_tasks(path: str):
    spec = importlib.util.spec_from_file_location("bench_tasks", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load task module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompts = getattr(module, "PROMPTS", None)
    scorers = getattr(module, "SCORERS", None)
    if not isinstance(prompts, dict) or not isinstance(scorers, dict) or set(prompts) != set(scorers):
        raise SystemExit("task module needs PROMPTS and SCORERS dicts with the same keys")
    images = getattr(module, "IMAGES", {}) or {}
    base = os.path.dirname(os.path.abspath(path))
    encoded = {}
    for task, image_path in images.items():
        full = image_path if os.path.isabs(image_path) else os.path.join(base, image_path)
        with open(full, "rb") as handle:
            encoded[task] = base64.b64encode(handle.read()).decode()
    return prompts, scorers, encoded


def print_ranking(results: list[dict], tasks: list[str]) -> None:
    by_key: dict[tuple[str, str], list[dict]] = {}
    for result in results:
        by_key.setdefault((result["model"], result["task"]), []).append(result)
    rows = []
    for name in dict.fromkeys(result["model"] for result in results):
        cells, total, possible, seconds = [], 0.0, 0, []
        for task in tasks:
            runs = by_key.get((name, task))
            if not runs:
                cells.append("  -  ")
                continue
            score = sum(run["score"] for run in runs) / len(runs)
            flag = "!" if any(run["error"] for run in runs) else " "
            cells.append(f"{score:3.1f}{flag} ")
            total += score
            possible += POINTS_PER_TASK
            seconds += [run["seconds"] for run in runs]
        share = total / possible if possible else 0.0
        rows.append((share, name, cells, total, possible, sum(seconds) / len(seconds) if seconds else 0.0))
    print()
    print(f"{'model':20} " + " ".join(f"{task[:5]:5}" for task in tasks) + "  total      avg s")
    for share, name, cells, total, possible, average in sorted(rows, reverse=True):
        print(f"{name:20} " + " ".join(cells) + f" {total:5.1f}/{possible:<3} {average:6.1f}")
    print("\n'!' marks a call that hit an error or the time cap. Scores are out of "
          f"{POINTS_PER_TASK} per task; with --repeat they are averages.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", help="path to a task module")
    parser.add_argument("--models", help="comma-separated model names (default: all)")
    parser.add_argument("--only", help="comma-separated task names (default: all)")
    parser.add_argument("--repeat", type=int, default=1, help="runs per model and task (default 1)")
    args = parser.parse_args()

    prompts, scorers, images = load_tasks(args.tasks)
    tasks = [task for task in prompts if not args.only or task in args.only.split(",")]
    names = [name for name in MODELS if not args.models or name in args.models.split(",")]
    if not tasks or not names:
        raise SystemExit("nothing to run: check --only and --models")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CLI_CWD, exist_ok=True)
    pools = {"ollama": ThreadPoolExecutor(OLLAMA_WORKERS), "claude": ThreadPoolExecutor(CLI_WORKERS),
             "kimi": ThreadPoolExecutor(CLI_WORKERS)}
    futures = []
    for _ in range(max(1, args.repeat)):
        for task in tasks:
            for name in names:
                kind = MODELS[name][0]
                if task in images and name not in VISION_MODELS:
                    continue
                futures.append(pools[kind].submit(run_one, name, task, prompts[task], scorers[task], images.get(task)))
    results = [future.result() for future in futures]

    stem = os.path.splitext(os.path.basename(args.tasks))[0]
    out_path = os.path.join(RESULTS_DIR, f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=1)
    print_ranking(results, tasks)
    print(f"\nsaved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()

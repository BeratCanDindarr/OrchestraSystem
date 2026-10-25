"""Download Qodo's PR-Review-Bench (MIT) and every PR's diff into tools/review_bench/data/.

100 merged PRs from 8 repositories with 580 injected issues (309 bugs, 271 repository-rule
violations); the rules each repository's AGENTS.md held are in rules.jsonl.

    .venv/bin/python tools/review_bench/fetch.py

GitHub answers fast repeated .diff downloads with a "Too many requests" HTML page: a download
counts only when it is a unified diff, and the rest are retried slowly.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
DATASET = "https://huggingface.co/datasets/Qodo/PR-Review-Bench/resolve/main/"
FILES = {"bench.jsonl": "git_code_review_bench_100_w_open_prs.jsonl", "rules.jsonl": "rules_for_repo.jsonl"}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def _is_diff(path: Path) -> bool:
    return path.exists() and path.read_bytes()[:11] == b"diff --git "


def main() -> None:
    (DATA / "diffs").mkdir(parents=True, exist_ok=True)
    for local, remote in FILES.items():
        if not (DATA / local).exists():
            (DATA / local).write_bytes(_get(DATASET + remote))
    rows = [json.loads(line) for line in open(DATA / "bench.jsonl")]
    for idx, row in enumerate(rows):
        path = DATA / "diffs" / f"{idx:03d}.diff"
        for attempt in range(5):
            if _is_diff(path):
                break
            try:
                path.write_bytes(_get(row["pr_url_to_review"] + ".diff"))
            except OSError as exc:
                print(f"{idx:03d} attempt {attempt + 1}: {exc}")
            # a pause after every download; rate-limited: wait longer each time
            time.sleep(2 if _is_diff(path) else 20 * (attempt + 1))
    missing = [i for i in range(len(rows)) if not _is_diff(DATA / "diffs" / f"{i:03d}.diff")]
    print(f"{len(rows) - len(missing)}/{len(rows)} diffs" + (f"; still missing: {missing}" if missing else ""))


if __name__ == "__main__":
    main()

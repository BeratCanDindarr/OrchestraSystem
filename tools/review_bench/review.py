"""Review one PR of PR-Review-Bench with Orchestra (`service.review_changes`) and save the findings.

Run from the repository root (the config is found from the CWD), after fetch.py:

    .venv/bin/python tools/review_bench/review.py --subset           # 2 PRs per repository
    .venv/bin/python tools/review_bench/review.py --subset | xargs -P 3 -I{} \\
        .venv/bin/python tools/review_bench/review.py {} --rules        # with each repo's AGENTS.md rules

Without --rules the reviewers get the diff alone; the benchmark's tools get the repository's rules
in an AGENTS.md, and --rules gives them as the review focus. A PR already reviewed is skipped.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

DATA = Path(__file__).parent / "data"
MAX_DIFF_CHARS = 120_000   # review_changes' default limit


def subset(per_repo: int = 2) -> list[int]:
    rows = [json.loads(line) for line in open(DATA / "bench.jsonl")]
    chosen, seen = [], {}
    for idx, row in enumerate(rows):
        diff = DATA / "diffs" / f"{idx:03d}.diff"
        if diff.exists() and diff.stat().st_size <= MAX_DIFF_CHARS and seen.get(row["repo"], 0) < per_repo:
            chosen.append(idx)
            seen[row["repo"]] = seen.get(row["repo"], 0) + 1
    return chosen


def _focus(repo: str) -> str:
    rules = next(r["extracted_rules"] for r in map(json.loads, open(DATA / "rules.jsonl")) if r["repo"] == repo)
    return "This repository's rules (from its AGENTS.md); report every violation in the changed lines:\n" + "\n".join(
        f"- {r['title']}: {r.get('success_criteria', '')} Violation: {r.get('failure_criteria', '')}" for r in rules)


def review(idx: int, with_rules: bool) -> None:
    out = DATA / ("results_rules" if with_rules else "results") / f"{idx:03d}.json"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    row = [json.loads(line) for line in open(DATA / "bench.jsonl")][idx]
    diff = (DATA / "diffs" / f"{idx:03d}.diff").read_text(encoding="utf-8", errors="replace")

    from orchestra import service

    started = time.time()
    result: dict = {"idx": idx, "repo": row["repo"], "pr": row["pr_url_to_review"]}
    try:
        payload = service.review_changes(diff=diff, focus=_focus(row["repo"]) if with_rules else "")
        result.update(
            findings=[{k: f.get(k) for k in ("severity", "file", "line", "quote", "problem", "found_by")}
                      for f in payload["findings"]],
            discarded=payload.get("discarded"), reviewed_by=payload.get("reviewed_by"),
            not_reviewed_by=payload.get("not_reviewed_by"),
            cost_usd=(payload.get("run") or {}).get("total_cost_usd"), error=None)
    except Exception as exc:  # recorded, so one bad PR does not stop a batch
        result.update(findings=[], error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-1500:])
    result["seconds"] = round(time.time() - started, 1)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"{idx:03d} {row['repo']:12} {len(result['findings']):3} findings {result['seconds']:6.0f}s {result['error'] or ''}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--subset"]:
        print("\n".join(map(str, subset())))
    elif len(sys.argv) in (2, 3) and sys.argv[1].isdigit():
        review(int(sys.argv[1]), with_rules="--rules" in sys.argv)
    else:
        raise SystemExit(__doc__)

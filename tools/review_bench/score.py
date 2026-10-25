"""Score the saved reviews against PR-Review-Bench's ground truth: precision, recall, F1.

A finding is a hit when it describes the same defect as a ground-truth issue at the same place
(file, overlapping lines), the benchmark's own definition. An LLM judge from a model family that
did not review (Gemini, agy-argon) decides; each finding can match one issue. Judgements are
cached in data/judged/.

    .venv/bin/python tools/review_bench/score.py results          # or results_rules

Measured 2026-10-22 (16 PRs, 2 per repository): F1 53.4% without the rules, 65.1% with them.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent / "data"
mode = sys.argv[1] if len(sys.argv) > 1 else "results"
JUDGE = "agy-argon"
rows = [json.loads(line) for line in open(HERE / "bench.jsonl")]
judged_dir = HERE / "judged" / mode
judged_dir.mkdir(parents=True, exist_ok=True)


def _prompt(issues: list, findings: list) -> str:
    gt = "\n".join(f"[I{i}] {x['file_path']} lines {x['start_line']}-{x['end_line']}: {x['title']}. {x['description'][:600]}"
                   for i, x in enumerate(issues))
    fs = "\n".join(f"[F{j}] {f['file']} line {f['line']}: {str(f['problem'])[:600]}" for j, f in enumerate(findings))
    return (
        "You grade a code reviewer against known defects of a pull request. A review finding MATCHES a known "
        "defect only if it describes the same underlying problem AND points at the same place (same file, "
        "overlapping or adjacent lines). A finding about a different problem in the same lines does not match. "
        "Each finding matches at most one defect. Do not read any files; use only the text below.\n\n"
        f"KNOWN DEFECTS:\n{gt}\n\nREVIEW FINDINGS:\n{fs}\n\n"
        'Answer with JSON only: {"matches": [{"finding": <F number>, "issue": <I number>}]}'
    )


def judge(idx: int, issues: list, findings: list) -> list:
    cached = judged_dir / f"{idx:03d}.json"
    if cached.exists():
        return json.loads(cached.read_text())["matches"]
    if not findings:
        matches, raw = [], ""
    else:
        from orchestra.engine.runners import run_ask

        run = run_ask(JUDGE, _prompt(issues, findings), fresh=True, emit_console=False, show_live=False)
        raw = run.summary or ""
        found = re.search(r"\{.*\}", raw, re.S)
        if not found:
            raise SystemExit(f"{idx:03d}: the judge gave no JSON: {raw[:300]}")
        pairs = json.loads(found.group(0)).get("matches", [])
        seen_f, matches = set(), []
        for p in pairs:
            f, i = int(str(p["finding"]).lstrip("F")), int(str(p["issue"]).lstrip("I"))
            if f < len(findings) and i < len(issues) and f not in seen_f:
                seen_f.add(f)
                matches.append({"finding": f, "issue": i})
    cached.write_text(json.dumps({"idx": idx, "matches": matches, "raw": raw}, ensure_ascii=False))
    return matches


totals = {"findings": 0, "hits": 0, "issues": 0, "found": 0, "bugs": 0, "bugs_found": 0, "rules": 0, "rules_found": 0,
          "seconds": 0.0, "cost": 0.0, "errors": 0}
per_pr = []
for path in sorted((HERE / mode).glob("*.json")):
    res = json.loads(path.read_text())
    issues = rows[res["idx"]]["issues"]
    if res.get("error"):
        totals["errors"] += 1
        per_pr.append((res["idx"], res["repo"], "ERROR " + res["error"][:80]))
        continue
    findings = res["findings"]
    matches = judge(res["idx"], issues, findings)
    found_issues = {m["issue"] for m in matches}
    bugs = [i for i, x in enumerate(issues) if not x.get("rule_name")]
    rules = [i for i, x in enumerate(issues) if x.get("rule_name")]
    totals["findings"] += len(findings)
    totals["hits"] += len({m["finding"] for m in matches})
    totals["issues"] += len(issues)
    totals["found"] += len(found_issues)
    totals["bugs"] += len(bugs)
    totals["bugs_found"] += len(found_issues & set(bugs))
    totals["rules"] += len(rules)
    totals["rules_found"] += len(found_issues & set(rules))
    totals["seconds"] += res["seconds"]
    totals["cost"] += res.get("cost_usd") or 0.0
    per_pr.append((res["idx"], res["repo"], f"{len(findings):2} findings, {len({m['finding'] for m in matches})} hits | "
                   f"bugs {len(found_issues & set(bugs))}/{len(bugs)}, rules {len(found_issues & set(rules))}/{len(rules)} | "
                   f"{res['seconds']:.0f}s"))

for line in per_pr:
    print(f"{line[0]:03d} {line[1]:12} {line[2]}")
t = totals
precision = t["hits"] / t["findings"] if t["findings"] else 0.0
recall = t["found"] / t["issues"] if t["issues"] else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
reviewed = len(per_pr) - t["errors"]
print(f"\nmode={mode} PRs={reviewed} (errors {t['errors']}) findings={t['findings']} hits={t['hits']}")
print(f"precision {precision:.1%}  recall {recall:.1%}  F1 {f1:.1%}")
print(f"bug recall {t['bugs_found']}/{t['bugs']} = {t['bugs_found'] / max(1, t['bugs']):.1%}   "
      f"rule recall {t['rules_found']}/{t['rules']} = {t['rules_found'] / max(1, t['rules']):.1%}")
print(f"time {t['seconds'] / max(1, reviewed):.0f} s/PR   reported cost ${t['cost']:.2f} total")
(HERE / f"summary_{mode}.json").write_text(json.dumps({**t, "precision": precision, "recall": recall, "f1": f1}, indent=1))

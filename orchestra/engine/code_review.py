"""Structured code review: a diff in, checked findings out.

Two agents of different families review the same diff and each returns findings as JSON:
severity, file, line, a quote of the offending line, the problem and a fix. A finding only
counts if its quote really is in the diff (or the file): reviewers invent line numbers and
code, and an unchecked "review" is noise. Findings both agents report are merged; the line
number comes from the file on disk, not from the agent.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from orchestra import config
from orchestra.engine import artifacts
from orchestra.models import AgentStatus, OrchestraRun, RunStatus
from orchestra.state import FailureKind, FailureState

logger = logging.getLogger("orchestra.engine.runner")

SEVERITIES = ("critical", "high", "medium", "low")
MIN_QUOTE_CHARS = 6
# Unity writes these; a reviewer cannot say anything useful about serialized YAML or binaries.
DEFAULT_EXCLUDES = ("*.meta", "*.prefab", "*.unity", "*.asset", "*.mat", "*.anim", "*.controller",
                    "*.png", "*.jpg", "*.psd", "*.fbx", "*.wav", "*.mp3", "*.ogg", "*.ttf", "*.lock")

_PROMPT = """You are reviewing a code change. Report only real defects: bugs, crashes, wrong \
results, data loss, race conditions, resource leaks, security problems, broken edge cases. \
Do not report style, naming, formatting, missing comments or personal preference.
{focus}
Work from the diff. Open another file only when a finding cannot be decided without it (for \
example to see how a changed method is called), and then only that file: do not survey the \
project. Every finding must be about a line this diff adds or changes.

Respond with ONLY a JSON object, no prose and no markdown fences:
{{"findings": [{{"severity": "critical|high|medium|low", "file": "path as in the diff", \
"line": 123, "quote": "the offending line copied exactly from the diff, without the leading + or space", \
"problem": "what goes wrong and with which input or state", "fix": "the concrete change", \
"confidence": 0.0}}]}}

Rules:
- At most {max_findings} findings, the most severe first. An empty list is a good answer when the change is sound.
- `quote` must be one line that appears in the diff. A finding whose quote is not in the diff is discarded.
- If you are not sure something is a defect, lower `confidence`; do not leave it out and do not overstate it.

[DIFF]
{diff}
"""


@dataclass
class Finding:
    severity: str
    file: str
    line: int
    quote: str
    problem: str
    fix: str = ""
    confidence: float = 0.5
    found_by: list[str] = field(default_factory=list)


def _section(key: str, default):
    return config.get("review_changes", key, default)


def reviewers() -> list[str]:
    """The two aliases that review a change ([review_changes] reviewers)."""
    personas = config.personas_config()
    default = [personas.get("implementer") or "cld-deep", personas.get("researcher") or "kimi-deep"]
    return [str(alias) for alias in (_section("reviewers", None) or default)]


def collect_diff(base: str = "HEAD", paths: Optional[list[str]] = None, root: Optional[Path] = None, *,
                 staged: bool = False) -> str:
    """The project's changes against ``base`` (default: everything not yet committed).

    staged: only what is staged for the next commit (`git diff --cached`), for the commit gate.
    """
    root = root or config.project_root()
    excludes = [f":(exclude){pattern}" for pattern in (_section("exclude", None) or DEFAULT_EXCLUDES)]
    against = ["--cached"] if staged else [base]
    command = ["git", "diff", "--no-color", "-U3", *against, "--", *(paths or ["."]), *excludes]
    done = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        raise ValueError(f"git diff failed: {done.stderr.strip()[:300]}")
    return done.stdout


# The defects Unity code has that a general review misses. Added for a change to C# files
# unless the caller gave a focus of their own; `focus="unity"` asks for it on any change.
FOCUS_PRESETS = {
    "unity": (
        "this is Unity C#. Check in particular: GetComponent / Find / FindObjectOfType / Camera.main "
        "called in Update, FixedUpdate, LateUpdate or a loop (cache it in Awake); allocations every "
        "frame (new collections, LINQ, string concatenation or format, boxing, closures) in hot paths; "
        "physics moved in Update instead of FixedUpdate, and Time.deltaTime missing from per-frame "
        "movement or used where fixedDeltaTime belongs; events, delegates and coroutines not "
        "unsubscribed or stopped in OnDisable/OnDestroy; a renamed [SerializeField] field without "
        "[FormerlySerializedAs] (its saved data is lost); a ScriptableObject asset changed at run time "
        "(the change persists in the editor); ?. or ?? on a UnityEngine.Object (they bypass Unity's "
        "null check); UnityEditor API outside an Editor folder or #if UNITY_EDITOR; async work that "
        "outlives its destroyed object."
    ),
}


def _focus_for(diff: str, focus: str) -> str:
    focus = focus.strip()
    if focus in FOCUS_PRESETS:
        return FOCUS_PRESETS[focus]
    if not focus and re.search(r"(?m)^\+\+\+ b/.*\.cs$", diff):
        return FOCUS_PRESETS["unity"]
    return focus


def build_prompt(diff: str, focus: str = "") -> str:
    focus = _focus_for(diff, focus)
    focus_line = f"Pay particular attention to: {focus}\n" if focus else ""
    from orchestra import skills

    named = skills.listing()
    if named:   # the project's rules: with them a review finds their violations (measured: F1 53% -> 65%)
        focus_line += "Check the changed lines against these project skills (load one with the Skill tool for its full text):\n" + \
            "".join(f"- {name}: {about}\n" for name, about in named)
    return _PROMPT.format(diff=diff, focus=focus_line, max_findings=int(_section("max_findings", 8)))


def _normal(text: str) -> str:
    return " ".join(text.split())


def diff_lines(diff: str) -> dict[str, list[str]]:
    """file -> the lines the diff shows for it (added and context), whitespace-normalised."""
    files: dict[str, list[str]] = {}
    current: Optional[str] = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            name = line[4:].strip()
            current = None if name == "/dev/null" else re.sub(r"^b/", "", name)
            if current:
                files.setdefault(current, [])
        elif current and line[:1] in ("+", " ") and not line.startswith("+++"):
            files[current].append(_normal(line[1:]))
    return files


def _match_file(name: str, files: dict[str, list[str]]) -> Optional[str]:
    name = name.strip().lstrip("./")
    if name in files:
        return name
    candidates = [path for path in files if path.endswith("/" + name) or name.endswith("/" + path)
                  or Path(path).name == Path(name).name]
    return candidates[0] if len(candidates) == 1 else None


def read_answer(text: str) -> Optional[list[dict]]:
    """The findings list out of an agent's answer, or None when the answer is not a review.

    [] and None are different things: [] is a reviewer saying the change is sound, None is
    an answer that cannot be read as one (prose, broken JSON, no findings list).
    """
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        try:
            import json_repair
            data = json_repair.loads(match.group(0))
        except Exception:
            return None
    findings = data.get("findings") if isinstance(data, dict) else None
    if not isinstance(findings, list):
        return None
    return [item for item in findings if isinstance(item, dict)]


def parse_findings(text: str) -> list[dict]:
    """The findings list out of an agent's answer; empty when there is none to be found."""
    return read_answer(text) or []


def _line_in_file(root: Path, file: str, quote: str, claimed: int) -> int:
    """Where the quoted line is in the file on disk: the occurrence nearest the claimed line."""
    try:
        lines = (root / file).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return claimed
    hits = [number for number, line in enumerate(lines, start=1) if quote in _normal(line)]
    return min(hits, key=lambda number: abs(number - claimed)) if hits else claimed


def check(raw: list[dict], alias: str, files: dict[str, list[str]], root: Path) -> tuple[list[Finding], list[dict]]:
    """Keep the findings whose quote is in the diff; return (kept, discarded with the reason)."""
    kept: list[Finding] = []
    discarded: list[dict] = []
    for item in raw:
        quote = _normal(str(item.get("quote") or "").splitlines()[0] if str(item.get("quote") or "").strip() else "")
        file = _match_file(str(item.get("file") or ""), files)
        reason = None
        if not file:
            reason = "file is not in the diff"
        elif len(quote) < MIN_QUOTE_CHARS:
            reason = "no usable quote"
        elif not any(quote in line for line in files[file]):
            reason = "quote is not in the diff"
        if reason:
            discarded.append({"alias": alias, "reason": reason, "file": item.get("file"), "problem": item.get("problem")})
            continue
        severity = str(item.get("severity") or "medium").lower()
        try:
            claimed = int(item.get("line") or 0)
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
        except (TypeError, ValueError):
            claimed, confidence = 0, 0.5
        assert file is not None   # a finding without a file was discarded above
        kept.append(Finding(
            severity=severity if severity in SEVERITIES else "medium",
            file=file,
            line=_line_in_file(root, file, quote, claimed),
            quote=quote,
            problem=str(item.get("problem") or "").strip(),
            fix=str(item.get("fix") or "").strip(),
            confidence=confidence,
            found_by=[alias],
        ))
    return kept, discarded


SAME_PLACE_LINES = 3


def _same_place(finding: Finding, merged: dict[tuple[str, int], Finding]) -> Optional[tuple[str, int]]:
    """The key of an already-merged finding about the same place.

    The same line, or the same quoted text a few lines away: when a statement is repeated
    (the defect is often the repetition), two reviewers point at different copies of it.
    """
    for key, known in merged.items():
        if known.file != finding.file:
            continue
        if known.line == finding.line:
            return key
        if known.quote == finding.quote and abs(known.line - finding.line) <= SAME_PLACE_LINES:
            return key
    return None


def merge(findings: list[Finding]) -> list[Finding]:
    """One finding per place: two reviewers pointing at the same line found the same thing."""
    merged: dict[tuple[str, int], Finding] = {}
    for finding in findings:
        key = _same_place(finding, merged) or (finding.file, finding.line)
        known = merged.get(key)
        if known is None:
            merged[key] = finding
            continue
        better = finding if SEVERITIES.index(finding.severity) < SEVERITIES.index(known.severity) else known
        merged[key] = Finding(
            severity=better.severity, file=better.file, line=better.line, quote=better.quote,
            problem=better.problem, fix=better.fix or known.fix or finding.fix,
            confidence=max(known.confidence, finding.confidence),
            found_by=sorted(set(known.found_by + finding.found_by)),
        )
    return sorted(merged.values(), key=lambda f: (SEVERITIES.index(f.severity), -len(f.found_by), f.file, f.line))


NOT_ANSWERED = "failed"
NOT_A_REVIEW = "answered without the findings format"


def absences(run: OrchestraRun) -> dict[str, str]:
    """Reviewers whose answer cannot count as a review, each with the reason."""
    absent: dict[str, str] = {}
    for agent in run.agents:
        if agent.status != AgentStatus.COMPLETED:
            absent[reviewer_name(agent)] = NOT_ANSWERED
        elif read_answer(agent.stdout_log or "") is None:
            absent[reviewer_name(agent)] = NOT_A_REVIEW
    return absent


def who_reviewed(run: OrchestraRun) -> tuple[list[str], list[str]]:
    """(reviewers whose answer was read as a review, reviewers whose answer was not)."""
    absent = absences(run)
    reviewed = [name for name in map(reviewer_name, run.agents) if name not in absent]
    return reviewed, list(absent)


def render(findings: list[Finding], discarded: list[dict], reviewer_aliases: list[str],
           absent: Optional[dict[str, str]] = None) -> str:
    absent = absent or {}
    if not reviewer_aliases:
        return "The change was not reviewed: " + "; ".join(f"{name} {why}" for name, why in absent.items()) + "."
    if not findings:
        text = f"No defects found by {' and '.join(reviewer_aliases)}."
    else:
        blocks = []
        for finding in findings:
            who = " + ".join(finding.found_by)
            agreed = " (both reviewers)" if len(finding.found_by) > 1 else ""
            blocks.append(
                f"**[{finding.severity.upper()}] {finding.file}:{finding.line}** — {who}{agreed}, "
                f"confidence {finding.confidence:.2f}\n`{finding.quote}`\n{finding.problem}"
                + (f"\nFix: {finding.fix}" if finding.fix else "")
            )
        text = "\n\n".join(blocks)
    if discarded:
        text += f"\n\n{len(discarded)} reported finding(s) were discarded because their quote or file is not in the diff."
    for name, why in absent.items():
        text += f"\n\n{name} did not review ({why})."
    return text


def reviewer_name(agent) -> str:
    """The alias, or the model that answered in its place when the alias fell back."""
    from orchestra.engine.parallel import _own_provider
    own = _own_provider(agent.alias)
    if own is not None and agent.provider and own.name != agent.provider:
        return f"{agent.model} (in place of {agent.alias})"
    return str(agent.alias)


def _ask_again_for_findings(run: OrchestraRun, prompt: str) -> None:
    """A reviewer that answered, but not with a findings object, is asked once more.

    Its second answer replaces the first when it can be read; the second call's cost is
    added to the reviewer.
    """
    from orchestra.engine import structured
    from orchestra.engine.runners.ask_runner import run_ask

    for agent in run.agents:
        if agent.status != AgentStatus.COMPLETED or read_answer(agent.stdout_log or "") is not None:
            continue
        again = structured.correction_prompt(prompt, agent.stdout_log or "", "it holds no findings object")
        try:
            retry = run_ask(agent.alias, again, internal=True, parent_run_id=run.run_id,
                            show_live=False, emit_console=False)
        except Exception:
            logger.warning("review: asking %s again failed", agent.alias, exc_info=True)
            continue
        answer = structured.first_output(retry)
        if read_answer(answer) is not None:
            agent.stdout_log = answer
            agent.estimated_cost_usd = round(
                (agent.estimated_cost_usd or 0.0) + sum(a.estimated_cost_usd or 0.0 for a in retry.agents), 6)


def run_review(diff: str, *, focus: str = "", title: str = "", aliases: Optional[list[str]] = None,
               root: Optional[Path] = None, **kwargs) -> tuple[OrchestraRun, list[Finding], list[dict]]:
    """Review ``diff`` with two agents and return (run, checked findings, discarded findings)."""
    from orchestra.engine.parallel import run_parallel
    from orchestra.engine.runners.common import _append_event_scrubbed, _finalize, show_in_status_line

    root = root or config.project_root()
    aliases = aliases or reviewers()
    files = diff_lines(diff)
    show_in_status_line()
    run = OrchestraRun(mode="review", task=title or f"Review of {len(files)} changed file(s)")
    run.status = RunStatus.RUNNING
    artifacts.write_manifest(run)
    _append_event_scrubbed(run.run_id, {"event": "run_started", "mode": "review", "files": sorted(files)})

    prompt = build_prompt(diff, focus)
    kwargs.setdefault("show_live", False)
    kwargs["install_signal_handlers"] = False
    run_parallel(run, [(alias, prompt) for alias in aliases], **kwargs)

    _ask_again_for_findings(run, prompt)
    kept: list[Finding] = []
    discarded: list[dict] = []
    models: list[str] = []
    absent = absences(run)
    reviewed, _ = who_reviewed(run)
    for agent in run.agents:
        name = reviewer_name(agent)
        if name in absent:
            continue
        models.append(agent.model)
        good, bad = check(parse_findings(agent.stdout_log or ""), name, files, root)
        kept.extend(good)
        discarded.extend(bad)
    findings = merge(kept)
    run.summary = render(findings, discarded, reviewed, absent)
    if not reviewed:
        # Nobody read the change: that is a failed review, not a clean one.
        unreadable = NOT_A_REVIEW in absent.values()
        run.status = RunStatus.FAILED
        run.failure = run.failure or FailureState(
            kind=(FailureKind.MODEL_SCHEMA_INVALID if unreadable else FailureKind.UNKNOWN_RUNTIME_ERROR).value,
            message=run.summary, retryable=True, source="code_review",
        )
    if len(models) > 1 and len(set(models)) == 1:
        run.summary += (f"\n\nBoth reviews came from the same model ({models[0]}): a reviewer could not "
                        "answer and its fallback did. Treat agreement between them as one opinion.")
    (artifacts.run_dir(run.run_id) / "findings.json").write_text(
        json.dumps({"findings": [asdict(f) for f in findings], "discarded": discarded}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _append_event_scrubbed(run.run_id, {"event": "review_checked", "findings": len(findings), "discarded": len(discarded)})
    _finalize(run)
    return run, findings, discarded

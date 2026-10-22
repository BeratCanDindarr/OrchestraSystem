"""Orchestra CLI — typer-based entry point."""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import typer
from orchestra import config

from rich.console import Console
from rich.table import Table

app = typer.Typer(name="orchestra", help="Multi-agent orchestration CLI.", add_completion=True)
console = Console()


def _print_json(payload: object) -> None:
    console.print_json(json.dumps(payload, ensure_ascii=False, indent=2))


_ALIAS_COST_PER_1K: dict[str, float] = {
    "cdx-deep": 0.012, "cdx-fast": 0.004,
    "gmn-pro":  0.006, "gmn-fast": 0.002,
    "cld-deep": 0.015, "cld-fast": 0.003,
    "oll-coder": 0.0,  "oll-fast": 0.0, "oll-analyst": 0.0,
}


def _announce_outcome(run):
    """Say what the agent table cannot: that a run is waiting for approval, or failed."""
    status = getattr(run.status, "value", run.status)
    from orchestra.engine.artifacts import read_events
    fallbacks = {e.get("alias"): e.get("model") for e in read_events(run.run_id) if e.get("event") == "agent_fallback"}
    for alias, model in fallbacks.items():
        console.print(f"[yellow]{alias} could not answer; {model} answered in its place.[/yellow]")
    if status == "waiting_approval":
        reasons = [e.get("reason") for e in read_events(run.run_id) if e.get("event") == "approval_requested"]
        reason = next((r for r in reversed(reasons) if r), "an approval policy asked for it")
        console.print(f"\n[yellow]Run {run.run_id} is waiting for approval[/yellow] ({reason}).")
        console.print(f"  approve: orchestra resume {run.run_id}    reject: orchestra resume {run.run_id} --reject")
    elif status == "failed":
        failure = getattr(run, "failure", None)
        message = getattr(failure, "message", None) or "no reason recorded"
        console.print(f"\n[red]Run {run.run_id} failed:[/red] {message}")
    return run


def _exit_unless_succeeded(run) -> None:
    """End the command with a non-zero status when the run failed or was cancelled."""
    if getattr(run.status, "value", run.status) in ("failed", "cancelled"):
        raise typer.Exit(1)


def _require_run(run_id: str) -> None:
    """End the command with an error when there is no such run.

    Looked up without `artifacts.run_dir`, which creates the folder it is asked about.
    """
    from rich.markup import escape

    from orchestra import config

    if not run_id or not (config.artifact_root() / run_id).is_dir():
        console.print(f"[red]Run {escape(run_id)} not found.[/red]")
        raise typer.Exit(1)


def _execute(mode: str, task: str, **kwargs):
    """Route and run a task in the terminal (live table, Ctrl-C cancels) through the service."""
    from rich.markup import escape

    from orchestra import service
    from orchestra.providers.fallback import UnknownAliasError

    try:
        run, route_class, routed_mode = service.execute_task(mode=mode, task=task, interactive=True, **kwargs)
    except (UnknownAliasError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1)
    if mode == "auto":
        console.print(f"[bold]\\[auto][/bold] {route_class} → {routed_mode}")
    return _announce_outcome(run)


def _dry_run_plan(mode: str, alias: Optional[str], task: str) -> dict:
    """Return routing estimate without executing anything."""
    estimated_tokens = int(len(task.split()) * 1.3) + 500
    cost_per_1k = _ALIAS_COST_PER_1K.get(alias or "", 0.004)
    return {
        "mode": mode,
        "alias": alias or "auto",
        "estimated_tokens": estimated_tokens,
        "estimated_cost_usd": round(cost_per_1k * estimated_tokens / 1000, 6),
    }


def _ensure_history_index() -> int:
    from orchestra.storage.db import backfill

    return backfill()


def _get_config_tree(*, include_env: bool = True):
    from orchestra import config

    source = config.raw_config() if include_env else (config.file_config() or config.raw_config())
    return dict(source)


def _config_at_path(tree: dict, path: str):
    cursor = tree
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(path)
        cursor = cursor[part]
    return cursor


def _set_config_path(tree: dict, path: str, value) -> None:
    parts = [part for part in path.split(".") if part]
    if not parts:
        raise ValueError("config path cannot be empty")
    cursor = tree
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, dict):
            existing = {}
            cursor[part] = existing
        cursor = existing
    cursor[parts[-1]] = value


def _cutoff_from_since(since: Optional[str]) -> Optional[datetime]:
    if not since:
        return None
    raw = since.strip().lower()
    if raw.endswith("h") and raw[:-1].isdigit():
        return datetime.now(timezone.utc) - timedelta(hours=int(raw[:-1]))
    if raw.endswith("d") and raw[:-1].isdigit():
        return datetime.now(timezone.utc) - timedelta(days=int(raw[:-1]))
    return None


def _render_logs(run_id: str, agent: Optional[str] = None) -> None:
    from orchestra.engine.artifacts import run_dir

    _require_run(run_id)
    agents_dir = run_dir(run_id) / "agents"

    found_any = False
    for log_file in sorted(agents_dir.glob("*.stdout.log")):
        alias = log_file.stem.replace(".stdout", "")
        if agent and alias != agent:
            continue
        found_any = True
        console.rule(f"[dim]{alias}[/dim]")
        from orchestra.engine.pii_scrubber import PiiScrubber
        console.print(PiiScrubber.scrub_text(log_file.read_text(encoding="utf-8")))

    if not found_any and agent:
        console.print(f"[yellow]No logs found for agent:[/yellow] {agent}")


def _watch_plain(run_id: str, interval: float) -> None:
    """Rich Live inline watch — works in any terminal including Claude Code."""
    from orchestra.engine.artifacts import run_dir
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    d = run_dir(run_id)
    manifest_file = d / "manifest.json"

    if not manifest_file.exists():
        console.print(f"[red]Run {run_id} not found.[/red]")
        raise typer.Exit(1)

    # Per-agent log tail state
    log_positions: dict[str, int] = {}
    log_lines: list[str] = []
    MAX_LOG_LINES = 18

    _STATUS_COLOR = {
        "queued": "dim", "pending": "dim",
        "started": "cyan", "running": "cyan",
        "completed": "green", "done": "green",
        "failed": "red", "cancelled": "dim",
    }
    _STATUS_ICON = {
        "queued": "○", "pending": "○",
        "started": "●", "running": "●",
        "completed": "✓", "done": "✓",
        "failed": "✗", "cancelled": "–",
    }

    def _read_manifest() -> dict:
        try:
            return json.loads(manifest_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _poll_logs(agents: list[dict]) -> None:
        for agent in agents:
            alias = agent.get("alias", "")
            log_path = d / "agents" / f"{alias}.stdout.log"
            if not log_path.exists():
                continue
            pos = log_positions.get(alias, 0)
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    new = f.readlines()
                    log_positions[alias] = f.tell()
                for raw in new:
                    line = raw.rstrip("\n")
                    if line:
                        log_lines.append(f"[dim]{alias}[/dim] {line}")
                        if len(log_lines) > MAX_LOG_LINES * 3:
                            del log_lines[:-MAX_LOG_LINES]
            except OSError:
                pass

    def _build_agent_table(manifest: dict) -> Table:
        t = Table(box=None, pad_edge=False, show_header=True, header_style="bold #888888")
        t.add_column("AGENT", style="bold", min_width=10)
        t.add_column("MODEL", style="dim", min_width=14)
        t.add_column("ST…", min_width=10)
        t.add_column("ELAPSED", min_width=7)
        t.add_column("COST", min_width=8)
        for agent in manifest.get("agents", []):
            status = agent.get("status", "?")
            color = _STATUS_COLOR.get(status, "white")
            icon = _STATUS_ICON.get(status, "?")
            cost = agent.get("estimated_cost_usd") or 0.0
            t.add_row(
                agent.get("alias", "?"),
                agent.get("model", "?"),
                Text(f"{icon} {status}", style=color),
                agent.get("elapsed", "--:--"),
                f"${cost:.4f}",
            )
        return t

    def _build_layout(manifest: dict) -> Layout:
        run_id_short = manifest.get("run_id", "?")[:8]
        mode = manifest.get("mode", "?")
        status = manifest.get("status", "?")
        status_color = _STATUS_COLOR.get(status, "white")
        total_cost = sum(
            (a.get("estimated_cost_usd") or 0.0) for a in manifest.get("agents", [])
        )

        header_text = Text.assemble(
            ("  Orchestra  ", "bold cyan"),
            (f"run {run_id_short}", "bold white"),
            ("  [", "dim"),
            (mode, "cyan"),
            ("]  ", "dim"),
            (status, status_color),
            (f"  ${total_cost:.4f}", "green"),
        )

        agent_table = _build_agent_table(manifest)
        log_text = Text.from_markup("\n".join(log_lines[-MAX_LOG_LINES:]) or "[dim]waiting for logs…[/dim]")

        layout = Layout()
        layout.split_column(
            Layout(Panel(header_text, style="on #1e1e1e", height=3), name="header", size=3),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(Panel(agent_table, title="[bold]agents[/bold]", border_style="#444444", style="on #1a1a1a"), name="left", ratio=2),
            Layout(Panel(log_text, title="[bold]stream[/bold]", border_style="#444444", style="on #1a1a1a"), name="right", ratio=3),
        )
        return layout

    try:
        with Live(refresh_per_second=2, screen=False) as live:
            while True:
                manifest = _read_manifest()
                _poll_logs(manifest.get("agents", []))
                live.update(_build_layout(manifest))
                if manifest.get("status") in ("completed", "failed", "cancelled"):
                    time.sleep(0.5)
                    live.update(_build_layout(manifest))
                    break
                time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching.[/dim]")


@app.command()
def ask(
    alias: str = typer.Argument(..., help="Agent alias: cdx-fast, cdx-deep, gmn-fast, gmn-pro, oll-fast, oll-deep"),
    task: str = typer.Argument(..., help="Task prompt to send"),
    raw: bool = typer.Option(False, "--raw", help="Print raw output instead of formatted"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show routing + cost estimate without executing"),
    fresh: bool = typer.Option(False, "--fresh", "--no-cache", help="Ask the model again instead of returning a stored answer"),
    follow: Optional[str] = typer.Option(None, "--follow", help="Run ID this question follows up: the agent is given that discussion"),
):
    """Send a single task to one agent alias."""
    if dry_run:
        plan = _dry_run_plan("ask", alias, task)
        console.print(f"[bold]\\[dry-run\\][/bold] mode=[cyan]{plan['mode']}[/cyan] alias=[cyan]{plan['alias']}[/cyan] "
                      f"~{plan['estimated_tokens']} tokens  ~${plan['estimated_cost_usd']:.5f}")
        return

    run = _execute("ask", task, alias=alias, fresh=fresh, follow_up_to=follow)
    if run.agents and run.agents[0].stdout_log and raw:
        from orchestra.engine.pii_scrubber import PiiScrubber
        console.print(PiiScrubber.scrub_text(run.agents[0].stdout_log))
    _exit_unless_succeeded(run)


@app.command()
def supervise(
    task: str = typer.Argument(..., help="Task to solve via hierarchical subagents"),
    max_depth: Optional[int] = typer.Option(None, "--max-depth", help="Max spawning depth (default: config)"),
    worker: Optional[str] = typer.Option(None, "--worker", help="Leaf worker alias (default: config)"),
    planner: Optional[str] = typer.Option(None, "--planner", help="Planner alias for decomposition (default: config)"),
    synthesizer: Optional[str] = typer.Option(None, "--synthesizer", help="Synthesizer alias (default: config)"),
    raw: bool = typer.Option(False, "--raw", help="Print raw final output"),
):
    """Decompose a task into subtasks, dispatch each to a subagent, then synthesize."""
    from orchestra.engine.supervisor import run_supervised

    run = run_supervised(
        task,
        max_depth=max_depth,
        worker_alias=worker,
        planner_alias=planner,
        synthesizer_alias=synthesizer,
    )
    output = run.summary or (run.agents[0].stdout_log if run.agents else "")
    if not output:
        console.print("[red]The supervised run produced no answer.[/red]")
        raise typer.Exit(1)
    if raw:
        console.print(output, markup=False)
    else:
        console.rule("[bold]Answer[/bold]")
        console.print(output, markup=False)


@app.command()
def run(
    mode: str = typer.Argument(..., help="Execution mode: auto | dual | consensus | planned"),
    task: str = typer.Argument(..., help="Task prompt"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show routing + cost estimate without executing"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ask the model again instead of returning a stored answer (single-agent runs)"),
    follow: Optional[str] = typer.Option(None, "--follow", help="Run ID this task follows up: the agents are given that discussion"),
):
    """Run auto-routed or explicit multi-agent modes."""
    if dry_run:
        if mode == "auto":
            from orchestra.router.classifier import route_task
            decision = route_task(task)
            routed = decision.mode
            alias = decision.alias
        else:
            routed = mode
            alias = None
        plan = _dry_run_plan(routed, alias, task)
        console.print(f"[bold]\\[dry-run\\][/bold] mode=[cyan]{plan['mode']}[/cyan] alias=[cyan]{plan['alias']}[/cyan] "
                      f"~{plan['estimated_tokens']} tokens  ~${plan['estimated_cost_usd']:.5f}")
        if mode == "auto":
            console.print(f"[dim]  router: {decision.reason} (confidence={decision.confidence:.2f})[/dim]")
        return

    if mode == "ask":
        console.print("[red]A single agent is asked with:[/red] orchestra ask <alias> <task>")
        raise typer.Exit(1)
    if mode not in ("auto", "dual", "consensus", "planned"):
        from rich.markup import escape
        console.print(f"[red]Unknown mode: {escape(mode)}[/red]")
        raise typer.Exit(1)
    run = _execute(mode, task, fresh=no_cache, follow_up_to=follow)
    if mode == "consensus" and run.summary:
        console.rule("[bold]Decision[/bold]")
        console.print(run.summary, markup=False)
    _exit_unless_succeeded(run)


@app.command("apply")
def apply_cmd(
    run_id: str = typer.Argument(..., help="Run whose changes to put into the project"),
    pick: bool = typer.Option(False, "--pick", help="Show each hunk and ask whether to apply it; the rest is "
                              "kept as rejected.patch in the run folder"),
):
    """Apply the changes a writing agent made in its isolated workspace ([isolation])."""
    if not pick:
        _settle_workspace_changes(run_id, apply=True)
        return
    from rich.markup import escape
    from rich.syntax import Syntax

    from orchestra.engine import isolation

    _require_run(run_id)

    def ask(hunk) -> bool:
        console.rule(f"[bold]{escape(hunk.file)}[/bold]")
        console.print(Syntax(hunk.body or hunk.header, "diff", theme="ansi_dark", word_wrap=True))
        return typer.confirm("Apply this hunk?", default=True)

    try:
        applied, kept = isolation.apply_picked(run_id, ask)
    except isolation.IsolationError as error:
        console.print(f"[red]{escape(str(error))}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Run {run_id}: {applied} applied, {kept} kept[/green]"
                  + (" (rejected.patch in the run folder)." if kept else "."))


@app.command("discard")
def discard_cmd(run_id: str = typer.Argument(..., help="Run whose changes to throw away")):
    """Throw away the changes a writing agent made in its isolated workspace."""
    _settle_workspace_changes(run_id, apply=False)


def _settle_workspace_changes(run_id: str, *, apply: bool) -> None:
    from rich.markup import escape

    from orchestra.engine import isolation

    _require_run(run_id)
    try:
        isolation.apply_changes(run_id) if apply else isolation.discard_changes(run_id)
    except isolation.IsolationError as error:
        console.print(f"[red]{escape(str(error))}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Changes of run {run_id} {'applied to the project' if apply else 'discarded'}.[/green]")


@app.command("forget")
def forget_cmd(
    run_ids: list[str] = typer.Argument(..., help="Run IDs (or 8-char prefixes) to erase"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation"),
):
    """Erase runs for good: folder, database rows, internal steps and memory row."""
    from orchestra.service import forget_runs

    if not yes and not typer.confirm(f"Erase {len(run_ids)} run(s) and their steps for good?"):
        raise typer.Exit(1)
    result = forget_runs(run_ids)
    console.print(f"Erased {len(result['forgotten'])} run(s) including steps.")
    for item in result["skipped"]:
        console.print(f"[yellow]left {item['run_id']}: {item['reason']}[/yellow]")


@app.command("review")
def review_cmd(
    paths: Optional[list[str]] = typer.Argument(None, help="Files or folders to review (default: every change)"),
    base: str = typer.Option("HEAD", "--base", help="Compare with this commit or branch (default: uncommitted changes)"),
    focus: str = typer.Option("", "--focus", help="What the reviewers should look at in particular; 'unity' for "
                              "the Unity checklist (used by itself when the change touches .cs files)"),
    json_output: bool = typer.Option(False, "--json", help="Return machine-readable JSON"),
    staged: bool = typer.Option(False, "--staged", help="Review only what is staged for the next commit"),
    fail_on: Optional[str] = typer.Option(None, "--fail-on", help="Exit 1 on a checked finding of this severity or "
                                          "worse: critical | high | medium | low (the commit gate)"),
    comment_pr: Optional[int] = typer.Option(None, "--comment-pr", help="Post the checked findings on this pull "
                                             "request (needs the gh CLI)"),
):
    """Review code changes with two agents; only findings found in the diff are reported."""
    from orchestra import service
    from orchestra.engine.code_review import SEVERITIES

    if fail_on is not None and fail_on not in SEVERITIES:
        console.print(f"[red]--fail-on must be one of: {', '.join(SEVERITIES)}[/red]")
        raise typer.Exit(2)
    try:
        result = service.review_changes(base=base, paths=paths or None, focus=focus, staged=staged)
    except ValueError as exc:
        if fail_on is not None and str(exc).startswith("Nothing to review"):
            return   # an empty commit has nothing to stop
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    if json_output:
        _print_json({"run_id": result["run"]["run_id"], "findings": result["findings"], "discarded": result["discarded"]})
    else:
        console.rule(f"[bold]Review {result['run']['run_id']}[/bold]")
        console.print(result["run"].get("summary") or "", markup=False)
    if comment_pr is not None and result["findings"]:
        _comment_on_pull_request(comment_pr, result)
    if fail_on is None:
        return
    if not result["reviewed_by"]:
        # The agents could not be reached: that is no verdict, and it must not lock every commit.
        console.print("[yellow]Not reviewed: no reviewer answered; the commit goes through.[/yellow]")
        return
    worst = SEVERITIES.index(fail_on)
    blocking = [f for f in result["findings"] if SEVERITIES.index(f["severity"]) <= worst]
    if blocking:
        console.print(f"[red]{len(blocking)} checked finding(s) at {fail_on} or worse: commit stopped "
                      "(git's no-verify option skips the gate).[/red]")
        raise typer.Exit(1)


def _comment_on_pull_request(number: int, result: dict) -> None:
    """One PR comment with the checked findings (each one's quote is in the diff)."""
    lines = [f"**Orchestra review** ({', '.join(result['reviewed_by'])}): "
             f"{len(result['findings'])} checked finding(s)", ""]
    for f in result["findings"]:
        lines.append(f"- **[{f['severity'].upper()}] `{f['file']}:{f['line']}`** {f['problem']}  \n  Fix: {f['fix']}")
    done = subprocess.run(["gh", "pr", "comment", str(number), "--body", "\n".join(lines)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        console.print(f"[red]Could not comment on PR {number}: {done.stderr.strip()[:300]}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Posted {len(result['findings'])} finding(s) on PR {number}.[/green]")


hook_app = typer.Typer(name="hook", help="The pre-commit review gate.")
app.add_typer(hook_app)


@hook_app.command("install")
def hook_install_cmd(
    fail_on: str = typer.Option("high", "--fail-on", help="Stop a commit on a checked finding this severe or worse"),
):
    """Install a git pre-commit hook that reviews the staged changes."""
    from orchestra.utils import git_hook

    try:
        path = git_hook.install(config.project_root(), fail_on=fail_on)
    except (FileExistsError, subprocess.CalledProcessError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Installed {path}[/green]: commits are reviewed; git's no-verify option skips it.")


@hook_app.command("uninstall")
def hook_uninstall_cmd():
    """Remove the pre-commit hook Orchestra installed (a hook someone else wrote is left alone)."""
    from orchestra.utils import git_hook

    removed = git_hook.uninstall(config.project_root())
    console.print("[green]Removed the review hook.[/green]" if removed else "[dim]No Orchestra hook installed.[/dim]")


@app.command("resume")
def resume_cmd(
    run_id: str = typer.Argument(..., help="Run ID to resume"),
    reject: bool = typer.Option(False, "--reject", help="Reject the paused run (fail it) instead of approving"),
):
    """Resume (or --reject) a paused/interrupted run. Works across ask/dual/planned modes."""
    from orchestra import service

    service.resume(run_id, reject=reject)


@app.command("job-worker", hidden=True)
def job_worker_cmd(job_id: str = typer.Argument(..., help="Job ID to execute")):
    """Internal command to execute a specific queued job."""
    from orchestra.service import execute_job

    execute_job(job_id)


@app.command()
def cancel(
    run_id: str = typer.Argument(..., help="Run ID to cancel"),
):
    """Cancel a running run: its agents are stopped, and so are the steps it started."""
    from orchestra.service import cancel_run

    _require_run(run_id)
    try:
        result = cancel_run(run_id)
    except ValueError:  # a folder without a manifest: nothing that could be running
        console.print(f"[red]Run {run_id} not found.[/red]")
        raise typer.Exit(1)
    if result.get("already_finished"):
        console.print(f"[dim]Run {run_id} had already ended ({result['run'].get('status')}); nothing to cancel.[/dim]")
        return
    console.print(f"[yellow]Run {run_id} cancelled.[/yellow]")


@app.command()
def ps(
    status: Optional[str] = typer.Option(None, "--filter", help="Status filter, e.g. failed"),
    since: Optional[str] = typer.Option(None, "--since", help="Relative time, e.g. 24h or 7d"),
    json_output: bool = typer.Option(False, "--json", help="Return machine-readable JSON"),
    all_runs: bool = typer.Option(False, "--all", help="Also list the steps of runs (verifier, synthesiser, planned nodes)"),
):
    """List recent runs."""
    from orchestra.engine.artifacts import list_runs
    from orchestra.service import reap_stale_runs

    reap_stale_runs()  # a run whose process died is shown as failed, not as running
    runs = list_runs(include_steps=all_runs)
    cutoff = _cutoff_from_since(since)
    if status:
        runs = [r for r in runs if r.get("status") == status]
    if cutoff is not None:
        filtered = []
        for run in runs:
            try:
                created = datetime.fromisoformat(run.get("created_at", ""))
            except ValueError:
                continue
            if created >= cutoff:
                filtered.append(run)
        runs = filtered
    if not runs:
        if json_output:
            _print_json({"runs": []})
        else:
            console.print("[dim]No runs found.[/dim]")
        return

    if json_output:
        _print_json({"runs": runs})
        return

    table = Table(title="Recent Runs", box=None, pad_edge=False)
    table.add_column("RUN ID", style="cyan")
    table.add_column("MODE", style="white")
    table.add_column("STATUS", style="white")
    table.add_column("CREATED", style="dim")
    table.add_column("TASK", style="white", no_wrap=True)

    for r in runs[:20]:
        status = r.get("status", "unknown")
        color = "white"
        if status == "completed":
            color = "green"
        elif status == "failed":
            color = "red"
        elif status == "running":
            color = "blue"

        table.add_row(
            r.get("run_id", "???"),
            r.get("mode", "???"),
            f"[{color}]{status}[/{color}]",
            r.get("created_at", "")[:19].replace("T", " "),
            r.get("task", "")[:60],
        )
    console.print(table)


@app.command()
def watch(
    run_id: str = typer.Argument(..., help="Run ID to watch live"),
    interval: float = typer.Option(1.0, "--interval", "-i", help="Refresh interval in seconds"),
):
    """Watch run events in real-time."""
    _watch_plain(run_id, interval)


@app.command()
def logs(
    run_id: str = typer.Argument(..., help="Run ID"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Filter by agent alias"),
    json_output: bool = typer.Option(False, "--json", help="Return machine-readable JSON"),
):
    """Print agent logs for a specific run."""
    if json_output:
        from orchestra.service import get_logs

        _print_json(get_logs(run_id, agent=agent, normalized=False))
        return
    _render_logs(run_id, agent)


# Command modules register on `app`; imported last because they import it from here.
from orchestra import cli_admin, cli_observe  # noqa: E402,F401

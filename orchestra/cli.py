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


from orchestra.utils.config_values import (
    parse_cli_value as _parse_cli_value,
    flatten_toml_lines as _flatten_toml_lines,
)


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
def apply_cmd(run_id: str = typer.Argument(..., help="Run whose changes to put into the project")):
    """Apply the changes a writing agent made in its isolated workspace ([isolation])."""
    _settle_workspace_changes(run_id, apply=True)


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
    focus: str = typer.Option("", "--focus", help="What the reviewers should look at in particular"),
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


# ─── Statusline Commands ─────────────────────────────────────────────────────

statusline_app = typer.Typer(name="statusline", help="Claude Code statusline integration.")
app.add_typer(statusline_app, name="statusline")


@statusline_app.command("render")
def statusline_render_cmd(
    project_dir: Optional[Path] = typer.Option(None, "--dir", help="Project directory (default: CWD)"),
    no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI colours"),
) -> None:
    """Preview the statusline output without Claude Code."""
    import os
    from orchestra.statusline import main as _statusline_main

    start_dir = project_dir if project_dir is not None else Path(os.getcwd())
    argv = ["--dir", str(start_dir)]
    if no_color:
        argv.append("--no-color")
    _statusline_main(argv)


@statusline_app.command("install")
def statusline_install_cmd(
    path: Optional[Path] = typer.Option(None, "--path", help="Explicit settings.json path"),
    user: bool = typer.Option(False, "--user", help="Target ~/.claude/settings.json"),
    interval: int = typer.Option(2, "--interval", min=1, help="Refresh interval in seconds (min 1)"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing statusLine key"),
) -> None:
    """Write statusLine config into a Claude Code settings.json (non-destructive by default)."""
    import sys, json  # noqa: E401
    from datetime import datetime
    from orchestra.utils.statusline_install import (
        ConflictError, InvalidSettingsError,
        build_statusline_block, load_existing_settings, merge_settings, resolve_target,
    )

    try:
        target = resolve_target(str(path) if path is not None else None, user)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    try:
        existing = load_existing_settings(target)
    except InvalidSettingsError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    block = build_statusline_block(sys.executable, interval)

    try:
        result = merge_settings(existing, block, force=force)
    except ConflictError as exc:
        console.print(f"[yellow]statusLine already configured in {target}:[/yellow]")
        console.print_json(json.dumps(exc.existing_value, indent=2))
        console.print()
        console.print("[yellow]Would write:[/yellow]")
        console.print_json(json.dumps(block, indent=2))
        console.print()
        console.print("Re-run with [bold]--force[/bold] to overwrite.")
        raise typer.Exit(1)

    if result.was_overwritten:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = target.parent / f"{target.name}.bak-{timestamp}"
        backup_path.write_bytes(target.read_bytes())

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    console.print(f"[green]Installed statusline:[/green] {block['command']}")
    console.print("Restart Claude Code or start a new session to see it.")


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


@app.command()
def aliases(
    json_output: bool = typer.Option(False, "--json", help="Return machine-readable JSON"),
):
    """Show available agent aliases and their status."""
    from orchestra.providers.fallback import available_aliases

    if json_output:
        _print_json({"aliases": available_aliases()})
        return

    console.print("\nAvailable aliases:")
    for alias in available_aliases():
        console.print(f"  {alias}")


def _fmt_size(nbytes: int) -> str:
    nbytes = nbytes or 0
    gb = nbytes / 1_000_000_000
    return f"{gb:.1f} GB" if gb >= 0.1 else f"{nbytes} B"


@app.command()
def models(
    refresh: bool = typer.Option(False, "--refresh", help="Bypass the discovery cache"),
    json_output: bool = typer.Option(False, "--json", help="Return machine-readable JSON"),
):
    """List Ollama models available on each configured backend."""
    from orchestra.providers import ollama_backends

    discovery = ollama_backends.discover_all(refresh=refresh)
    if json_output:
        _print_json(discovery)
        return
    for name, info in discovery.items():
        status = "[green]✓ online[/green]" if info["online"] else "[red]✗ offline[/red]"
        console.print(f"\n[bold]{name}[/bold] ({info['base_url']})  {status}")
        for m in info["models"]:
            console.print(f"  {m['name']:<28} {_fmt_size(m['size'])}")
        if info["online"] and not info["models"]:
            console.print("  [dim](no models installed)[/dim]")


@app.command()
def backends():
    """Show configured Ollama backends and whether they are reachable."""
    from orchestra.providers import ollama_backends

    discovery = ollama_backends.discover_all()
    for name, info in discovery.items():
        status = "[green]✓ reachable[/green]" if info["online"] else "[red]✗ unreachable[/red]"
        count = len(info["models"])
        console.print(f"  {name:<14} {info['base_url']:<32} {status}  ({count} models)")


@app.command()
def backfill():
    """Backfill SQLite event store from JSONL artifacts."""
    processed = _ensure_history_index()
    console.print(f"[green]Backfilled {processed} runs into SQLite.[/green]")


@app.command()
def version():
    """Show Orchestra version."""
    from orchestra import __version__

    console.print(f"Orchestra v{__version__}")


# ─── Config Commands ────────────────────────────────────────────────────────


config_app = typer.Typer(name="config", help="Config file and env override management.")
app.add_typer(config_app)


@config_app.command("get")
def config_get_cmd(
    key: str = typer.Argument(..., help="Dotted config path, e.g. review.tie_margin"),
):
    """Read a config value after env overrides are applied."""
    try:
        value = _config_at_path(_get_config_tree(include_env=True), key)
    except KeyError:
        console.print(f"[red]Config key not found:[/red] {key}")
        raise typer.Exit(1)

    if isinstance(value, dict):
        console.print_json(json.dumps(value, ensure_ascii=False, indent=2))
    else:
        console.print(value)


@config_app.command("set")
def config_set_cmd(
    key: str = typer.Argument(..., help="Dotted config path, e.g. routing.large_task_chars"),
    value: str = typer.Argument(..., help="New value. Lists use [a,b,c]."),
):
    """Set a config value in .orchestra/config.toml."""
    from orchestra import config

    path = config.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = _get_config_tree(include_env=False)
    _set_config_path(tree, key, _parse_cli_value(value))
    content = "\n".join(_flatten_toml_lines(tree)).strip() + "\n"
    path.write_text(content, encoding="utf-8")
    config.reload_config()
    console.print(f"[green]Updated[/green] {key} -> {value}")


@config_app.command("validate")
def config_validate_cmd():
    """Validate the active config file."""
    from orchestra import config

    config.reload_config()
    error = config.config_error()
    if error:
        console.print(f"[red]Invalid config:[/red] {error}")
        raise typer.Exit(1)
    console.print(f"[green]Valid[/green] {config.config_path()}")


@config_app.command("init")
def config_init_cmd(
    force: bool = typer.Option(False, "--force", help="Overwrite existing config.toml"),
):
    """Create a documented .orchestra/config.toml if missing."""
    from orchestra import config

    path = config.config_path()
    if path.exists() and not force:
        console.print(f"[yellow]Config already exists:[/yellow] {path}")
        raise typer.Exit(1)

    path.parent.mkdir(parents=True, exist_ok=True)
    template = "\n".join(_flatten_toml_lines(_get_config_tree(include_env=False))).strip() + "\n"
    path.write_text(template, encoding="utf-8")
    config.reload_config()
    console.print(f"[green]Initialized[/green] {path}")


# ─── Memory Commands ────────────────────────────────────────────────────────

memory_app = typer.Typer(name="memory", help="Semantic memory (RAG) management.")
app.add_typer(memory_app)

@memory_app.command("add")
def memory_add(
    content: str = typer.Argument(..., help="Content to add to memory"),
    meta: Optional[str] = typer.Option(None, "--meta", help="Optional JSON metadata"),
):
    """Add a fact or snippet to semantic memory."""
    from orchestra.storage.memory import get_memory
    
    metadata = json.loads(meta) if meta else {}
    mem = get_memory()
    if mem.add(content, metadata):
        console.print("[green]Added to memory.[/green]")
    else:
        console.print("[red]Failed to add to memory (embedding error).[/red]")

@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, "--limit", help="Max results"),
    tasks: bool = typer.Option(False, "--tasks", help="Also show the task text stored for earlier runs"),
):
    """Search semantic memory for relevant snippets."""
    from orchestra.storage.memory import get_memory, search_notes

    mem = get_memory()
    results = mem.search(query, limit=limit) if tasks else search_notes(mem, query, limit=limit)
    if not results:
        console.print("[dim]No relevant results found.[/dim]")
        return
        
    table = Table(title="Semantic Memory Results", box=None)
    table.add_column("SIMILARITY", width=12)
    table.add_column("CONTENT", style="dim")
    
    for res in results:
        table.add_row(f"{res['similarity']:.3f}", res['content'].strip()[:200] + "...")
        
    console.print(table)

@memory_app.command("index-project")
def memory_index_project(
    path: str = typer.Argument(".", help="Path to index"),
    glob: str = typer.Option("**/*.md", "--glob", help="Glob pattern for files"),
):
    """Index project files into semantic memory with dependency graph (Project Atlas)."""
    from pathlib import Path
    import re
    from orchestra.storage.memory import get_memory
    
    mem = get_memory()
    root = Path(path)
    files = list(root.glob(glob))
    
    if not files:
        console.print(f"[yellow]No files found matching {glob} in {path}[/yellow]")
        return
        
    console.print(f"[bold]Indexing {len(files)} files into Project Atlas...[/bold]")
    with typer.progressbar(files) as progress:
        for f in progress:
            if f.is_file():
                try:
                    content = f.read_text(encoding="utf-8")
                    if not content.strip(): continue
                    
                    if len(content) > 2000:
                        chunks = [content[i:i+2000] for i in range(0, len(content), 1800)]
                        for i, chunk in enumerate(chunks):
                            mem.add(chunk, {"path": str(f), "chunk": i})
                    else:
                        mem.add(content, {"path": str(f)})
                        
                    # Extract dependencies for C# files (Project Atlas)
                    if str(f).endswith(".cs"):
                        usings = re.findall(r'using\s+([a-zA-Z0-9_.]+);', content)
                        for u in usings:
                            if not u.startswith("System") and not u.startswith("UnityEngine"):
                                mem.add_relation(str(f), u, "imports_namespace")
                                
                except Exception as e:
                    console.print(f"[dim]Failed to index {f}: {e}[/dim]")
                    
    console.print(f"[green]Project Atlas indexing complete.[/green]")


# ─── Trace Commands ─────────────────────────────────────────────────────────

@app.command("trace")
def trace_cmd(
    run_id: str = typer.Argument(..., help="Run ID to trace"),
    json_output: bool = typer.Option(False, "--json", help="Return machine-readable JSON"),
):
    """Show detailed execution trace for a run."""
    from rich.tree import Tree
    from orchestra.engine.artifacts import read_events

    _require_run(run_id)
    events = read_events(run_id)
    if not events:
        console.print(f"[red]No events found for run {run_id}.[/red]")
        raise typer.Exit(1)
        
    spans = [e for e in events if e.get("event") == "span_finished"]
    if not spans:
        console.print(f"[yellow]No trace spans found. Tracing was recently added.[/yellow]")
        return

    if json_output:
        _print_json({"run_id": run_id, "spans": spans})
        return
        
    tree = Tree(f"[bold cyan]Run {run_id} Trace[/bold cyan]")
    
    # Simple flat tree for now, can be hierarchical if parent_id is used
    for span in spans:
        status_color = "green" if span["status"] == "completed" else "red"
        duration = f"{span['duration_ms']:.1f}ms"
        meta = span.get("metadata", {})
        info = []
        if "tokens" in meta: info.append(f"tokens:{meta['tokens']}")
        if "cost" in meta: info.append(f"cost:{meta['cost']:.4f}$")
        
        info_str = f" [dim]({', '.join(info)})[/dim]" if info else ""
        tree.add(
            f"{span['name']} [{status_color}]{span['status']}[/{status_color}] "
            f"[bold]{duration}[/bold]{info_str}"
        )
        
    console.print(tree)


# ─── Eval Commands ──────────────────────────────────────────────────────────


@app.command("eval")
def eval_cmd(
    task: str = typer.Argument(..., help="Task to evaluate (20–5000 characters)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without executing"),
) -> None:
    """Evaluate a task using ask/dual/consensus modes and judge quality.

    Runs three modes (ask/dual/consensus) in parallel, judges each output on
    quality/grounding/conciseness, and returns ranked results by composite score.
    """
    from orchestra import service
    from orchestra.engine.eval_judge import task_hash

    # Validate task length
    if len(task) < 20:
        console.print("[red]Error: task must be 20+ characters[/red]")
        raise typer.Exit(1)
    if len(task) > 5000:
        console.print("[red]Error: task must be ≤5000 characters[/red]")
        raise typer.Exit(1)

    # Compute task_hash for storage
    task_id = task_hash(task)

    # Dry-run plan
    if dry_run:
        console.print(f"\n[bold cyan][dry-run] Evaluation Plan[/bold cyan]\n")
        console.print(f"Task hash: [cyan]{task_id}[/cyan]")
        console.print(f"Task: {task[:100]}...")
        console.print(f"\nModes to run:")
        console.print(f"  1. ask      — single fast agent ({config.default_light_alias()})")
        console.print("  2. dual     — two-agent synthesis (ask + judge)")
        console.print("  3. consensus — for, against and neutral agents, then a decision")
        console.print(f"\nEstimated cost: $0.03–$0.15 total")
        console.print(f"  - Execution: ~$0.01–$0.10")
        console.print(f"  - Judge (3x): ~$0.01")
        console.print(f"\nResults will be saved to: ~/.orchestra/evals/{task_id}.jsonl")
        return

    console.print(f"\n[bold cyan]Evaluating:[/bold cyan] {task[:100]}...\n")
    ordered = service.evaluate_modes(task, note=console.print)["results"]

    if json_output:
        _print_json({"task": task, "task_hash": task_id, "results": ordered})
        return

    # Terminal: rich table output
    table = Table(title="[bold]LLM-as-Judge Evaluation[/bold]", box=None, pad_edge=False)
    table.add_column("MODE", style="cyan", min_width=9)
    table.add_column("STATUS", style="white", min_width=9)
    table.add_column("CONF", style="green", min_width=6)
    table.add_column("COST", style="yellow", min_width=8)
    table.add_column("JUDGE", style="magenta", min_width=8)
    table.add_column("$/QUALITY", style="dim", min_width=10)
    table.add_column("REASON", style="dim", max_width=35)

    for i, res in enumerate(ordered):
        judge = res.get("judge") or {}
        mode = res["mode"]
        status = res["status"]
        confidence = res["confidence"]
        cost = res["cost_usd"]
        judge_score = float(judge.get("score") or 0)
        cost_per_quality = res.get("cost_per_quality", 0)
        reasoning = str(judge.get("reasoning", ""))[:35]

        # Winner indicator
        winner = " ⭐" if i == 0 and judge_score > 0 else ""

        table.add_row(
            mode + winner,
            status,
            f"{confidence:.2f}",
            f"${cost:.4f}",
            f"{judge_score:.2f}",
            f"${cost_per_quality:.5f}",
            reasoning,
        )

    console.print(table)

    # Summary footer
    best_mode = ordered[0]["mode"] if ordered else "—"
    total_cost = sum(r.get("cost_usd", 0) + r.get("judge_cost_usd", 0) for r in ordered)
    console.print(
        f"\n[dim]Best quality: [cyan]{best_mode}[/cyan] | "
        f"Total cost: [yellow]${total_cost:.4f}[/yellow] | "
        f"Saved to: [cyan]~/.orchestra/evals/{task_id}.jsonl[/cyan][/dim]"
    )


# ─── Observability Commands ──────────────────────────────────────────────────


@app.command("timeline")
def timeline_cmd(
    run_id: str = typer.Argument(..., help="Run ID or 8-char prefix"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Display Gantt-style timeline of agent execution phases and timing."""
    from orchestra.engine.observability import (
        resolve_run_id,
        load_manifest,
        load_events,
        enrich_timeline_events,
        format_duration,
    )
    from rich.tree import Tree

    # Resolve run_id prefix
    resolved_id = resolve_run_id(run_id)
    if not resolved_id:
        if (run_id_prefix := run_id).startswith("/"):
            console.print(f"[red]Error: run_id prefix '{run_id}' is ambiguous[/red]")
        else:
            console.print(f"[red]Error: run_id '{run_id}' not found[/red]")
        raise typer.Exit(1)

    # Load manifest and events
    manifest = load_manifest(resolved_id)
    if not manifest:
        console.print(f"[red]Error: run_id '{resolved_id}' not found[/red]")
        raise typer.Exit(1)

    events = load_events(resolved_id)
    if not events:
        console.print(f"[red]Error: no events found for run '{resolved_id}'[/red]")
        raise typer.Exit(1)

    # Enrich events with timeline metadata
    enriched_events = enrich_timeline_events(events)

    if json_output:
        _print_json({"run_id": resolved_id, "events": enriched_events})
        return

    # Build Gantt-style tree grouped by phase and agent
    created_at = manifest.created_at or "2026-01-01T00:00:00Z"
    try:
        dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError, TypeError):
        dt_str = str(created_at)

    title = f"[bold cyan]Timeline: {resolved_id}[/bold cyan] ({dt_str})"
    tree = Tree(title)

    # Group events by phase_id, then by agent alias
    phases: dict[Optional[str], dict[Optional[str], list[dict]]] = {}
    for ev in enriched_events:
        phase_id = ev.get("phase_id")
        alias = ev.get("alias")
        if phase_id not in phases:
            phases[phase_id] = {}
        if alias not in phases[phase_id]:
            phases[phase_id][alias] = []
        phases[phase_id][alias].append(ev)

    # Render phases
    for phase_id in sorted((p for p in phases.keys() if p is not None), key=str) + [None]:
        agents_dict = phases.get(phase_id, {})

        # Phase header
        phase_label = f"phase_{phase_id}" if phase_id else "[dim]ungrouped[/dim]"
        phase_events = [e for alias_list in agents_dict.values() for e in alias_list]
        if phase_events:
            first_ts = phase_events[0].get("ts_sec", 0.0)
            last_ts = phase_events[-1].get("ts_sec", 0.0)
            duration = last_ts - first_ts
            dur_str = format_duration(duration)
            phase_node = tree.add(f"{phase_label} [{dur_str}]")
        else:
            phase_node = tree.add(phase_label)

        # Render agents within phase
        for alias in sorted((a for a in agents_dict.keys() if a), key=str) + [None]:
            agent_events = agents_dict.get(alias, [])
            if not agent_events:
                continue

            alias_label = alias or "[dim]no-agent[/dim]"
            agent_events_sorted = sorted(agent_events, key=lambda e: e.get("ts_sec", 0.0))

            # Agent timing
            first_ts = agent_events_sorted[0].get("ts_sec", 0.0)
            last_ts = agent_events_sorted[-1].get("ts_sec", 0.0)
            duration = last_ts - first_ts
            dur_str = format_duration(duration)

            # Determine status from events
            event_types = {e.get("event") for e in agent_events_sorted}
            if "agent_failed" in event_types:
                status_icon = "✗"
                status_color = "red"
            elif "agent_completed" in event_types:
                status_icon = "✓"
                status_color = "green"
            else:
                status_icon = "–"
                status_color = "dim"

            agent_node = phase_node.add(
                f"{alias_label} [{status_color}]{status_icon}[/{status_color}] {dur_str}"
            )

            # Render individual steps/events
            for ev in agent_events_sorted:
                ev_type = ev.get("event", "?")
                ts_sec = ev.get("ts_sec", 0.0)
                ts_str = format_duration(ts_sec)

                # Event detail
                parts = []
                if ev_type == "agent_completed" and "confidence" in ev:
                    parts.append(f"conf={ev['confidence']:.2f}")
                if "elapsed" in ev:
                    parts.append(f"dur={ev['elapsed']}")
                if ev.get("error"):
                    parts.append(f"err={str(ev['error'])[:40]}")

                detail = " ".join(parts) if parts else ""
                event_label = f"[dim]{ts_str}[/dim] {ev_type}"
                if detail:
                    event_label += f" {detail}"
                agent_node.add(event_label)

    console.print(tree)

    # Summary footer
    conf = float(manifest.avg_confidence or 0)
    cost = float(manifest.total_cost_usd or 0)
    mode = manifest.mode or "?"
    status = getattr(manifest.status, "value", manifest.status) or "?"
    console.print(f"\n[dim]mode={mode}  status={status}  confidence={conf:.2f}  cost=${cost:.4f}[/dim]")


@app.command("diff")
def diff_cmd(
    run_id_a: str = typer.Argument(..., help="First run ID or 8-char prefix"),
    run_id_b: str = typer.Argument(..., help="Second run ID or 8-char prefix"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Compare two runs: cost, confidence, agent count, synthesis output."""
    from orchestra.engine.observability import (
        resolve_run_id,
        load_manifest,
    )

    # Resolve run IDs from prefixes
    resolved_a = resolve_run_id(run_id_a)
    resolved_b = resolve_run_id(run_id_b)

    if not resolved_a:
        console.print(f"[red]Error: run_id '{run_id_a}' not found[/red]")
        raise typer.Exit(1)
    if not resolved_b:
        console.print(f"[red]Error: run_id '{run_id_b}' not found[/red]")
        raise typer.Exit(1)

    # Check if both are the same run
    if resolved_a == resolved_b:
        console.print(f"[red]Error: run_a and run_b are the same ({resolved_a})[/red]")
        raise typer.Exit(1)

    # Load manifests
    manifest_a = load_manifest(resolved_a)
    manifest_b = load_manifest(resolved_b)

    if not manifest_a:
        console.print(f"[red]Error: run_id '{resolved_a}' not found[/red]")
        raise typer.Exit(1)
    if not manifest_b:
        console.print(f"[red]Error: run_id '{resolved_b}' not found[/red]")
        raise typer.Exit(1)

    # Extract comparison fields from OrchestraRun models
    cost_a = manifest_a.total_cost_usd or 0.0
    cost_b = manifest_b.total_cost_usd or 0.0
    conf_a = manifest_a.avg_confidence or 0.0
    conf_b = manifest_b.avg_confidence or 0.0
    mode_a = manifest_a.mode or "?"
    mode_b = manifest_b.mode or "?"
    status_a = manifest_a.status.value if manifest_a.status else "?"
    status_b = manifest_b.status.value if manifest_b.status else "?"
    agents_a = len(manifest_a.agents) if manifest_a.agents else 0
    agents_b = len(manifest_b.agents) if manifest_b.agents else 0
    winner_a = manifest_a.latest_review_winner or ""
    winner_b = manifest_b.latest_review_winner or ""

    # Compute deltas
    delta_cost = cost_b - cost_a
    delta_conf = conf_b - conf_a
    delta_agents = agents_b - agents_a

    # Compute percentages
    percent_cost_delta = (delta_cost / cost_a * 100) if cost_a > 0 else 0
    percent_agents_delta = (delta_agents / agents_a * 100) if agents_a > 0 else 0

    if json_output:
        _print_json({
            "run_a": {
                "run_id": resolved_a,
                "mode": mode_a,
                "status": status_a,
                "confidence": round(conf_a, 4),
                "cost_usd": round(cost_a, 6),
                "agents": agents_a,
                "winner": winner_a,
            },
            "run_b": {
                "run_id": resolved_b,
                "mode": mode_b,
                "status": status_b,
                "confidence": round(conf_b, 4),
                "cost_usd": round(cost_b, 6),
                "agents": agents_b,
                "winner": winner_b,
            },
            "delta": {
                "cost_usd": round(delta_cost, 6),
                "confidence": round(delta_conf, 4),
                "agents": delta_agents,
                "cost_percent": round(percent_cost_delta, 1),
            },
        })
        return

    # Terminal rendering: side-by-side table
    short_a = resolved_a[:8]
    short_b = resolved_b[:8]

    table = Table(
        title=f"[bold]Diff: {short_a} vs {short_b}[/bold]",
        box=None,
        show_header=True,
    )
    table.add_column("FIELD", style="cyan", min_width=14)
    table.add_column(f"A {short_a}", style="white", min_width=12)
    table.add_column(f"B {short_b}", style="white", min_width=12)
    table.add_column("DELTA", style="bold", min_width=14)

    # Helper to format status with color
    def _status_fmt(s: str) -> str:
        return f"[green]{s}[/green]" if s == "completed" else f"[red]{s}[/red]"

    # Helper to format delta with sign and color
    def _delta_fmt(value: float, is_cost: bool = False, is_percent: bool = False) -> str:
        """Format delta with appropriate color and sign.
        For cost: red if positive (bad), green if negative (good).
        For confidence: green if positive (good), red if negative (bad).
        """
        sign_str = f"+{value:.4f}" if value >= 0 else f"{value:.4f}"
        if is_cost:
            color = "red" if value > 0 else "green"
        else:
            color = "green" if value >= 0 else "red"
        return f"[{color}]{sign_str}[/{color}]"

    # Cost row
    table.add_row(
        "Cost (USD)",
        f"${cost_a:.4f}",
        f"${cost_b:.4f}",
        _delta_fmt(delta_cost, is_cost=True) + f" ({percent_cost_delta:+.0f}%)" if cost_a > 0 else _delta_fmt(delta_cost, is_cost=True),
    )

    # Confidence row
    table.add_row(
        "Confidence",
        f"{conf_a:.2f}",
        f"{conf_b:.2f}",
        _delta_fmt(delta_conf),
    )

    # Mode row
    table.add_row(
        "Mode",
        mode_a,
        mode_b,
        "✓" if mode_a == mode_b else "different",
    )

    # Status row
    table.add_row(
        "Status",
        _status_fmt(status_a),
        _status_fmt(status_b),
        "✓" if status_a == status_b else "different",
    )

    # Agents count row
    agents_delta_str = (
        _delta_fmt(float(delta_agents)) if delta_agents != 0
        else "="
    )
    table.add_row(
        "Agents",
        f"{agents_a}",
        f"{agents_b}",
        agents_delta_str,
    )

    # Review winner row
    table.add_row(
        "Review Winner",
        winner_a or "—",
        winner_b or "—",
        "✓" if winner_a == winner_b else "different",
    )

    console.print(table)

    # Summary section (synthesis output comparison)
    summary_a = (manifest_a.summary or "").strip()
    summary_b = (manifest_b.summary or "").strip()

    if summary_a or summary_b:
        console.rule("[dim]Synthesis Comparison[/dim]")
        if summary_a:
            console.rule("[dim]Run A Synthesis[/dim]")
            console.print(summary_a[:500] or "[dim](none)[/dim]")
        if summary_b:
            console.rule("[dim]Run B Synthesis[/dim]")
            console.print(summary_b[:500] or "[dim](none)[/dim]")


@app.command("replay")
def replay_cmd(
    run_id: str = typer.Argument(..., help="Run ID or 8-char prefix"),
    synthesizer: Optional[str] = typer.Option(None, "--synthesizer", "-s", help="Synthesizer alias (default: the configured synthesiser)"),
    json_output: bool = typer.Option(False, "--json", help="Return machine-readable JSON"),
) -> None:
    """Re-synthesize a run from cached agent outputs (no re-execution).

    Loads agent outputs from an existing run and re-synthesizes them using the specified
    synthesizer. Useful for testing different synthesis models or strategies without
    re-running agents. Compares cost and confidence between original and replayed synthesis.
    """
    from orchestra import service

    console.print(f"[bold cyan]Replaying: {run_id} using synthesizer={synthesizer or 'the configured one'}[/bold cyan]\n")
    try:
        result = service.replay_synthesis(run_id, synthesizer)
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)
    resolved_id, synthesizer = result["run_id"], result["synthesizer"]
    original_output, replayed_output = result["original_output"], result["replayed_output"]
    original_cost, replayed_cost = result["original_cost_usd"], result["replayed_cost_usd"]
    original_confidence, replayed_confidence = result["original_confidence"], result["replayed_confidence"]
    completed_meta = result["agents"]
    agents_str = ", ".join(a.get("alias", "?") for a in completed_meta)

    if json_output:
        _print_json({
            "run_id": resolved_id,
            "synthesizer": synthesizer,
            "replay_run_id": result["replay_run_id"],
            "original_output": original_output[:500],
            "replayed_output": replayed_output[:500],
            "original_cost_usd": round(original_cost, 6),
            "replayed_cost_usd": round(replayed_cost, 6),
            "cost_savings_usd": round(original_cost - replayed_cost, 6),
            "original_confidence": round(original_confidence, 3),
            "replayed_confidence": round(replayed_confidence, 3),
            "confidence_delta": round(replayed_confidence - original_confidence, 3),
            "agents": completed_meta,
        })
        return

    # Terminal: side-by-side original vs replayed
    console.print("[bold cyan]Original Synthesis:[/bold cyan]")
    console.rule("────────────────────────────────────────")
    console.print(original_output[:500] + ("..." if len(original_output) > 500 else ""))
    console.rule("────────────────────────────────────────")

    console.print()
    console.print(f"[bold cyan]Replayed Synthesis ({synthesizer}):[/bold cyan]")
    console.rule("────────────────────────────────────────")
    console.print(replayed_output[:500] + ("..." if len(replayed_output) > 500 else ""))
    console.rule("────────────────────────────────────────")

    # Summary stats
    cost_percent_savings = (
        round((original_cost - replayed_cost) / original_cost * 100, 1)
        if original_cost > 0 else 0.0
    )
    confidence_percent_delta = round((replayed_confidence - original_confidence) * 100, 0)

    console.print()
    console.print(
        f"[bold]Cost Savings:[/bold] ${original_cost:.4f} → ${replayed_cost:.4f} "
        f"([green]-{cost_percent_savings}%[/green])" if cost_percent_savings > 0
        else f"[bold]Cost Savings:[/bold] ${original_cost:.4f} → ${replayed_cost:.4f} "
        f"([red]+{abs(cost_percent_savings)}%[/red])"
    )
    console.print(
        f"[bold]Confidence:[/bold] {original_confidence:.2f} → {replayed_confidence:.2f} "
        f"([green]+{confidence_percent_delta}%[/green])" if confidence_percent_delta >= 0
        else f"[bold]Confidence:[/bold] {original_confidence:.2f} → {replayed_confidence:.2f} "
        f"([red]{confidence_percent_delta}%[/red])"
    )
    console.print(f"[bold]Agents:[/bold] {len(completed_meta)} ({agents_str})")

@app.command("checkpoints")
def list_checkpoints_cmd(
    run_id: str = typer.Argument(...),
    json_output: bool = typer.Option(False, "--json", help="Return machine-readable JSON"),
):
    """List all snapshots (checkpoints) for a specific run."""
    from orchestra.engine.artifacts import list_checkpoints

    _require_run(run_id)
    snaps = list_checkpoints(run_id)
    if not snaps:
        if json_output:
            _print_json({"run_id": run_id, "checkpoints": []})
        else:
            console.print(f"[yellow]No checkpoints found for run {run_id}.[/yellow]")
        return
    if json_output:
        _print_json({"run_id": run_id, "checkpoints": snaps})
        return
        
    table = Table(title=f"Checkpoints for {run_id}", box=None)
    table.add_column("VER", style="dim")
    table.add_column("LABEL", style="cyan")
    table.add_column("STATUS", style="white")
    table.add_column("CONF", style="green")
    table.add_column("COST", style="yellow")
    
    for s in snaps:
        table.add_row(
            f"{s['checkpoint_version']:03d}",
            s['label'],
            s['status'],
            f"{s['avg_confidence']:.2f}",
            f"{s['total_cost_usd']:.4f}$"
        )
    console.print(table)


# ─── Backtracking Commands ──────────────────────────────────────────────────

@app.command("undo")
def undo_cmd(
    run_id: str = typer.Argument(..., help="Run ID to undo"),
):
    """Bring the working tree back to the snapshot a run took before it started.

    Only a run that recorded a snapshot can be undone, and no run mode records one at
    present; without a snapshot the command changes nothing.
    """
    from orchestra.engine.backtrack import find_snapshot, undo_last_run

    if find_snapshot(run_id) is None:
        console.print(f"[yellow]Run {run_id} has no snapshot to go back to. Nothing was changed.[/yellow]")
        raise typer.Exit(1)
    console.print("[yellow]This discards every uncommitted change and untracked file in the working tree, "
                  "then applies the run's snapshot.[/yellow]")
    if typer.confirm(f"Go back to the snapshot of run {run_id}?"):
        if undo_last_run(run_id):
            console.print(f"[green]Working tree is back at the snapshot of run {run_id}.[/green]")
        else:
            console.print("[red]The snapshot could not be applied.[/red]")


# ─── Evaluation Commands ────────────────────────────────────────────────────────


@app.command("eval-stats")
def eval_stats_cmd(
    task_hashes_file: str = typer.Argument(..., help="File with task hashes (one per line)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Compute overall evaluation statistics across multiple tasks.

    Aggregates metrics, pass rates, and identifies best/worst performing tasks.
    """
    from pathlib import Path
    from orchestra.engine.eval_judge import compute_eval_stats

    file_path = Path(task_hashes_file)
    if not file_path.exists():
        console.print(f"[red]Error: file not found: {file_path}[/red]")
        raise typer.Exit(1)

    task_hashes = [line.strip() for line in file_path.read_text().splitlines() if line.strip()]
    if not task_hashes:
        console.print("[red]Error: no task hashes found in file[/red]")
        raise typer.Exit(1)

    console.print(f"[bold cyan]Computing Stats:[/bold cyan] {len(task_hashes)} task(s)\n")

    stats = compute_eval_stats(task_hashes)

    if json_output:
        console.print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        console.print(f"[bold]Overall Statistics[/bold]")
        console.print(f"  Tasks evaluated: {stats['total_tasks_evaluated']}")
        console.print(f"  Total evals: {stats['total_evals']}")
        console.print(f"  Pass rate: {stats['overall_pass_rate']:.1%}")
        console.print(f"  Avg score: {stats['overall_avg_score']:.3f}")
        if stats['best_task']:
            console.print(f"  Best task: {stats['best_task']}")
        if stats['worst_task']:
            console.print(f"  Worst task: {stats['worst_task']}")


# ─── Report Commands ─────────────────────────────────────────────────────────

report_app = typer.Typer(name="report", help="Analytics and reports.")
app.add_typer(report_app)

@report_app.command("stats")
def report_stats_cmd(period: str = "daily", format: str = "table"):
    from orchestra.monitoring.analytics import UsageAnalytics
    analytics = UsageAnalytics()
    data = analytics.get_breakdown(period)
    if not data:
        console.print("[dim]No data found.[/dim]")
        return
    if format == "heatmap":
        print(analytics.generate_heatmap(data))
    elif format == "csv":
        print(analytics.export_csv(data))
    else:
        table = Table(title=f"Usage ({period})")
        table.add_column("Period")
        table.add_column("Runs", justify="right")
        table.add_column("Cost", justify="right")
        for r in data:
            table.add_row(r["period_key"], str(r["total_runs"]), f"${r['total_cost']:.4f}")
        console.print(table)

@report_app.command("cost")
def report_cost_cmd():
    from orchestra.storage.db import get_db
    conn = get_db()
    rows = conn.execute("SELECT provider, model, SUM(estimated_cost_usd) as cost FROM agents GROUP BY provider, model ORDER BY cost DESC").fetchall()
    conn.close()
    if not rows:
        console.print("[yellow]No data.[/yellow]")
        return
    table = Table(title="Cost by Model")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Cost", justify="right")
    for r in rows:
        table.add_row(r["provider"], r["model"], f"${r['cost']:.4f}")
    console.print(table)
    console.print("[dim]claude: what the CLI reports per call, at API prices (a subscription is not billed this way). "
                  "ollama: its real token counts times the [pricing] rate. Others: estimated tokens times that rate.[/dim]")

@app.command()
def doctor():
    """System health check for Orchestra components."""
    from rich.panel import Panel
    from rich.table import Table
    
    is_valid, errors = config.validate_config()
    config_status = "[green]✓ Valid[/green]" if is_valid else f"[red]✗ Invalid ({len(errors)} errors)[/red]"
    
    from orchestra.storage.db import get_db
    try:
        conn = get_db()
        run_count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        db_status = f"[green]✓ Connected ({run_count} runs)[/green]"
        conn.close()
    except Exception as e:
        db_status = f"[red]✗ Error: {e}[/red]"
        
    from orchestra.providers.fallback import ALIAS_TABLE
    providers: dict[str, bool] = {}
    for _alias, (provider, _effort) in ALIAS_TABLE.items():
        if provider.name not in providers:
            providers[provider.name] = bool(provider.is_available())
    ready = [name for name, ok in providers.items() if ok]
    down = [name for name, ok in providers.items() if not ok]
    provider_status = (f"[green]✓ {', '.join(ready)}[/green]" if ready else "[red]✗ none can run[/red]")
    if down:
        provider_status += f"  [dim]not available: {', '.join(down)}[/dim]"

    # A role pointing at an agent that cannot run is answered by a fallback model.
    roles = {**config.personas_config(), "supervisor worker": config.supervisor_worker_alias(),
             "supervisor planner": config.supervisor_planner_alias(), "single-agent default": config.default_light_alias()}
    broken = [f"{role} → {alias}" for role, alias in roles.items()
              if isinstance(alias, str) and alias in ALIAS_TABLE and not ALIAS_TABLE[alias][0].is_available()]
    role_status = "[green]✓ every role has an agent that can run[/green]" if not broken else f"[red]✗ {'; '.join(broken)}[/red]"
    
    from orchestra.engine.worktree_manager import WorktreeManager
    try:
        wtm = WorktreeManager()
        wts = wtm.list_worktrees()
        wt_status = f"[green]✓ Functional ({len(wts)} active)[/green]"
    except Exception as e:
        wt_status = f"[red]✗ Error: {e}[/red]"

    table = Table(box=None, show_header=False)
    table.add_row("Configuration", config_status)
    table.add_row("Database", db_status)
    table.add_row("AI Providers", provider_status)
    table.add_row("Roles", role_status)
    table.add_row("Git Worktrees", wt_status)
    
    console.print(Panel(table, title="[bold white]Orchestra Health Check[/bold white]", border_style="blue"))
    if errors:
        for err in errors: console.print(f"- {err}")


if __name__ == "__main__":
    app()

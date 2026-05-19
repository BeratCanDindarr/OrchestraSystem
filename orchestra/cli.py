"""Orchestra CLI — typer-based entry point."""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from orchestra.engine.pii_scrubber import PiiScrubber

app = typer.Typer(name="orchestra", help="Multi-agent orchestration CLI.", add_completion=True)
console = Console()


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _print_json(payload: object) -> None:
    console.print_json(json.dumps(payload, ensure_ascii=False, indent=2))


def _format_elapsed(start_time: str, end_time: Optional[str]) -> str:
    if not start_time:
        return "--:--"
    start = datetime.fromisoformat(start_time)
    end = datetime.fromisoformat(end_time) if end_time else datetime.now(timezone.utc)
    total_seconds = int((end - start).total_seconds())
    if total_seconds < 0:
        return "--:--"
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"


_ALIAS_COST_PER_1K: dict[str, float] = {
    "cdx-deep": 0.012, "cdx-fast": 0.004,
    "gmn-pro":  0.006, "gmn-fast": 0.002,
    "cld-deep": 0.015, "cld-fast": 0.003,
    "oll-coder": 0.0,  "oll-fast": 0.0, "oll-analyst": 0.0,
}


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


def _percent(part: int, whole: int) -> int:
    if whole <= 0:
        return 0
    return round((part / whole) * 100)


def _ensure_history_index() -> int:
    from orchestra.storage.db import backfill

    return backfill()


def _parse_scalar(value: str):
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_cli_value(value: str):
    stripped = value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    return _parse_scalar(stripped)


def _toml_literal(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_literal(item) for item in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f"\"{escaped}\""


def _flatten_toml_lines(data: dict, prefix: list[str] | None = None) -> list[str]:
    prefix = prefix or []
    scalar_items: list[tuple[str, object]] = []
    nested_items: list[tuple[str, dict]] = []
    for key, value in data.items():
        if isinstance(value, dict):
            nested_items.append((key, value))
        else:
            scalar_items.append((key, value))

    lines: list[str] = []
    if prefix:
        lines.append(f"[{'.'.join(prefix)}]")
    for key, value in scalar_items:
        lines.append(f"{key} = {_toml_literal(value)}")
    if prefix and nested_items:
        lines.append("")
    for index, (key, value) in enumerate(nested_items):
        lines.extend(_flatten_toml_lines(value, prefix + [key]))
        if index < len(nested_items) - 1:
            lines.append("")
    return lines


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

    d = run_dir(run_id)
    agents_dir = d / "agents"

    if not agents_dir.exists():
        console.print(f"[red]Run {run_id} not found.[/red]")
        raise typer.Exit(1)

    found_any = False
    for log_file in sorted(agents_dir.glob("*.stdout.log")):
        alias = log_file.stem.replace(".stdout", "")
        if agent and alias != agent:
            continue
        found_any = True
        console.rule(f"[dim]{alias}[/dim]")
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
                        log_lines.append(f"[dim]{alias}[/dim] {PiiScrubber.scrub_text(line)}")
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
):
    """Send a single task to one agent alias."""
    if dry_run:
        plan = _dry_run_plan("ask", alias, task)
        console.print(f"[bold]\\[dry-run\\][/bold] mode=[cyan]{plan['mode']}[/cyan] alias=[cyan]{plan['alias']}[/cyan] "
                      f"~{plan['estimated_tokens']} tokens  ~${plan['estimated_cost_usd']:.5f}")
        return

    from orchestra.engine.runner import run_ask

    run = run_ask(alias, task)
    if run.agents and run.agents[0].stdout_log and raw:
        console.print(PiiScrubber.scrub_text(run.agents[0].stdout_log))


@app.command()
def run(
    mode: str = typer.Argument(..., help="Execution mode: auto | ask | dual | critical | planned"),
    task: str = typer.Argument(..., help="Task prompt"),
    require_approval: bool = typer.Option(
        False,
        "--require-approval",
        help="Pause critical runs after round 1 and wait for approval before round 2",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show routing + cost estimate without executing"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Skip FTS5 cache lookups and storage"),
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

    from orchestra.engine.runner import run_ask, run_critical, run_dual, run_planned
    from orchestra.router import classify, task_to_mode

    if mode == "planned":
        run_planned(task)
        return

    if mode == "dual":
        if require_approval:
            console.print("[dim]--require-approval is only used by critical mode.[/dim]")
        run_dual(task)
        return

    if mode == "critical":
        run_critical(task, require_approval=require_approval)
        return

    if mode == "auto":
        route_class = classify(task)
        routed_mode = task_to_mode(task)

        if routed_mode.startswith("ask "):
            alias = routed_mode.split(" ", 1)[1].strip()
            console.print(f"[bold][auto][/bold] {route_class} → {alias}")
            if require_approval:
                console.print("[dim]--require-approval is ignored unless auto resolves to critical.[/dim]")
            run_ask(alias, task)
            return

        console.print(f"[bold][auto][/bold] {route_class} → {routed_mode}")
        if routed_mode == "critical":
            run_critical(task, require_approval=require_approval)
            return
        if routed_mode == "dual":
            if require_approval:
                console.print("[dim]--require-approval is only used when auto resolves to critical.[/dim]")
            run_dual(task)
            return
    
    console.print(f"[red]Unknown mode: {mode}[/red]")
    raise typer.Exit(1)


@app.command("resume")
def resume_cmd(
    run_id: str = typer.Argument(..., help="Run ID to resume"),
):
    """Resume a failed or interrupted run."""
    from orchestra.engine.runner import resume_run

    resume_run(run_id)


@app.command("job-worker", hidden=True)
def job_worker_cmd(job_id: str = typer.Argument(..., help="Job ID to execute")):
    """Internal command to execute a specific queued job."""
    from orchestra.service import execute_job

    execute_job(job_id)


@app.command("speculate")
def speculate_cmd(
    plan_id: str = typer.Argument(..., help="Speculative plan ID to execute"),
    mode: str = typer.Option("ask", "--mode", help="Execution mode: ask | fast"),
):
    """Execute a speculative plan in worktrees."""
    from orchestra.service import execute_speculation

    if mode not in {"ask", "fast"}:
        console.print(f"[red]Invalid mode: {mode}. Must be 'ask' or 'fast'.[/red]")
        raise typer.Exit(1)

    try:
        res = execute_speculation(plan_id, mode=mode)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    plan = res["plan"]
    evaluation = plan.get("evaluation", {})

    console.print(f"[bold]Speculative plan {plan_id} completed.[/bold]")

    table = Table(title="Hypothesis Results", box=None)
    table.add_column("Hypothesis", style="bold")
    table.add_column("Branch", style="dim")
    table.add_column("Status", style="white")
    table.add_column("Confidence", style="cyan")

    for hypothesis in plan.get("hypotheses", []):
        status = hypothesis.get("status", "unknown")
        status_color = "green" if status == "completed" else "red"
        table.add_row(
            hypothesis.get("name", "?"),
            hypothesis.get("branch", "?"),
            f"[{status_color}]{status}[/{status_color}]",
            f"{hypothesis.get('confidence', 0.0):.2f}",
        )
    console.print(table)

    winner = evaluation.get("selected_branch", "")
    if winner:
        console.print(f"\n[green]Winner:[/green] {winner}")
        console.print(f"[dim]{evaluation.get('notes', '')}[/dim]")
    else:
        console.print("\n[red]No winner selected.[/red]")


@app.command("speculate-prepare")
def speculate_prepare_cmd(
    plan_id: str = typer.Argument(..., help="Speculative plan ID to prepare"),
):
    """Prepare worktrees for a speculative plan."""
    from orchestra.service import prepare_speculation

    res = prepare_speculation(plan_id)
    console.print(f"[green]Prepared speculative worktrees for {plan_id}.[/green]")


@app.command()
def cancel(
    run_id: str = typer.Argument(..., help="Run ID to cancel"),
):
    """Cancel a running run."""
    from orchestra.engine.artifacts import load_manifest, write_manifest_data
    from orchestra.models import RunStatus

    manifest = load_manifest(run_id)
    if not manifest:
        console.print(f"[red]Run {run_id} not found.[/red]")
        return

    manifest["status"] = RunStatus.CANCELLED.value
    write_manifest_data(run_id, manifest)
    console.print(f"[yellow]Run {run_id} cancellation requested.[/yellow]")


@app.command()
def ps(
    status: Optional[str] = typer.Option(None, "--filter", help="Status filter, e.g. failed"),
    since: Optional[str] = typer.Option(None, "--since", help="Relative time, e.g. 24h or 7d"),
    json_output: bool = typer.Option(False, "--json", help="Return machine-readable JSON"),
):
    """List recent runs."""
    from orchestra.engine.artifacts import list_runs

    runs = list_runs()
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


@app.command("tui")
def tui(
    run_id: str = typer.Argument(..., help="Run ID to watch in the TUI"),
    interval: float = typer.Option(1.0, "--interval", "-i", help="Refresh interval in seconds"),
):
    """Launch the interactive Textual TUI for a run."""
    try:
        from orchestra.tui.app import run_tui
    except ImportError as exc:
        console.print("[red]Textual is not installed. Install it with:[/red] pip install textual")
        raise typer.Exit(1) from exc
    run_tui(run_id=run_id, interval=interval)


@app.command()
def logs(
    run_id: str = typer.Argument(..., help="Run ID"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Filter by agent alias"),
    json_output: bool = typer.Option(False, "--json", help="Return machine-readable JSON"),
):
    """Print agent logs for a specific run."""
    if json_output:
        from orchestra.service import get_logs

        _print_json(PiiScrubber.scrub_dict(get_logs(run_id, agent=agent, normalized=False)))
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
):
    """Search semantic memory for relevant snippets."""
    from orchestra.storage.memory import get_memory
    
    mem = get_memory()
    results = mem.search(query, limit=limit)
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
    
    events = read_events(run_id)
    if not events:
        console.print(f"[red]No events found for run {run_id}.[/red]")
        return
        
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
    """Evaluate a task using ask/dual/critical modes and judge quality.

    Runs three modes (ask/dual/critical) in parallel, judges each output on
    quality/grounding/conciseness, and returns ranked results by composite score.
    """
    import concurrent.futures
    from contextvars import copy_context
    from orchestra.engine.runner import run_ask, run_dual, run_critical
    from orchestra.engine.eval_harness import (
        task_hash,
        run_judge,
        save_eval,
    )

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
        console.print("  1. ask      — single fast agent (cdx-fast)")
        console.print("  2. dual     — two-agent synthesis (ask + judge)")
        console.print("  3. critical — multi-round critical evaluation")
        console.print(f"\nEstimated cost: $0.03–$0.15 total")
        console.print(f"  - Execution: ~$0.01–$0.10")
        console.print(f"  - Judge (3x): ~$0.01")
        console.print(f"\nResults will be saved to: ~/.orchestra/evals/{task_id}.jsonl")
        return

    console.print(f"\n[bold cyan]Evaluating:[/bold cyan] {task[:100]}...\n")

    def _run_mode(mode: str) -> dict:
        """Run a single mode and return result with output."""
        try:
            if mode == "ask":
                run = run_ask("cdx-fast", task, emit_console=False, show_live=False)
            elif mode == "dual":
                run = run_dual(task, emit_console=False, show_live=False)
            else:  # critical
                run = run_critical(task, emit_console=False, show_live=False)

            # Extract output from run
            output = ""
            if run.summary:
                output = run.summary
            elif run.agents and run.agents[0].stdout_log:
                output = run.agents[0].stdout_log

            return {
                "mode": mode,
                "run_id": run.run_id,
                "status": run.status.value,
                "confidence": round(float(run.avg_confidence or 0), 3),
                "cost_usd": round(float(run.total_cost_usd or 0), 6),
                "output": output,
                "error": None,
            }
        except Exception as e:
            return {
                "mode": mode,
                "run_id": "",
                "status": "failed",
                "confidence": 0.0,
                "cost_usd": 0.0,
                "output": "",
                "error": str(e),
            }

    # Run three modes in parallel
    console.print("[dim]Running ask / dual / critical in parallel...[/dim]")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(copy_context().run, _run_mode, m): m for m in ("ask", "dual", "critical")}
        mode_results = {}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            mode = res["mode"]
            mode_results[mode] = res
            if res["error"]:
                console.print(f"[yellow]  {mode}: {res['error']}[/yellow]")
            else:
                console.print(f"[green]  {mode}: completed[/green]")

    # Judge each output
    console.print("\n[dim]Judging outputs...[/dim]")
    for mode, res in mode_results.items():
        if res["error"] or not res["output"]:
            res["judge"] = {"score": 0.0, "reasoning": "No output to judge"}
            res["cost_per_quality"] = 0.0
            continue

        try:
            eval_result = run_judge(
                task_text=task,
                agent_output=res["output"],
                run_id=res["run_id"],
                task_hash=task_id,
                mode=mode,
            )
            judge_criteria = eval_result.judge_criteria
            composite_score = eval_result.composite_score
            judge_cost = eval_result.judge_cost_usd

            res["judge"] = {
                "score": composite_score,
                "quality": judge_criteria.answer_quality,
                "grounding": judge_criteria.factual_grounding,
                "conciseness": judge_criteria.conciseness,
                "reasoning": judge_criteria.answer_quality_reasoning,
            }
            res["judge_cost_usd"] = judge_cost
            res["cost_per_quality"] = round(
                (res["cost_usd"] + judge_cost) / max(composite_score, 0.1), 6
            )

            save_eval(eval_result)
            console.print(f"[green]  {mode}: scored {composite_score:.2f}[/green]")
        except ValueError as parse_err:
            console.print(f"[red]  {mode}: judge parse error: {parse_err}[/red]")
            res["judge"] = {"score": 0.0, "reasoning": f"Parse error: {parse_err}"}
            res["cost_per_quality"] = 0.0
        except Exception as judge_err:
            console.print(f"[red]  {mode}: judge error: {judge_err}[/red]")
            res["judge"] = {"score": 0.0, "reasoning": f"Judge error: {judge_err}"}
            res["cost_per_quality"] = 0.0

    # Rank by composite score
    ordered = sorted(
        mode_results.values(),
        key=lambda r: -float((r.get("judge") or {}).get("score") or 0),
    )

    if json_output:
        _print_json(PiiScrubber.scrub_dict({"task": task, "task_hash": task_id, "results": ordered}))
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

def _parse_ts(ts_str: str | None) -> datetime | None:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 0:
        return "—"
    if seconds < 60:
        return f"+{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"+{m:02d}:{s:02d}"


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
    status = manifest.status or "?"
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
        compare_field_values,
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
            console.print(PiiScrubber.scrub_text(summary_a[:500]) or "[dim](none)[/dim]")
        if summary_b:
            console.rule("[dim]Run B Synthesis[/dim]")
            console.print(PiiScrubber.scrub_text(summary_b[:500]) or "[dim](none)[/dim]")


@app.command("replay-synth")
def replay_synth_cmd(
    run_id: str = typer.Argument(..., help="Run ID or 8-char prefix"),
    synthesizer: str = typer.Option("claude", "--synthesizer", "-s", help="Synthesizer alias to use (default: claude)"),
    json_output: bool = typer.Option(False, "--json", help="Return machine-readable JSON"),
) -> None:
    """Re-synthesize a run from cached agent outputs (no re-execution).

    Loads agent outputs from an existing run and re-synthesizes them using the specified
    synthesizer. Useful for testing different synthesis models or strategies without
    re-running agents. Compares cost and confidence between original and replayed synthesis.
    """
    from orchestra.engine.observability import resolve_run_id
    from orchestra.engine.artifacts import load_manifest, read_agent_log
    from orchestra.engine.synthesizer import build_synthesis_prompt, outputs_sufficient, select_synthesis_alias
    from orchestra.engine.runner import run_ask
    from orchestra.models import AgentRun, AgentStatus
    from orchestra.providers.fallback import available_aliases

    # Resolve run_id prefix
    resolved_id = resolve_run_id(run_id)
    if not resolved_id:
        console.print(f"[red]Error: run_id '{run_id}' not found[/red]")
        raise typer.Exit(1)

    # Load manifest
    manifest = load_manifest(resolved_id)
    if not manifest:
        console.print(f"[red]Error: run_id '{resolved_id}' not found[/red]")
        raise typer.Exit(1)

    # Validate dual+ mode (2+ agents required for synthesis)
    agents_meta = manifest.get("agents") or []
    completed_meta = [a for a in agents_meta if a.get("status") == "completed" and not a.get("soft_failed")]
    if len(completed_meta) < 2:
        console.print(
            f"[red]Error: run requires 2+ agents for synthesis (found: {len(completed_meta)})[/red]"
        )
        raise typer.Exit(1)

    # Validate synthesizer alias
    valid_aliases = set(available_aliases())
    if synthesizer not in valid_aliases:
        console.print(f"[red]Error: unknown synthesizer '{synthesizer}'[/red]")
        raise typer.Exit(1)

    # Load cached agent outputs
    reconstructed: list[AgentRun] = []
    for meta in completed_meta[:2]:
        alias = meta.get("alias", "")
        log = read_agent_log(resolved_id, alias) or ""
        agent = AgentRun(alias=alias, provider=meta.get("provider", ""), model=meta.get("model", ""))
        agent.status = AgentStatus.COMPLETED
        agent.confidence = float(meta.get("confidence") or 0.5)
        agent.stdout_log = log
        agent.soft_failed = bool(meta.get("soft_failed", False))
        agent.validation_status = meta.get("validation_status", "not_run")
        reconstructed.append(agent)

    agent_a, agent_b = reconstructed[0], reconstructed[1]
    if not outputs_sufficient(agent_a, agent_b):
        console.print("[red]Error: no cached agent outputs found for run[/red]")
        raise typer.Exit(1)

    # Build synthesis prompt and run replay synthesis
    synth_prompt = build_synthesis_prompt(agent_a, agent_b)
    console.print(f"[bold cyan]Replaying: {resolved_id} using synthesizer={synthesizer}[/bold cyan]\n")

    synth_run = run_ask(synthesizer, synth_prompt, emit_console=False, show_live=False)
    if not synth_run.agents or not synth_run.agents[0].stdout_log:
        console.print("[red]Error: synthesis produced no output[/red]")
        raise typer.Exit(1)

    replayed_output = synth_run.agents[0].stdout_log
    original_output = (manifest.get("summary") or "").strip()
    original_cost = float(manifest.get("total_cost_usd") or 0.0)
    replayed_cost = float(synth_run.total_cost_usd or 0.0)
    original_confidence = float(manifest.get("avg_confidence") or 0.0)
    replayed_confidence = float(synth_run.avg_confidence or 0.0)
    agents_str = ", ".join(a.get("alias", "?") for a in completed_meta[:2])

    if json_output:
        _print_json({
            "run_id": resolved_id,
            "synthesizer": synthesizer,
            "replay_run_id": synth_run.run_id,
            "original_output": original_output[:500],
            "replayed_output": replayed_output[:500],
            "original_cost_usd": round(original_cost, 6),
            "replayed_cost_usd": round(replayed_cost, 6),
            "cost_savings_usd": round(original_cost - replayed_cost, 6),
            "original_confidence": round(original_confidence, 3),
            "replayed_confidence": round(replayed_confidence, 3),
            "confidence_delta": round(replayed_confidence - original_confidence, 3),
            "agents": completed_meta[:2],
        })
        return

    # Terminal: side-by-side original vs replayed
    console.print("[bold cyan]Original Synthesis:[/bold cyan]")
    console.rule("────────────────────────────────────────")
    console.print(PiiScrubber.scrub_text(original_output[:500]) + ("..." if len(original_output) > 500 else ""))
    console.rule("────────────────────────────────────────")

    console.print()
    console.print(f"[bold cyan]Replayed Synthesis ({synthesizer}):[/bold cyan]")
    console.rule("────────────────────────────────────────")
    console.print(PiiScrubber.scrub_text(replayed_output[:500]) + ("..." if len(replayed_output) > 500 else ""))
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
    console.print(f"[bold]Agents:[/bold] {len(completed_meta[:2])} ({agents_str})")


@app.command("replay-run")
def replay_run_cmd(
    run_id: str = typer.Argument(..., help="Run ID or 8-char prefix"),
    deterministic: bool = typer.Option(False, "--deterministic", "-d", help="Pin temperature=0 and seed for supported providers"),
    json_output: bool = typer.Option(False, "--json", help="Return machine-readable JSON"),
) -> None:
    """Replay a run from its recorded replay log.

    Reconstructs the execution path by re-running agent dispatches and
    serving cached tool outputs from the replay log.
    """
    from orchestra.engine.observability import resolve_run_id
    from orchestra.engine.replay import ReplayRunner

    resolved_id = resolve_run_id(run_id)
    if not resolved_id:
        console.print(f"[red]Error: run_id '{run_id}' not found[/red]")
        raise typer.Exit(1)

    runner = ReplayRunner(resolved_id)
    try:
        runner.replay(deterministic=deterministic)
    except Exception as exc:
        console.print(f"[red]Replay failed: {exc}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Replay completed for {resolved_id}[/bold green]")
    if deterministic:
        console.print("[dim]Deterministic mode: temperature=0 + fixed seed (where supported)[/dim]")


@app.command("checkpoints")
def list_checkpoints_cmd(
    run_id: str = typer.Argument(...),
    json_output: bool = typer.Option(False, "--json", help="Return machine-readable JSON"),
):
    """List all snapshots (checkpoints) for a specific run."""
    from orchestra.engine.artifacts import list_checkpoints
    
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


@app.command("branch")
def branch_cmd(
    run_id: str = typer.Argument(..., help="Source run ID or 8-char prefix"),
    from_version: int = typer.Option(0, "--from-version", help="Checkpoint version to branch from (0 = latest)"),
    from_label: str = typer.Option("", "--from-label", help="Checkpoint label to branch from"),
) -> None:
    """Create a new run from a checkpoint, preserving agent state."""
    import uuid
    from orchestra.engine.artifacts import load_checkpoint, write_manifest_data
    from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus

    checkpoint = load_checkpoint(run_id, checkpoint_version=from_version or None, label=from_label or None)
    if not checkpoint:
        console.print(f"[red]Checkpoint not found for run '{run_id}'[/red]")
        raise typer.Exit(1)

    new_run_id = f"branch-{uuid.uuid4().hex[:8]}"
    new_run = OrchestraRun(
        run_id=new_run_id,
        mode=checkpoint.get("mode", "ask"),
        task=f"(branched from {run_id} v{checkpoint.get('checkpoint_version')})",
        status=RunStatus.PENDING,
    )
    for agent_dict in checkpoint.get("agents", []):
        new_run.agents.append(AgentRun.from_dict(agent_dict))

    manifest = new_run.to_manifest()
    manifest["branched_from"] = {
        "run_id": run_id,
        "checkpoint_version": checkpoint.get("checkpoint_version"),
        "label": checkpoint.get("label"),
    }
    write_manifest_data(new_run_id, manifest)
    console.print(f"[green]Branched into new run: {new_run_id}[/green]")
    console.print(f"  Agents restored: {len(new_run.agents)}")
    console.print(f"  Resume with: orchestra run --resume {new_run_id}")


# ─── Backtracking Commands ──────────────────────────────────────────────────

@app.command("undo")
def undo_cmd(
    run_id: str = typer.Argument(..., help="Run ID to undo"),
):
    """Revert all file system changes made by a specific run."""
    from orchestra.engine.backtrack import undo_last_run
    
    if typer.confirm(f"Are you sure you want to revert changes for run {run_id}?"):
        if undo_last_run(run_id):
            console.print(f"[green]Successfully reverted changes for run {run_id}.[/green]")
        else:
            console.print(f"[red]Failed to undo. Snapshot might be missing or expired.[/red]")


# ─── Batch Commands ─────────────────────────────────────────────────────────

@app.command(name="batch-run")
def batch_run(
    file: Path = typer.Argument(..., help="JSONL file; each line: {mode, task, alias?}"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Max parallel runs (default 1)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate file and show plan without executing"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSONL to stdout"),
):
    """Submit multiple tasks from a JSONL file. Each line: {mode, task, alias?}."""
    from rich.table import Table as RichTable

    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(2)

    tasks: list[dict] = []
    errors: list[str] = []
    for lineno, raw_line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            entry = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: invalid JSON — {exc}")
            continue
        if "task" not in entry:
            errors.append(f"line {lineno}: missing required key 'task'")
            continue
        if "mode" not in entry:
            errors.append(f"line {lineno}: missing required key 'mode'")
            continue
        tasks.append(entry)

    if errors:
        for err in errors:
            console.print(f"[red][batch-run] {err}[/red]")
        raise typer.Exit(2)

    if not tasks:
        console.print("[red][batch-run] File is empty or contains no valid task lines.[/red]")
        raise typer.Exit(2)

    if dry_run:
        console.print(f"[bold][dry-run][/bold] {len(tasks)} task(s) in [cyan]{file.name}[/cyan]")
        t = RichTable("#", "mode", "alias", "task")
        for i, entry in enumerate(tasks, 1):
            t.add_row(str(i), entry["mode"], entry.get("alias", "—"), entry["task"][:60])
        console.print(t)
        return

    from orchestra.service import run_task

    results: list[dict] = []
    for i, entry in enumerate(tasks, 1):
        mode_val = entry["mode"]
        task_text = entry["task"]
        alias_val = entry.get("alias")
        console.print(f"[bold][{i}/{len(tasks)}][/bold] mode={mode_val} alias={alias_val or 'auto'} ...")
        try:
            result = run_task(mode=mode_val, task=task_text, alias=alias_val)
            results.append({"index": i, "status": "ok", **result})
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]  failed: {exc}[/red]")
            results.append({"index": i, "status": "error", "error": str(exc)})

    if json_output:
        for r in results:
            console.print(json.dumps(PiiScrubber.scrub_dict(r), ensure_ascii=False))
    else:
        ok = sum(1 for r in results if r["status"] == "ok")
        console.print(f"\n[bold]Done:[/bold] {ok}/{len(tasks)} succeeded.")


# ─── Evaluation Commands ────────────────────────────────────────────────────────

@app.command("eval-batch")
def eval_batch_cmd(
    task_hashes_file: str = typer.Argument(..., help="File with task hashes (one per line)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Batch evaluate multiple tasks and aggregate statistics.

    Reads task hashes from a file and computes batch statistics including pass rates,
    score distributions, and per-task aggregates.
    """
    from pathlib import Path
    from orchestra.engine.eval_harness import batch_evaluate

    file_path = Path(task_hashes_file)
    if not file_path.exists():
        console.print(f"[red]Error: file not found: {file_path}[/red]")
        raise typer.Exit(1)

    task_hashes = [line.strip() for line in file_path.read_text().splitlines() if line.strip()]
    if not task_hashes:
        console.print("[red]Error: no task hashes found in file[/red]")
        raise typer.Exit(1)

    console.print(f"[bold cyan]Batch Evaluating:[/bold cyan] {len(task_hashes)} task(s)\n")

    batch_results = batch_evaluate(task_hashes)

    if not batch_results:
        console.print("[yellow]No evaluation results found.[/yellow]")
        raise typer.Exit(0)

    if json_output:
        for result in batch_results:
            console.print(json.dumps(result.to_dict(), ensure_ascii=False))
    else:
        from rich.table import Table
        t = Table("Task Hash", "Total", "Passed", "Failed", "Avg Score", "Min", "Max")
        for result in batch_results:
            t.add_row(
                result.task_hash[:8],
                str(result.total_evals),
                f"[green]{result.passed_evals}[/green]",
                f"[red]{result.failed_evals}[/red]",
                f"{result.avg_composite_score:.3f}",
                f"{result.min_composite_score:.3f}",
                f"{result.max_composite_score:.3f}",
            )
        console.print(t)


@app.command("eval-stats")
def eval_stats_cmd(
    task_hashes_file: str = typer.Argument(..., help="File with task hashes (one per line)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Compute overall evaluation statistics across multiple tasks.

    Aggregates metrics, pass rates, and identifies best/worst performing tasks.
    """
    from pathlib import Path
    from orchestra.engine.eval_harness import compute_eval_stats

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


# ─── Skill Commands ─────────────────────────────────────────────────────────

@app.command("teach")
def teach_cmd(
    instruction: str = typer.Argument(..., help="Natural language description of the skill"),
    name: str = typer.Option("", "--name", help="Skill name (auto-generated if omitted)"),
    alias: str = typer.Option("", "--alias", help="Agent alias to use for generation"),
) -> None:
    """Teach Orchestra a new user skill from a natural language instruction."""
    from orchestra import config as _config
    from orchestra.engine.runner import run_ask
    from orchestra.engine.skills.dispatch import clear_registry
    from orchestra.engine.skills.loader import _user_skills_dir
    from orchestra.engine.skills.manifest import SkillManifest, save_manifest

    teach_alias = alias or _config.skills_config().get("teach_alias", "cld-fast")
    skill_name = name or instruction.lower().replace(" ", "_")[:32]
    user_dir = _user_skills_dir()
    target_dir = user_dir / skill_name
    if target_dir.exists():
        console.print(f"[red]Skill '{skill_name}' already exists.[/red]")
        raise typer.Exit(1)

    prompt = (
        f"Generate a concise Python module for an Orchestra skill named '{skill_name}'.\n"
        f"The skill should enforce this rule: {instruction}\n\n"
        "Return ONLY the Python code (no markdown fences). "
        "The module must have a docstring, a can_execute(context) function returning bool, "
        "and an execute(context) function returning a SkillResult-like dict with keys success, output, metrics."
    )
    console.print(f"[bold cyan]Teaching skill '{skill_name}' via {teach_alias}...[/bold cyan]")
    output, rc = run_ask(teach_alias, prompt, emit_console=False, show_live=False)
    if rc != 0:
        console.print(f"[red]Skill generation failed (exit {rc}).[/red]")
        raise typer.Exit(1)

    target_dir.mkdir(parents=True, exist_ok=True)
    code = output.strip()
    if code.startswith("```python"):
        code = code[len("```python"):]
    if code.startswith("```"):
        code = code[len("```"):]
    if code.endswith("```"):
        code = code[:-len("```")]
    code = code.strip()
    (target_dir / "skill.py").write_text(code, encoding="utf-8")

    manifest = SkillManifest(
        name=skill_name,
        version="1.0.0",
        tier="user",
        description=instruction,
        entrypoint="skill.py",
        policy="auto",
    )
    save_manifest(target_dir / "manifest.yaml", manifest)
    clear_registry()
    console.print(f"[green]Skill '{skill_name}' saved to {target_dir}[/green]")


skill_app = typer.Typer(name="skill", help="Tiered skill system management.")
app.add_typer(skill_app)


@skill_app.command("list")
def skill_list_cmd(
    tier: str = typer.Option("", "--tier", help="Filter by tier: core|user|marketplace"),
    all_tiers: bool = typer.Option(False, "--all", help="Include marketplace skills not yet installed"),
) -> None:
    """List available skills."""
    from orchestra.engine.skills.dispatch import list_skills
    from orchestra.engine.skills.marketplace.index import load_index
    from rich.table import Table

    manifests = list_skills(tier=tier or None)
    if all_tiers:
        for name, manifest in load_index().items():
            if name not in {m.name for m in manifests}:
                if tier and manifest.tier != tier:
                    continue
                manifests.append(manifest)

    t = Table("Name", "Tier", "Version", "Policy", "Reputation", "Description")
    for m in sorted(manifests, key=lambda x: (x.tier, x.name)):
        t.add_row(
            m.name,
            m.tier,
            m.version,
            m.policy,
            str(round(m.reputation_score, 1)),
            m.description or "",
        )
    console.print(t)


@skill_app.command("run")
def skill_run_cmd(
    name: str = typer.Argument(..., help="Skill name"),
    path: str = typer.Option(".", "--path", help="Project path to operate on"),
    json_output: bool = typer.Option(False, "--json", help="Return machine-readable JSON"),
) -> None:
    """Run a skill by name."""
    from orchestra.engine.skills.base import SkillContext
    from orchestra.engine.skills.dispatch import run_skill

    result = run_skill(name, SkillContext(task="", project_path=path))
    if json_output:
        console.print_json(json.dumps({"success": result.success, "output": result.output, "metrics": result.metrics}, ensure_ascii=False))
    else:
        color = "green" if result.success else "red"
        console.print(f"[{color}]{'Success' if result.success else 'Failed'}[/[{color}]]: {result.output}")


@skill_app.command("install")
def skill_install_cmd(
    name: str = typer.Argument(..., help="Marketplace skill name"),
) -> None:
    """Install a marketplace skill into the user skills directory."""
    from orchestra.engine.skills.dispatch import install_marketplace_skill

    result = install_marketplace_skill(name)
    if result.success:
        console.print(f"[green]{result.output}[/green]")
    else:
        console.print(f"[red]{result.output}[/red]")
        raise typer.Exit(1)


@skill_app.command("uninstall")
def skill_uninstall_cmd(
    name: str = typer.Argument(..., help="Skill name to remove"),
) -> None:
    """Uninstall a user-tier skill."""
    from orchestra.engine.skills.dispatch import uninstall_skill

    result = uninstall_skill(name)
    if result.success:
        console.print(f"[green]{result.output}[/green]")
    else:
        console.print(f"[red]{result.output}[/red]")
        raise typer.Exit(1)


# ─── Graph Commands ─────────────────────────────────────────────────────────

@app.command("graph")
def graph_cmd(
    run_id: str = typer.Argument(..., help="Run ID or 8-char prefix"),
    fmt: str = typer.Option("ascii", "--format", help="Output format: ascii | dot | json | html | svg | png"),
    out: str = typer.Option("", "--out", help="Output file path (for svg/png/dot)"),
) -> None:
    """Render the consensus graph for a completed run."""
    from orchestra.engine.consensus import build_consensus_graph, format_consensus_ascii, format_consensus_dot
    from orchestra.engine.skills.dispatch import run_skill
    from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus
    from orchestra.storage.db import get_db

    conn = get_db()
    row = conn.execute(
        "SELECT run_id, mode, status, task FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if not row:
        console.print(f"[red]Run '{run_id}' not found.[/red]")
        raise typer.Exit(1)

    agent_rows = conn.execute(
        "SELECT alias, provider, model, status, confidence, soft_failed FROM agents WHERE run_id = ?",
        (run_id,),
    ).fetchall()

    run = OrchestraRun(
        run_id=row["run_id"],
        mode=row["mode"],
        task=row["task"],
        status=RunStatus(row["status"]),
    )
    for ar in agent_rows:
        run.agents.append(
            AgentRun(
                alias=ar["alias"],
                provider=ar["provider"],
                model=ar["model"],
                status=AgentStatus(ar["status"]),
                confidence=ar["confidence"],
                soft_failed=bool(ar["soft_failed"]),
            )
        )

    graph = build_consensus_graph(run)

    if fmt == "ascii":
        console.print(format_consensus_ascii(graph))
    elif fmt == "dot":
        dot = format_consensus_dot(graph)
        if out:
            from pathlib import Path
            Path(out).write_text(dot, encoding="utf-8")
            console.print(f"[green]Saved DOT to {out}[/green]")
        else:
            console.print(dot)
    elif fmt == "json":
        import json

        console.print_json(json.dumps(graph.to_dict(), ensure_ascii=False))
    elif fmt == "html":
        import shutil
        import subprocess

        if not shutil.which("dot"):
            console.print("[red]graphviz 'dot' not found in PATH[/red]")
            raise typer.Exit(1)
        dot = format_consensus_dot(graph)
        proc = subprocess.run(
            ["dot", "-Tsvg"],
            input=dot,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            console.print(f"[red]graphviz failed: {proc.stderr}[/red]")
            raise typer.Exit(1)
        svg = proc.stdout
        # Build simple HTML page with embedded SVG + metadata
        node_rows = "\n".join(
            f"<tr><td>{n.alias}</td><td>{n.model_family}</td><td>{n.confidence:.2f}</td><td>{n.status}</td></tr>"
            for n in graph.nodes.values()
        )
        edge_rows = "\n".join(
            f"<tr><td>{e.source}</td><td>{e.target}</td><td>{e.kind}</td><td>{e.similarity_score:.3f}</td></tr>"
            for e in graph.edges
        )
        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Consensus Graph — {graph.run_id}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #0e1117; color: #c9d1d9; }}
h1 {{ color: #58a6ff; }}
.svg-container {{ background: #161b22; border-radius: 8px; padding: 1rem; margin: 1rem 0; overflow: auto; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #30363d; padding: 0.5rem; text-align: left; }}
th {{ background: #21262d; }}
.aligned {{ color: #3fb950; }} .dissent {{ color: #f85149; }} .partial {{ color: #d29922; }}
</style>
</head>
<body>
<h1>Consensus Graph — {graph.run_id}</h1>
<p>Level: <strong class="{graph.consensus_level}">{graph.consensus_level}</strong></p>
<div class="svg-container">
{svg}
</div>
<h2>Nodes</h2>
<table>
<tr><th>Alias</th><th>Family</th><th>Confidence</th><th>Status</th></tr>
{node_rows}
</table>
<h2>Edges</h2>
<table>
<tr><th>Source</th><th>Target</th><th>Kind</th><th>Score</th></tr>
{edge_rows}
</table>
</body>
</html>"""
        out_path = out or f"{run_id}_consensus.html"
        from pathlib import Path
        Path(out_path).write_text(html, encoding="utf-8")
        console.print(f"[green]Saved HTML to {out_path}[/green]")
    elif fmt in ("svg", "png"):
        result = run_skill(
            "graphviz_export",
            args={"run_id": run_id, "format": fmt, "out": out or f"{run_id}_consensus.{fmt}"},
        )
        if result.success:
            console.print(f"[green]{result.output}[/green]")
        else:
            console.print(f"[red]{result.output}[/red]")
            raise typer.Exit(1)
    else:
        console.print(f"[red]Unknown format: {fmt}[/red]")
        raise typer.Exit(1)


# ─── Federation Commands ────────────────────────────────────────────────────

federation_app = typer.Typer(name="federation", help="Federated semantic memory across Orchestra instances.")
app.add_typer(federation_app)


@federation_app.command("serve")
def federation_serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(8766, "--port", help="Bind port"),
) -> None:
    """Start the federation HTTP server."""
    from orchestra.federation_server.federation_http import run_app
    console.print(f"[bold green]Federation server starting on {host}:{port}...[/bold green]")
    run_app(host=host, port=port)


@federation_app.command("sync")
def federation_sync_cmd(
    peer: str = typer.Option("", "--peer", help="Peer URL (if omitted, syncs all configured peers)"),
) -> None:
    """Push outbox summaries to peer(s) and clear on success."""
    import asyncio
    from orchestra.federation_server.federation_client import sync_with_peer, sync_all_peers
    from orchestra.engine.federation import FederationPeer

    async def _run():
        from orchestra.federation_server.federation_client import _peer_from_config
        if peer:
            p = _peer_from_config(peer, 0)
            result = await sync_with_peer(p)
            console.print_json(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            results = await sync_all_peers()
            console.print_json(json.dumps(results, ensure_ascii=False, indent=2))

    asyncio.run(_run())


@federation_app.command("query")
def federation_query_cmd(
    task: str = typer.Argument(..., help="Task text to search across peers"),
    peer: str = typer.Option("", "--peer", help="Query a specific peer URL"),
    limit: int = typer.Option(5, "--limit", help="Max results per peer"),
) -> None:
    """Query remote peers for federated summaries similar to a task."""
    import asyncio
    from orchestra.federation_server.federation_client import query_peer
    from orchestra.engine.federation import FederationPeer

    async def _run():
        from orchestra.federation_server.federation_client import _peer_from_config
        if peer:
            peers = [_peer_from_config(peer, 0)]
        else:
            urls = config.federation_config().get("peer_urls", [])
            peers = [_peer_from_config(u, i) for i, u in enumerate(urls)]
        for p in peers:
            results = await query_peer(p, task, limit)
            console.print(f"[bold cyan]{p.endpoint_url}[/bold cyan]")
            if not results:
                console.print("  (no results)")
            for r in results:
                if "error" in r:
                    console.print(f"  [red]error: {r['error']}[/red]")
                else:
                    console.print(f"  {r['content_hash']}  score={r.get('score', 0)}  peer={r.get('peer_id', '?')}")

    asyncio.run(_run())


@federation_app.command("peers")
def federation_peers_cmd() -> None:
    """List configured federation peers."""
    peers = config.federation_config().get("peer_urls", [])
    if not peers:
        console.print("[yellow]No peers configured.[/yellow]")
        return
    from rich.table import Table
    t = Table("Index", "URL")
    for i, url in enumerate(peers):
        t.add_row(str(i), url)
    console.print(t)


@federation_app.command("status")
def federation_status_cmd() -> None:
    """Show federation outbox/inbox statistics."""
    from orchestra.engine.federation import FederationBus
    bus = FederationBus()
    stats = bus.stats()
    console.print(f"[bold]Federation Status[/bold]")
    console.print(f"  Outbox: {stats['outbox_count']} pending")
    console.print(f"  Inbox:  {stats['inbox_count']} received")
    console.print(f"  Root:   {stats['root']}")


@federation_app.command("rotate-token")
def federation_rotate_token_cmd(
    ttl_hours: int = typer.Option(720, "--ttl", help="Token lifetime in hours (default: 30 days)"),
) -> None:
    """Generate a new federation auth token and print rotation instructions."""
    import secrets
    import time

    new_token = secrets.token_urlsafe(32)
    issued_at = time.time()
    expires_at = issued_at + (ttl_hours * 3600)

    console.print("[bold green]New federation auth token generated.[/bold green]")
    console.print(f"\nToken: [bold]{new_token}[/bold]")
    console.print(f"Issued at: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(issued_at))}")
    console.print(f"Expires at: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(expires_at))}")
    console.print(f"\nAdd the following to your .orchestra/config.toml [federation] section:")
    console.print(f"  auth_token = \"{new_token}\"")
    console.print(f"  auth_token_issued_at = {issued_at}")
    console.print(f"  auth_token_ttl_hours = {ttl_hours}")
    console.print("\n[yellow]Remember to distribute the new token to all peers.[/yellow]")


@federation_app.command("push")
def federation_push_cmd(
    run_id: str = typer.Argument(..., help="Run ID to push to federation"),
) -> None:
    """Summarize a completed run and enqueue it to the federation outbox."""
    from orchestra.engine.artifacts import load_manifest
    from orchestra.engine.federation import FederationBus
    from orchestra.engine.federation_summarizer import summarize_run
    from orchestra.models import OrchestraRun

    manifest = load_manifest(run_id)
    if not manifest:
        console.print(f"[red]Run '{run_id}' not found.[/red]")
        raise typer.Exit(1)

    run = OrchestraRun.from_manifest(manifest)
    summary = summarize_run(run)
    if not summary:
        console.print("[yellow]Run not suitable for federation (no completed agents or empty task).[/yellow]")
        raise typer.Exit(1)

    bus = FederationBus()
    bus.enqueue_outbox(summary)
    console.print(f"[green]Pushed summary {summary.content_hash} to federation outbox.[/green]")


@federation_app.command("search")
def federation_search_cmd(
    query: str = typer.Argument(..., help="Task text to search in local federation inbox"),
    limit: int = typer.Option(5, "--limit", help="Max results"),
) -> None:
    """Search the local federation inbox for summaries matching a task."""
    from orchestra.engine.federation import FederationBus
    from rich.table import Table

    bus = FederationBus()
    inbox = bus.read_inbox()

    # Simple keyword match for MVP
    results: list[tuple[float, Any]] = []
    for summary in inbox:
        score = 0.0
        task_hash = summary.metadata.get("task_hash", "")
        mode = summary.metadata.get("mode", "")
        tags = list(summary.crdt_state.tags.elements())

        query_lower = query.lower()
        if query_lower in task_hash.lower():
            score += 0.5
        if mode.lower() in query_lower:
            score += 0.3
        for tag in tags:
            if tag.lower() in query_lower:
                score += 0.2

        if score > 0:
            results.append((score, summary))

    results.sort(key=lambda x: x[0], reverse=True)
    results = results[:limit]

    if not results:
        console.print("[yellow]No matching summaries in local inbox.[/yellow]")
        return

    table = Table("Hash", "Score", "Mode", "Agents", "Peer")
    for score, summary in results:
        table.add_row(
            summary.content_hash,
            f"{score:.2f}",
            summary.metadata.get("mode", "?"),
            str(summary.metadata.get("agent_count", 0)),
            summary.peer_id,
        )
    console.print(table)


# ─── Report Commands ────────────────────────────────────────────────────────

report_app = typer.Typer(name="report", help="Quality and operational reports.")
app.add_typer(report_app)


@report_app.command("synthesis")
def report_synthesis_cmd(
    alias: str = typer.Option("", "--alias", help="Filter by synthesizer alias"),
    limit: int = typer.Option(20, "--limit", help="Max rows to show"),
) -> None:
    """Show synthesis quality report with averages per alias."""
    from orchestra.storage.db import get_db
    from rich.table import Table

    conn = get_db()
    where = "WHERE alias = ?" if alias else ""
    params = (alias,) if alias else ()

    # Per-alias averages
    rows = conn.execute(
        f"""
        SELECT alias, COUNT(*) as cnt, ROUND(AVG(score), 3) as avg_score,
               ROUND(AVG(divergence_index), 3) as avg_div,
               ROUND(AVG(length), 0) as avg_len
        FROM synthesis_quality
        {where}
        GROUP BY alias
        ORDER BY avg_score DESC
        """,
        params,
    ).fetchall()

    t = Table("Alias", "Runs", "Avg Score", "Avg Divergence", "Avg Length")
    for r in rows:
        t.add_row(r["alias"], str(r["cnt"]), str(r["avg_score"]), str(r["avg_div"]), str(int(r["avg_len"])))
    console.print(t)

    # Recent individual rows
    recent = conn.execute(
        f"""
        SELECT run_id, alias, score, divergence_index, created_at
        FROM synthesis_quality
        {where}
        ORDER BY created_at DESC
        LIMIT ?
        """,
        params + (limit,),
    ).fetchall()

    t2 = Table("Run ID", "Alias", "Score", "Divergence", "Created")
    for r in recent:
        t2.add_row(r["run_id"][:8], r["alias"], str(r["score"]), str(r["divergence_index"]), r["created_at"])
    console.print(t2)
    conn.close()


# ─── Daemon Commands ────────────────────────────────────────────────────────

@app.command("daemon")
def daemon_cmd(
    host: str = typer.Option("127.0.0.1", help="Host to bind"),
    port: int = typer.Option(8765, help="Port to bind"),
):
    """Start the Orchestra WebSocket server for Unity integration."""
    import asyncio
    from orchestra.server import start_server
    
    console.print(f"[bold green]Orchestra Daemon is starting on {host}:{port}...[/bold green]")
    try:
        asyncio.run(start_server(host, port))
    except KeyboardInterrupt:
        console.print("\n[yellow]Daemon stopped by user.[/yellow]")


if __name__ == "__main__":
    app()

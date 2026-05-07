"""Orchestra CLI — typer-based entry point."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="orchestra", help="Multi-agent orchestration CLI.")
console = Console()


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _format_elapsed(start_time: str, end_time: Optional[str]) -> str:
    if not start_time:
        return "--:--"
    start = datetime.fromisoformat(start_time)
    end = datetime.fromisoformat(end_time) if end_time else datetime.now(timezone.utc)
    total_seconds = int((end - start).total_seconds())
    if total_seconds < 0:
        return "--:--"
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"


def _percent(part: int, whole: int) -> int:
    if whole <= 0:
        return 0
    return round((part / whole) * 100)


def _ensure_history_index() -> int:
    from orchestra.storage.db import backfill

    return backfill()


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
        console.print(log_file.read_text(encoding="utf-8"))

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
):
    """Send a single task to one agent alias."""
    from orchestra.engine.runner import run_ask

    run = run_ask(alias, task)
    if run.agents and run.agents[0].stdout_log and raw:
        console.print(run.agents[0].stdout_log)


@app.command()
def run(
    mode: str = typer.Argument(..., help="Execution mode: auto | dual | critical | planned"),
    task: str = typer.Argument(..., help="Task prompt"),
    require_approval: bool = typer.Option(
        False,
        "--require-approval",
        help="Pause critical runs after round 1 and wait for approval before round 2",
    ),
):
    """Run auto-routed or explicit multi-agent modes."""
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
):
    """(STUB) Execute a speculative plan in worktrees."""
    console.print(f"[yellow]Speculative execution for {plan_id} is not yet implemented.[/yellow]")


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
def ps():
    """List recent runs."""
    from orchestra.engine.artifacts import list_runs

    runs = list_runs()
    if not runs:
        console.print("[dim]No runs found.[/dim]")
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
):
    """Print agent logs for a specific run."""
    _render_logs(run_id, agent)


@app.command()
def aliases():
    """Show available agent aliases and their status."""
    from orchestra.providers.fallback import available_aliases

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
    task: str = typer.Argument(..., help="Task to evaluate across different modes"),
):
    """Evaluate task across ask, dual, and critical modes to compare efficiency."""
    from orchestra.engine.runner import run_ask, run_dual, run_critical
    import time
    
    modes = ["ask", "dual", "critical"]
    results = []
    
    console.print(f"\n[bold cyan]Starting evaluation for task:[/bold cyan] {task[:100]}...\n")
    
    for mode in modes:
        start_time = time.time()
        if mode == "ask":
            run = run_ask("cdx-deep", task, emit_console=False, show_live=False)
        elif mode == "dual":
            run = run_dual(task, emit_console=False, show_live=False, synthesize=True)
        else: # critical
            run = run_critical(task, emit_console=False, show_live=False)
            
        elapsed = time.time() - start_time
        
        results.append({
            "mode": mode,
            "id": run.run_id,
            "status": run.status.value,
            "confidence": run.avg_confidence,
            "cost": f"{run.total_cost_usd:.4f}$",
            "time": f"{elapsed:.1f}s"
        })
        
    table = Table(title="Mode Evaluation Report", box=None)
    table.add_column("MODE", style="cyan")
    table.add_column("STATUS", style="white")
    table.add_column("CONFIDENCE", style="green")
    table.add_column("COST", style="yellow")
    table.add_column("TIME", style="magenta")
    
    for res in results:
        table.add_row(res["mode"], res["status"], f"{res['confidence']:.2f}", res["cost"], res["time"])
        
    console.print(table)
    console.print("\n[dim]Use 'orchestra ps' to see full run details.[/dim]")


# ─── Observability Commands ──────────────────────────────────────────────────

@app.command("diff")
def diff_cmd(
    run_a: str = typer.Argument(..., help="First Run ID or 'alias' in same run"),
    run_b: str = typer.Option(None, "--with", help="Second Run ID"),
    agent_a: str = typer.Option(None, "--agent-a", help="First agent alias"),
    agent_b: str = typer.Option(None, "--agent-b", help="Second agent alias"),
):
    """Semantically compare two runs or two agents within the same run."""
    from orchestra.engine.artifacts import read_agent_log
    from orchestra.engine.runner import run_ask
    
    # Logic for comparing same run agents or different runs
    log_a = ""
    log_b = ""
    
    if run_b: # Compare different runs
        log_a = read_agent_log(run_a, agent_a or "cdx-deep") or ""
        log_b = read_agent_log(run_b, agent_b or "cdx-deep") or ""
    else: # Compare agents in same run
        log_a = read_agent_log(run_a, agent_a or "cdx-deep") or ""
        log_b = read_agent_log(run_a, agent_b or "gmn-pro") or ""

    if not log_a or not log_b:
        console.print("[red]Could not find logs for comparison.[/red]")
        return

    diff_prompt = f"""\
Aşağıdaki iki AI çıktısını karşılaştır. 
1. Farklılıkları listele.
2. Hangisi daha stabil ve projenin mimarisine uygun?
3. Kritik eksiklikleri belirt.

ÇIKTI A:
{log_a[:3000]}

ÇIKTI B:
{log_b[:3000]}
"""
    console.print(f"[bold cyan]Comparing {run_a} vs {run_b or run_a}...[/bold cyan]")
    # Run a hidden ask for diffing
    diff_run = run_ask("gmn-pro", diff_prompt, emit_console=False, show_live=False, reflect=False)
    if diff_run.agents:
        console.rule("Semantic Diff Report")
        console.print(diff_run.agents[0].stdout_log)

@app.command("checkpoints")
def list_checkpoints_cmd(run_id: str = typer.Argument(...)):
    """List all snapshots (checkpoints) for a specific run."""
    from orchestra.engine.artifacts import list_checkpoints
    
    snaps = list_checkpoints(run_id)
    if not snaps:
        console.print(f"[yellow]No checkpoints found for run {run_id}.[/yellow]")
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
    """Revert all file system changes made by a specific run."""
    from orchestra.engine.backtrack import undo_last_run
    
    if typer.confirm(f"Are you sure you want to revert changes for run {run_id}?"):
        if undo_last_run(run_id):
            console.print(f"[green]Successfully reverted changes for run {run_id}.[/green]")
        else:
            console.print(f"[red]Failed to undo. Snapshot might be missing or expired.[/red]")


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

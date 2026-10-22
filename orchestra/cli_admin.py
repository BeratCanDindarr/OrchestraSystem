"""Admin commands: statusline, config, memory, aliases, models, backends, backfill, version
and doctor. Registered on `orchestra.cli.app` when `cli` imports this module last."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import typer
from orchestra import config
from rich.table import Table

from orchestra.utils.config_values import flatten_toml_lines as _flatten_toml_lines
from orchestra.utils.config_values import parse_cli_value as _parse_cli_value

from orchestra.cli import (
    console,
    app,
    _config_at_path,
    _ensure_history_index,
    _get_config_tree,
    _print_json,
    _set_config_path,
)


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
):
    """Search semantic memory for relevant snippets."""
    from orchestra.storage.memory import get_memory, search_notes

    mem = get_memory()
    results = search_notes(mem, query, limit=limit)
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

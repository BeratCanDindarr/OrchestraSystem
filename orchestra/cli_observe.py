"""Observability commands: trace, eval, timeline, diff, replay, checkpoints, undo, eval-stats
and report. Registered on `orchestra.cli.app` when `cli` imports this module last."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional
import typer
from orchestra import config
from rich.table import Table

from orchestra.cli import (
    console,
    app,
    _print_json,
    _require_run,
)


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

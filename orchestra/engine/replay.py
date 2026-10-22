from __future__ import annotations
"""Deterministic replay system — recording and replaying runs.

Every run that has replay enabled leaves an append-only JSONL log at
``.orchestra/runs/<run_id>/replay.jsonl``.  The log contains ordered
agent dispatches, tool calls, and synthesis steps.  ``ReplayRunner``
reads this log to reconstruct the execution path, optionally using
cached tool outputs and deterministic sampling parameters.
"""

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestra import config

logger = logging.getLogger(__name__)

# Module-level cache of open recorders per run_id
_recorders: dict[str, ReplayRecorder] = {}
_lock = threading.Lock()


def get_recorder(run_id: str) -> "ReplayRecorder":
    """Return the recorder for *run_id*, creating it if necessary."""
    with _lock:
        if run_id not in _recorders:
            _recorders[run_id] = ReplayRecorder(run_id)
        return _recorders[run_id]


def _replay_path(run_id: str) -> Path:
    """Path to the replay JSONL file for a run."""
    return config.artifact_root() / run_id / "replay.jsonl"


class ReplayRecorder:
    """Append-only recorder for a single run."""

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._seq = 0
        self._path = _replay_path(run_id)

    def _append(self, entry: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._seq += 1
        entry["seq"] = self._seq
        entry["ts"] = datetime.now(timezone.utc).isoformat()
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Public recording API ───────────────────────────────────────────────

    def record_run_started(self, mode: str, task: str) -> None:
        if not config.replay_enabled():
            return
        self._append({"type": "run_started", "mode": mode, "task": task})

    def record_agent_dispatch(self, alias: str, prompt: str, model: str, provider: str) -> None:
        if not config.replay_enabled():
            return
        self._append({
            "type": "agent_dispatch",
            "alias": alias,
            "prompt": prompt,
            "model": model,
            "provider": provider,
        })

    def record_agent_completed(self, alias: str, stdout: str, returncode: int) -> None:
        if not config.replay_enabled():
            return
        from orchestra.engine.artifacts import compute_digest
        digest = compute_digest(stdout)
        self._append({
            "type": "agent_completed",
            "alias": alias,
            "stdout_digest": f"sha256:{digest}",
            "returncode": returncode,
        })

    def record_tool_call(self, tool: str, args: dict, result: Any) -> None:
        if not config.replay_enabled():
            return
        from orchestra.engine.pii_scrubber import PiiScrubber
        scrubbed_args = PiiScrubber.scrub_dict(args)
        scrubbed_result = PiiScrubber.scrub_dict(result)
        result_str = json.dumps(scrubbed_result, ensure_ascii=False)
        if len(result_str) > 2000:
            digest = hashlib.sha256(result_str.encode("utf-8")).hexdigest()[:16]
            self._append({
                "type": "tool_call",
                "tool": tool,
                "args": scrubbed_args,
                "result_digest": f"sha256:{digest}",
            })
        else:
            self._append({
                "type": "tool_call",
                "tool": tool,
                "args": scrubbed_args,
                "result": scrubbed_result,
            })

    def record_synthesis(self, alias: str, prompt: str) -> None:
        if not config.replay_enabled():
            return
        self._append({"type": "synthesis", "alias": alias, "prompt": prompt})

    def record_consensus_graph(self, graph_dict: dict) -> None:
        if not config.replay_enabled():
            return
        self._append({"type": "consensus_graph", "graph": graph_dict})

    def record_run_finalized(self, status: str, summary: str | None) -> None:
        if not config.replay_enabled():
            return
        digest = None
        if summary:
            digest = hashlib.sha256(summary.encode("utf-8")).hexdigest()[:16]
        self._append({
            "type": "run_finalized",
            "status": status,
            "summary_digest": f"sha256:{digest}" if digest else None,
        })


class ReplayRunner:
    """Reconstruct a run from its replay log."""

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._path = _replay_path(run_id)
        self._tool_cache: dict[tuple[str, str], Any] = {}

    def _load_entries(self) -> list[dict]:
        if not self._path.exists():
            return []
        entries: list[dict] = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def _build_tool_cache(self, entries: list[dict]) -> None:
        for entry in entries:
            if entry.get("type") != "tool_call":
                continue
            tool = entry["tool"]
            args = json.dumps(entry.get("args", {}), sort_keys=True)
            result = entry.get("result")
            if result is not None:
                self._tool_cache[(tool, args)] = result

    def replay(self, *, deterministic: bool = False) -> None:
        """Replay the run from its log.

        Replays agent dispatches sequentially.  When a recorded
        ``stdout_digest`` is present and the cached agent log still
        matches, the dispatch is skipped.  Tool outputs are served from
        the replay log cache where available.

        Args:
            deterministic: When ``True``, providers that support it will
                receive ``temperature=0`` and a fixed seed.
        """
        entries = self._load_entries()
        if not entries:
            logger.warning("No replay log found for run %s", self._run_id)
            return

        self._build_tool_cache(entries)

        if deterministic:
            logger.info("Deterministic replay: seeding random with run_id hash")
            import random
            deterministic_seed = int(hashlib.sha256(self._run_id.encode()).hexdigest(), 16) % (2 ** 32)
            random.seed(deterministic_seed)

        # Lazy imports to avoid circular dependency
        from collections import deque
        from orchestra.engine.runners import run_ask
        from orchestra.engine.artifacts import compute_digest, read_agent_log

        completed_queue = deque(e for e in entries if e.get("type") == "agent_completed")
        skipped_outputs: dict[str, str] = {}

        for entry in entries:
            etype = entry.get("type")
            if etype == "agent_dispatch":
                alias = entry["alias"]
                prompt = entry["prompt"]
                logger.info("Replaying agent_dispatch: %s", alias)

                recorded_digest = None
                if completed_queue:
                    completed_entry = completed_queue.popleft()
                    if completed_entry.get("alias") == alias:
                        recorded_digest = completed_entry.get("stdout_digest")
                    else:
                        completed_queue.appendleft(completed_entry)

                should_skip = False
                if recorded_digest:
                    cached_stdout = read_agent_log(self._run_id, alias)
                    if cached_stdout is not None:
                        cached_digest = f"sha256:{compute_digest(cached_stdout)}"
                        if cached_digest == recorded_digest:
                            logger.info("Digest match for %s, skipping re-execution", alias)
                            should_skip = True
                            skipped_outputs[alias] = cached_stdout

                if not should_skip:
                    if deterministic:
                        deterministic_seed = int(hashlib.sha256(self._run_id.encode()).hexdigest(), 16) % 2_147_483_647
                        result = run_ask(
                            alias, prompt, emit_console=False, show_live=False,
                            temperature=0.0, seed=deterministic_seed,
                        )
                    else:
                        result = run_ask(alias, prompt, emit_console=False, show_live=False)

                    if recorded_digest and result.agents:
                        new_stdout = result.agents[0].stdout_log
                        new_digest = f"sha256:{compute_digest(new_stdout)}"
                        if new_digest != recorded_digest:
                            logger.warning(
                                "Model drift detected for %s: recorded digest %s != new digest %s",
                                alias, recorded_digest, new_digest,
                            )
                        else:
                            logger.info("Deterministic replay: digest match for %s after re-execution", alias)

            elif etype == "synthesis":
                alias = entry["alias"]
                prompt = entry["prompt"]
                logger.info("Replaying synthesis: %s", alias)
                if deterministic:
                    deterministic_seed = int(hashlib.sha256(self._run_id.encode()).hexdigest(), 16) % 2_147_483_647
                    run_ask(
                        alias, prompt, emit_console=False, show_live=False,
                        temperature=0.0, seed=deterministic_seed,
                    )
                else:
                    run_ask(alias, prompt, emit_console=False, show_live=False)

        # Rebuild consensus graph from replayed outputs
        try:
            from orchestra.engine.consensus import build_consensus_graph
            from orchestra.models import OrchestraRun, AgentRun, AgentStatus

            replay_run = OrchestraRun(run_id=self._run_id, mode="replayed", task="")
            for entry in entries:
                if entry.get("type") == "agent_completed":
                    alias = entry["alias"]
                    stdout_log = skipped_outputs.get(alias, "")
                    replay_run.agents.append(
                        AgentRun(alias=alias, provider="", model="", status=AgentStatus.COMPLETED, stdout_log=stdout_log)
                    )
            if len(replay_run.agents) >= 2:
                graph = build_consensus_graph(replay_run)
                get_recorder(self._run_id).record_consensus_graph(graph.to_dict())
        except Exception as exc:
            logger.warning("Failed to rebuild consensus graph during replay: %s", exc)

        logger.info("Replay of run %s completed", self._run_id)


def replay_synthesis(run_id: str, synthesizer: str | None = None) -> dict:
    """Synthesise a finished run again from its stored agent outputs, without asking the agents.

    Raises ValueError (with what is wrong) when the run is unknown, has fewer than two good
    answers, or the synthesiser alias is not an agent. Returns both syntheses, their cost and
    confidence, and the replay step's run id.
    """
    from orchestra import config
    from orchestra.engine.artifacts import load_manifest, read_agent_log
    from orchestra.engine.observability import resolve_run_id
    from orchestra.engine.runners import run_ask
    from orchestra.engine.synthesizer import build_synthesis_prompt, outputs_sufficient
    from orchestra.models import AgentRun, AgentStatus
    from orchestra.providers.fallback import known_aliases

    synthesizer = synthesizer or str(config.personas_config().get("synthesizer") or "cld-deep")
    resolved_id = resolve_run_id(run_id)
    manifest = load_manifest(resolved_id) if resolved_id else None
    if not manifest:
        raise ValueError(f"run_id '{run_id}' not found")
    completed = [a for a in manifest.get("agents") or [] if a.get("status") == "completed" and not a.get("soft_failed")]
    if len(completed) < 2:
        raise ValueError(f"run requires 2+ agents for synthesis (found: {len(completed)})")
    if synthesizer not in known_aliases():
        raise ValueError(f"unknown synthesizer '{synthesizer}'")
    agents: list[AgentRun] = []
    for meta in completed[:2]:
        agent = AgentRun(alias=meta.get("alias", ""), provider=meta.get("provider", ""), model=meta.get("model", ""))
        agent.status = AgentStatus.COMPLETED
        agent.confidence = float(meta.get("confidence") or 0.5)
        agent.stdout_log = read_agent_log(resolved_id, agent.alias) or ""
        agent.soft_failed = bool(meta.get("soft_failed", False))
        agent.validation_status = meta.get("validation_status", "not_run")
        agents.append(agent)
    if not outputs_sufficient(agents):
        raise ValueError("no cached agent outputs found for run")
    synth_run = run_ask(synthesizer, build_synthesis_prompt(agents, task=manifest.get("task")),
                        emit_console=False, show_live=False, internal=True, parent_run_id=resolved_id)
    if not synth_run.agents or not synth_run.agents[0].stdout_log:
        raise ValueError("synthesis produced no output")
    return {
        "run_id": resolved_id,
        "synthesizer": synthesizer,
        "replay_run_id": synth_run.run_id,
        "original_output": (manifest.get("summary") or "").strip(),
        "replayed_output": synth_run.agents[0].stdout_log,
        "original_cost_usd": float(manifest.get("total_cost_usd") or 0.0),
        "replayed_cost_usd": float(synth_run.total_cost_usd or 0.0),
        "original_confidence": float(manifest.get("avg_confidence") or 0.0),
        "replayed_confidence": float(synth_run.avg_confidence or 0.0),
        "agents": completed[:2],
    }

"""Deterministic replay system — recording and replaying runs.

Every run that has replay enabled leaves an append-only JSONL log at
``.orchestra/runs/<run_id>/replay.jsonl``.  The log contains ordered
agent dispatches, tool calls, and synthesis steps.  ``ReplayRunner``
reads this log to reconstruct the execution path, optionally using
cached tool outputs and deterministic sampling parameters.
"""
from __future__ import annotations

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
    return Path(".orchestra") / "runs" / run_id / "replay.jsonl"


class ReplayRecorder:
    """Append-only recorder for a single run."""

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._seq = 0
        self._path = _replay_path(run_id)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, entry: dict) -> None:
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
        digest = hashlib.sha256(stdout.encode("utf-8")).hexdigest()[:16]
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

        Currently replays agent dispatches sequentially.  Tool-call
        interception and stdout-digest comparison are deferred to a
        follow-up when the speculative-execution bridge is stable.

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

        # Lazy import to avoid circular dependency
        from orchestra.engine.runner import run_ask

        for entry in entries:
            etype = entry.get("type")
            if etype == "agent_dispatch":
                alias = entry["alias"]
                prompt = entry["prompt"]
                logger.info("Replaying agent_dispatch: %s", alias)
                if deterministic:
                    deterministic_seed = int(hashlib.sha256(self._run_id.encode()).hexdigest(), 16) % 2_147_483_647
                    run_ask(
                        alias, prompt, emit_console=False, show_live=False,
                        temperature=0.0, seed=deterministic_seed,
                    )
                else:
                    run_ask(alias, prompt, emit_console=False, show_live=False)
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
                    replay_run.agents.append(
                        AgentRun(alias=alias, provider="", model="", status=AgentStatus.COMPLETED)
                    )
            if len(replay_run.agents) >= 2:
                graph = build_consensus_graph(replay_run)
                self._append({"type": "consensus_graph", "graph": graph.to_dict()})
        except Exception as exc:
            logger.warning("Failed to rebuild consensus graph during replay: %s", exc)

        logger.info("Replay of run %s completed", self._run_id)

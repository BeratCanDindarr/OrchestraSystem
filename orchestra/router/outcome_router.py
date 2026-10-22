"""Outcome-weighted routing for Orchestra execution modes."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Dict, Iterable, List, Optional, Set

from orchestra import config
from orchestra.models import OrchestraRun
from orchestra.storage.db import get_db
from orchestra.utils.outcome_recorder import OutcomeRecorder

logger = logging.getLogger(__name__)


class OutcomeRouter:
    """Suggests the execution mode that worked on similar tasks before."""

    MODES = ("ask", "dual", "planned")
    MIN_SIMILARITY = 0.2     # below this a past task is about something else
    MIN_SAMPLES = 3          # fewer similar runs of a mode are not evidence
    TIE_MARGIN = 0.05        # success rates this close count as equal; the cheaper mode wins

    @staticmethod
    def jaccard_similarity(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> float:
        """Compute Jaccard similarity between two token collections.

        Args:
            tokens_a: First token collection.
            tokens_b: Second token collection.

        Returns:
            A similarity score in the range [0.0, 1.0].

        Raises:
            ValueError: If either token collection is missing.
        """
        if tokens_a is None or tokens_b is None:
            raise ValueError("tokens_a and tokens_b must not be None.")

        set_a = set(tokens_a)
        set_b = set(tokens_b)
        union = set_a | set_b
        if not union:
            return 0.0
        return len(set_a & set_b) / len(union)

    @classmethod
    def find_top_similar(
        cls,
        task_tokens: Set[str],
        history: List[Dict],
        k: int = 10,
    ) -> List[Dict]:
        """Return the top-k most similar outcome records.

        Args:
            task_tokens: Tokenized current task.
            history: Historical outcome records.
            k: Maximum number of similar records to return.

        Returns:
            The top-k records sorted by descending similarity.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not isinstance(task_tokens, set):
            raise ValueError("task_tokens must be a set of strings.")
        if not isinstance(history, list):
            raise ValueError("history must be a list of outcome records.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")

        scored_records: List[Dict] = []
        for record in history:
            if not isinstance(record, dict):
                continue
            record_tokens = record.get("task_tokens", [])
            if not isinstance(record_tokens, list):
                continue
            similarity = cls.jaccard_similarity(task_tokens, set(record_tokens))
            enriched = dict(record)
            enriched["similarity"] = similarity
            scored_records.append(enriched)

        scored_records.sort(
            key=lambda item: (item["similarity"], item.get("confidence", 0.0)),
            reverse=True,
        )
        return scored_records[:k]

    @classmethod
    def suggest(
        cls,
        task_description: str,
        history: List[Dict],
        *,
        min_history: int = 50,
        top_k: int = 10,
    ) -> Optional[str]:
        """The mode with the best record on tasks like this one, or None when the history
        does not say.

        Among the *top_k* past outcomes most similar to the task (and at least MIN_SIMILARITY
        similar), each mode with MIN_SAMPLES or more runs gets its success rate: completed
        runs over all runs. The highest rate wins; rates within TIE_MARGIN are a tie and the
        mode with the lower average cost wins. The agents' stated confidence is not used: a
        run that states none counts 0.5, which says nothing.

        Raises:
            ValueError: If the task is empty or the history is not a list.
        """
        if not isinstance(task_description, str) or not task_description.strip():
            raise ValueError("task_description must be a non-empty string.")
        if not isinstance(history, list):
            raise ValueError("history must be a list of outcome records.")
        if len(history) < min_history:
            return None

        task_tokens = OutcomeRecorder.tokenize(task_description)
        similar = [record for record in cls.find_top_similar(task_tokens, history, k=top_k)
                   if record["similarity"] >= cls.MIN_SIMILARITY]
        scores: Dict[str, tuple[float, float]] = {}
        for mode in cls.MODES:
            runs = [record for record in similar if record.get("mode") == mode]
            if len(runs) < cls.MIN_SAMPLES:
                continue
            succeeded = sum(1 for record in runs if record.get("status") == "completed")
            average_cost = sum(float(record.get("cost_usd") or 0.0) for record in runs) / len(runs)
            scores[mode] = (succeeded / len(runs), average_cost)
        if not scores:
            return None
        best_rate = max(rate for rate, _ in scores.values())
        contenders = [mode for mode, (rate, _) in scores.items() if best_rate - rate <= cls.TIE_MARGIN]
        choice = min(contenders, key=lambda mode: (scores[mode][1], cls.MODES.index(mode)))
        logger.info("Outcome routing: mode=%s scores=%s similar=%s", choice, scores, len(similar))
        return choice


_RECORDED_STATUSES = ("completed", "failed")
_CREATE_OUTCOMES = """
    CREATE TABLE IF NOT EXISTS outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        mode TEXT,
        task_hash TEXT,
        aliases TEXT,
        avg_confidence REAL,
        total_cost_usd REAL,
        status TEXT,
        created_at REAL,
        task_tokens TEXT
    )
"""


def _ensure_outcomes_table(connection) -> None:
    connection.execute(_CREATE_OUTCOMES)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(outcomes)")}
    if "task_tokens" not in columns:      # a table from before the tokens were stored
        connection.execute("ALTER TABLE outcomes ADD COLUMN task_tokens TEXT")


def append_outcome(run: OrchestraRun) -> None:
    """Record how a run ended, for `OutcomeRouter.suggest` to learn from.

    Only a task someone gave is an outcome: a step of another run (verifier, synthesiser,
    planned node) is not, and neither is a run that was cancelled.
    """
    status = run.status.value if hasattr(run.status, "value") else str(run.status)
    if run.parent_run_id or status not in _RECORDED_STATUSES:
        return
    task = run.task.strip()
    connection = get_db()
    try:
        with connection:
            _ensure_outcomes_table(connection)
            connection.execute(
                """
                INSERT INTO outcomes
                (run_id, mode, task_hash, aliases, avg_confidence, total_cost_usd, status, created_at, task_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.mode,
                    hashlib.sha256(task.encode("utf-8")).hexdigest(),
                    json.dumps([a.alias for a in run.agents]),
                    float(run.avg_confidence),
                    float(run.total_cost_usd),
                    status,
                    time.time(),
                    json.dumps(sorted(OutcomeRecorder.tokenize(task))),
                ),
            )
    finally:
        connection.close()


def load_history(limit: int = 500, max_age_days: Optional[float] = None) -> List[Dict]:
    """The most recent recorded outcomes, newest first, as `OutcomeRouter.suggest` reads them."""
    if max_age_days is None:
        max_age_days = float(config.outcome_routing_config()["time_decay_days"])
    oldest = time.time() - max_age_days * 86400
    connection = get_db()
    try:
        with connection:
            _ensure_outcomes_table(connection)
        rows = connection.execute(
            """
            SELECT mode, status, total_cost_usd, avg_confidence, created_at, task_tokens
            FROM outcomes
            WHERE task_tokens IS NOT NULL AND created_at >= ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (oldest, limit),
        ).fetchall()
    finally:
        connection.close()
    history: List[Dict] = []
    for mode, status, cost, confidence, created_at, tokens in rows:
        try:
            task_tokens = json.loads(tokens)
        except (TypeError, json.JSONDecodeError):
            continue
        history.append({"mode": mode, "status": status, "cost_usd": float(cost or 0.0),
                        "confidence": float(confidence or 0.0), "created_at": created_at,
                        "task_tokens": task_tokens})
    return history


def suggest_mode_for(task: str) -> Optional[str]:
    """The mode the history recommends for *task*; None when outcome routing is off or silent."""
    settings = config.outcome_routing_config()
    if not settings["enabled"]:
        return None
    return OutcomeRouter.suggest(
        task, load_history(), min_history=int(settings["min_history_threshold"]),
        top_k=int(settings["top_k_similar"]),
    )

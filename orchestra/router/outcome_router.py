"""Outcome-weighted routing for Orchestra execution modes."""

from __future__ import annotations

import logging
import random
from typing import Dict, Iterable, List, Set

from orchestra.utils.outcome_recorder import OutcomeRecorder

logger = logging.getLogger(__name__)


class OutcomeRouter:
    """Suggests execution modes using historical outcome expected value."""

    EPSILON = 0.0001
    EXPLORATION_RATE = 0.05
    MODES = ("ask", "dual", "critical", "planned")

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
    def compute_ev(cls, similar_tasks: List[Dict]) -> Dict[str, float]:
        """Compute expected value per mode from similar historical outcomes.

        Expected value is defined as average confidence divided by average cost.

        Args:
            similar_tasks: Similar historical outcome records.

        Returns:
            A mapping of mode name to expected value.

        Raises:
            ValueError: If similar_tasks is not a list.
        """
        if not isinstance(similar_tasks, list):
            raise ValueError("similar_tasks must be a list of outcome records.")

        ev_by_mode: Dict[str, float] = {mode: 0.0 for mode in cls.MODES}
        for mode in cls.MODES:
            mode_records = [
                record for record in similar_tasks
                if record.get("mode") == mode
                and isinstance(record.get("confidence"), (int, float))
                and isinstance(record.get("cost_usd"), (int, float))
            ]
            if not mode_records:
                continue
            avg_confidence = sum(float(record["confidence"]) for record in mode_records) / len(mode_records)
            avg_cost = sum(float(record["cost_usd"]) for record in mode_records) / len(mode_records)
            ev_by_mode[mode] = avg_confidence / (avg_cost + cls.EPSILON)
        return ev_by_mode

    @classmethod
    def suggest_mode(
        cls,
        task_description: str,
        history: List[Dict],
        min_history: int = 50,
    ) -> str:
        """Suggest the best execution mode for a task.

        Args:
            task_description: Natural-language task description.
            history: Historical outcome records.
            min_history: Minimum records required before EV routing activates.

        Returns:
            One of the supported execution modes.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not isinstance(task_description, str) or not task_description.strip():
            raise ValueError("task_description must be a non-empty string.")
        if not isinstance(history, list):
            raise ValueError("history must be a list of outcome records.")
        if not isinstance(min_history, int) or min_history < 1:
            raise ValueError("min_history must be a positive integer.")

        task_tokens = OutcomeRecorder.tokenize(task_description)
        if len(history) < min_history:
            mode = cls._keyword_fallback(task_description)
            logger.info(
                "Outcome routing fallback: insufficient history (%s/%s). mode=%s",
                len(history),
                min_history,
                mode,
            )
            return mode

        if random.random() < cls.EXPLORATION_RATE:
            mode = cls._pick_least_explored_mode(history)
            logger.info("Outcome routing exploration triggered: mode=%s", mode)
            return mode

        similar_tasks = cls.find_top_similar(task_tokens, history, k=10)
        ev_by_mode = cls.compute_ev(similar_tasks)
        best_mode = max(cls.MODES, key=lambda mode: (ev_by_mode[mode], -cls.MODES.index(mode)))
        logger.info(
            "Outcome routing selected mode=%s tokens=%s similar=%s ev=%s",
            best_mode,
            sorted(task_tokens),
            len(similar_tasks),
            ev_by_mode,
        )
        return best_mode

    @classmethod
    def _pick_least_explored_mode(cls, history: List[Dict]) -> str:
        counts = {mode: 0 for mode in cls.MODES}
        for record in history:
            mode = record.get("mode")
            if mode in counts:
                counts[mode] += 1
        lowest_count = min(counts.values())
        candidates = [mode for mode, count in counts.items() if count == lowest_count]
        return random.choice(candidates)

    @classmethod
    def _keyword_fallback(cls, task_description: str) -> str:
        text = task_description.lower()
        if any(word in text for word in ("urgent", "critical", "blocker", "production", "incident")):
            return "critical"
        if any(word in text for word in ("plan", "design", "roadmap", "architecture", "strategy")):
            return "planned"
        if any(word in text for word in ("compare", "evaluate", "tradeoff", "versus", "vs")):
            return "dual"
        return "ask"

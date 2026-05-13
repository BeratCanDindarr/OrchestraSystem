"""Outcome persistence utilities for Orchestra routing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class OutcomeRecorder:
    """Persists and retrieves routing outcomes from a JSONL history file."""

    HISTORY_PATH = os.path.expanduser("~/.orchestra/outcomes.jsonl")
    VALID_MODES = {"ask", "dual", "critical", "planned"}

    @classmethod
    def load_history(cls) -> List[Dict]:
        """Load all valid outcome records from the JSONL history file.

        Returns:
            A list of decoded outcome records. Invalid JSON lines are skipped.
        """
        if not os.path.exists(cls.HISTORY_PATH):
            return []

        records: List[Dict] = []
        try:
            with open(cls.HISTORY_PATH, "r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Skipping corrupted outcome record at line %s in %s.",
                            line_number,
                            cls.HISTORY_PATH,
                        )
                        continue
                    if isinstance(record, dict):
                        records.append(record)
        except OSError as error:
            logger.warning("Failed to read outcome history from %s: %s", cls.HISTORY_PATH, error)
            return []
        return records

    @classmethod
    def record_outcome(
        cls,
        task_text: str,
        mode: str,
        confidence: float,
        cost_usd: float,
    ) -> None:
        """Append a single outcome record with cross-platform file locking.

        Args:
            task_text: Original task description.
            mode: Selected execution mode.
            confidence: Observed confidence score in the range [0.0, 1.0].
            cost_usd: Observed run cost in USD.

        Raises:
            ValueError: If any input is invalid.
        """
        if not isinstance(task_text, str) or not task_text.strip():
            raise ValueError("task_text must be a non-empty string.")
        if mode not in cls.VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(cls.VALID_MODES)}.")
        if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be a number between 0.0 and 1.0.")
        if not isinstance(cost_usd, (int, float)) or float(cost_usd) < 0.0:
            raise ValueError("cost_usd must be a non-negative number.")

        os.makedirs(os.path.dirname(cls.HISTORY_PATH), exist_ok=True)
        record = {
            "task_hash": hashlib.sha256(task_text.strip().encode("utf-8")).hexdigest(),
            "task_tokens": sorted(cls.tokenize(task_text)),
            "mode": mode,
            "confidence": float(confidence),
            "cost_usd": float(cost_usd),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        with open(cls.HISTORY_PATH, "a+", encoding="utf-8") as handle:
            cls._lock_file(handle)
            try:
                handle.seek(0, os.SEEK_END)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
            finally:
                cls._unlock_file(handle)

    @staticmethod
    def tokenize(text: str) -> Set[str]:
        """Tokenize text into a normalized, stop-word-filtered token set.

        Args:
            text: Source text to tokenize.

        Returns:
            A set of lowercase tokens.

        Raises:
            ValueError: If text is not a string.
        """
        if not isinstance(text, str):
            raise ValueError("text must be a string.")
        return {
            token
            for token in _TOKEN_RE.findall(text.lower())
            if token and token not in _STOP_WORDS
        }

    @staticmethod
    def _lock_file(handle) -> None:
        if os.name == "nt":
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

    @staticmethod
    def _unlock_file(handle) -> None:
        if os.name == "nt":
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

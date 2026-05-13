"""Evaluation tracking module for Orchestra."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Union

from orchestra.storage.db import get_db


class EvalTracker:
    """Track evaluation runs and compute Pass@K metrics."""

    def log_run(self, run_id: str, task: str, mode: str, status: str, created_at: str) -> None:
        """Log an evaluation run outcome.

        Args:
            run_id: Unique run identifier
            task: Task description/prompt
            mode: Execution mode (ask, dual, critical, etc.)
            status: Outcome status (PASS or FAIL)
            created_at: ISO timestamp
        """
        connection = get_db()
        with connection:
            connection.execute(
                """
                INSERT INTO eval_runs (run_id, task, mode, status, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    task = excluded.task,
                    mode = excluded.mode,
                    status = excluded.status,
                    created_at = excluded.created_at
                """,
                (run_id, task, mode, status, created_at)
            )

    def calculate_pass_at_k(self, k: int) -> float:
        """Calculate the Pass@k metric across all eval runs grouped by task.

        Pass@k = 1 - (probability of all k samples failing)
        For each task: P(fail all k) = C(n-c, k) / C(n, k)
        where n = total samples, c = passed samples

        Args:
            k: Number of samples to consider

        Returns:
            Average Pass@k across all tasks (0.0 to 1.0)
        """
        connection = get_db()
        cursor = connection.execute("SELECT task, status FROM eval_runs")
        rows = cursor.fetchall()
        connection.close()

        if not rows:
            return 0.0

        # Group results by task
        tasks: dict[str, dict[str, int]] = {}
        for row in rows:
            # row is sqlite3.Row with dict-like access
            task_name = row[0] if isinstance(row, tuple) else row["task"]
            status = row[1] if isinstance(row, tuple) else row["status"]

            if task_name not in tasks:
                tasks[task_name] = {"n": 0, "c": 0}
            tasks[task_name]["n"] += 1
            if status == "PASS":
                tasks[task_name]["c"] += 1

        pass_rates = []
        for stats in tasks.values():
            n = stats["n"]
            c = stats["c"]

            # Skip tasks with fewer than k samples
            if n < k:
                continue

            # Calculate Pass@k: 1 - P(all k fail)
            if c == 0:
                # All failures -> Pass@k = 0
                pass_rates.append(0.0)
            else:
                # P(all k fail) = C(n-c, k) / C(n, k)
                numerator = math.comb(n - c, k)
                denominator = math.comb(n, k)
                prob_all_fail = numerator / denominator if denominator > 0 else 0.0
                pass_rates.append(1.0 - prob_all_fail)

        if not pass_rates:
            return 0.0

        return sum(pass_rates) / len(pass_rates)

    def export_jsonl(self, path: Union[str, Path]) -> None:
        """Export all evaluation records to a JSONL file.

        Each line is a JSON object with fields:
        - run_id: Unique identifier
        - task: Task description
        - mode: Execution mode
        - status: Outcome (PASS or FAIL)
        - created_at: ISO timestamp

        Args:
            path: Output file path
        """
        connection = get_db()
        cursor = connection.execute("SELECT run_id, task, mode, status, created_at FROM eval_runs")
        rows = cursor.fetchall()
        connection.close()

        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                # Handle both tuple and Row object formats
                if isinstance(row, tuple):
                    run_id, task, mode, status, created_at = row
                else:
                    run_id = row["run_id"]
                    task = row["task"]
                    mode = row["mode"]
                    status = row["status"]
                    created_at = row["created_at"]

                record = {
                    "run_id": run_id,
                    "task": task,
                    "mode": mode,
                    "status": status,
                    "created_at": created_at,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

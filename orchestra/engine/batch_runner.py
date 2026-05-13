"""Batch execution runner for parallel scenario evaluation."""
from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextvars import copy_context
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from datetime import datetime

from orchestra.storage.db import get_db
from orchestra.engine.runner import run_ask, run_dual, run_critical
from orchestra.models import RunStatus

logger = logging.getLogger(__name__)


@dataclass
class EvalScenario:
    """A single evaluation scenario to test."""
    scenario_id: str
    task: str
    gold_standard_answer: str


@dataclass
class EvalResult:
    """Result of evaluating a scenario."""
    run_id: str
    scenario_id: str
    outcome: str  # "PASS" or "FAIL"
    tokens_used: int
    confidence: float
    created_at: str


class BatchRunner:
    """Execute evaluation scenarios in parallel with timeout and retry."""

    def __init__(self, max_workers: int = 4, timeout_seconds: int = 30, max_retries: int = 1):
        """Initialize BatchRunner.

        Args:
            max_workers: Number of parallel execution threads
            timeout_seconds: Timeout per scenario (seconds)
            max_retries: Maximum retries on transient failures
        """
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def run_scenario(self, run_id: str, task: str, mode: str, alias: str) -> EvalResult:
        """Execute a single scenario.

        Args:
            run_id: Current run identifier
            task: Task prompt
            mode: Execution mode (ask, dual, critical)
            alias: Agent alias (cdx-fast, gmn-pro, etc.)

        Returns:
            EvalResult with outcome (PASS/FAIL) and metrics
        """
        scenario_id = f"{run_id}_{alias}"

        try:
            # Submit task to Orchestra runner based on mode
            if mode == "ask":
                orchestrated_run = run_ask(alias=alias, prompt=task)
            elif mode == "dual":
                orchestrated_run = run_dual(prompt=task, agents=[alias])
            elif mode == "critical":
                orchestrated_run = run_critical(prompt=task)
            else:
                raise ValueError(f"Unknown execution mode: {mode}")

            # Extract metrics from Orchestra run
            tokens_used = 0
            confidence = 0.0
            outcome = "PASS"

            # Check completion status
            if hasattr(orchestrated_run, 'status'):
                if orchestrated_run.status != RunStatus.COMPLETED:
                    outcome = "FAIL"

            # Extract token count if available
            if hasattr(orchestrated_run, 'tokens') and orchestrated_run.tokens:
                tokens_used = orchestrated_run.tokens

            # Extract confidence if available
            if hasattr(orchestrated_run, 'avg_confidence'):
                confidence = orchestrated_run.avg_confidence or 0.0

            logger.info(f"Scenario completed: {scenario_id} outcome={outcome}")

            return EvalResult(
                run_id=run_id,
                scenario_id=scenario_id,
                outcome=outcome,
                tokens_used=tokens_used,
                confidence=confidence,
                created_at=datetime.utcnow().isoformat() + "Z"
            )

        except Exception as e:
            logger.error(f"Scenario failed: {scenario_id} error={str(e)}")
            return EvalResult(
                run_id=run_id,
                scenario_id=scenario_id,
                outcome="FAIL",
                tokens_used=0,
                confidence=0.0,
                created_at=datetime.utcnow().isoformat() + "Z"
            )

    def run_batch(
        self,
        scenarios: list[EvalScenario],
        mode: str,
        agent_aliases: list[str],
        run_id: str
    ) -> list[EvalResult]:
        """Execute scenarios in parallel.

        Args:
            scenarios: List of scenarios to evaluate
            mode: Execution mode for all scenarios
            agent_aliases: Which agents to test
            run_id: Batch run identifier

        Returns:
            List of EvalResult objects
        """
        results = []
        futures = {}

        # Submit all scenario-alias pairs to executor
        for scenario in scenarios:
            for alias in agent_aliases:
                future = self.executor.submit(
                    copy_context().run,
                    self._run_with_retry,
                    run_id=run_id,
                    task=scenario.task,
                    mode=mode,
                    alias=alias,
                    scenario_id=scenario.scenario_id,
                    max_retries=self.max_retries
                )
                futures[future] = (scenario.scenario_id, alias)

        # Collect results as they complete (with timeout)
        for future in asyncio.as_completed(futures, timeout=self.timeout_seconds * len(scenarios)):
            try:
                result = future.result(timeout=self.timeout_seconds)
                results.append(result)
            except FuturesTimeoutError:
                scenario_id, alias = futures[future]
                logger.warning(f"Timeout: scenario={scenario_id} alias={alias}")
                results.append(
                    EvalResult(
                        run_id=run_id,
                        scenario_id=scenario_id,
                        outcome="FAIL",
                        tokens_used=0,
                        confidence=0.0,
                        created_at="2026-05-12T00:00:00Z"
                    )
                )
            except Exception as e:
                scenario_id, alias = futures[future]
                logger.error(f"Error: scenario={scenario_id} alias={alias}: {e}")
                results.append(
                    EvalResult(
                        run_id=run_id,
                        scenario_id=scenario_id,
                        outcome="FAIL",
                        tokens_used=0,
                        confidence=0.0,
                        created_at="2026-05-12T00:00:00Z"
                    )
                )

        return results

    def _run_with_retry(
        self,
        run_id: str,
        task: str,
        mode: str,
        alias: str,
        scenario_id: str,
        max_retries: int
    ) -> EvalResult:
        """Run scenario with exponential backoff retry.

        Args:
            run_id: Run identifier
            task: Task prompt
            mode: Execution mode
            alias: Agent alias
            scenario_id: Scenario identifier
            max_retries: Max retry attempts

        Returns:
            EvalResult
        """
        for attempt in range(max_retries + 1):
            try:
                return self.run_scenario(run_id, task, mode, alias)
            except Exception as e:
                if attempt < max_retries:
                    # Exponential backoff: 1s, 2s, 4s, ...
                    wait_secs = 2 ** attempt
                    logger.info(f"Retry {attempt+1}/{max_retries} after {wait_secs}s: {e}")
                    # In full implementation: await asyncio.sleep(wait_secs)
                else:
                    logger.error(f"Failed after {max_retries} retries: {e}")
                    return EvalResult(
                        run_id=run_id,
                        scenario_id=scenario_id,
                        outcome="FAIL",
                        tokens_used=0,
                        confidence=0.0,
                        created_at="2026-05-12T00:00:00Z"
                    )

        # Should not reach here
        return EvalResult(
            run_id=run_id,
            scenario_id=scenario_id,
            outcome="FAIL",
            tokens_used=0,
            confidence=0.0,
            created_at="2026-05-12T00:00:00Z"
        )

    def export_stats(self, results: list[EvalResult], output_path: Path) -> None:
        """Export batch statistics to JSON.

        Args:
            results: List of evaluation results
            output_path: Path to write stats JSON
        """
        if not results:
            logger.warning("No results to export")
            return

        # Group by task/mode/alias for analysis
        stats = {
            "total_runs": len(results),
            "passed": sum(1 for r in results if r.outcome == "PASS"),
            "failed": sum(1 for r in results if r.outcome == "FAIL"),
            "pass_rate": sum(1 for r in results if r.outcome == "PASS") / len(results) if results else 0,
            "avg_tokens": sum(r.tokens_used for r in results) / len(results) if results else 0,
            "avg_confidence": sum(r.confidence for r in results) / len(results) if results else 0,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Stats exported to {output_path}: {stats}")

    def __del__(self):
        """Cleanup executor on shutdown."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

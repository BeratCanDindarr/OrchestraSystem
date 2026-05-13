"""Idempotency Wrapper: Check EventLog for prior execution and return cached result."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from functools import wraps
from typing import Any, Callable, Optional

from orchestra.storage.event_log import EventLog

logger = logging.getLogger(__name__)


@dataclass
class IdempotencyKey:
    """Immutable idempotency key with SHA-256 hash computation."""

    run_id: str
    operation: str
    params_hash: str

    @staticmethod
    def compute(run_id: str, operation: str, params: dict) -> str:
        """
        Compute SHA-256 hash of (run_id, operation, params).

        Args:
            run_id: Unique run identifier
            operation: Operation name
            params: Parameters dict to hash

        Returns:
            SHA-256 hash string (hexdigest)
        """
        data = json.dumps(
            {"run_id": run_id, "operation": operation, "params": params},
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()


def idempotent(
    operation: str,
    key_fn: Optional[Callable[..., str]] = None,
    ttl_hours: int = 48,
    event_log: Optional[EventLog] = None,
):
    """
    Decorator for idempotent operations.

    Checks EventLog for prior cache_hit event with same operation and key.
    If found and not expired, returns cached result.
    Otherwise executes and logs cache_miss event.

    Args:
        operation: Operation name (used as payload field for tracking)
        key_fn: Optional function to compute custom key from *args, **kwargs.
                If None, uses json.dumps of kwargs.
        ttl_hours: Time-to-live for cached results in hours (default 48)
        event_log: Optional EventLog instance for caching.
                   If None, no caching is performed.

    Returns:
        Decorator function

    Example:
        @idempotent(
            operation="run_agent",
            key_fn=lambda alias, prompt: f"{alias}:{hashlib.sha256(prompt.encode()).hexdigest()}",
            event_log=event_log
        )
        def run_agent_idempotent(run_id: str, alias: str, prompt: str) -> AgentRun:
            return run_agent(alias, prompt)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(run_id: str, *args, **kwargs) -> Any:
            # Compute idempotency key
            if key_fn:
                key_str = key_fn(*args, **kwargs)
            else:
                key_str = json.dumps(kwargs, sort_keys=True)

            key_hash = hashlib.sha256(key_str.encode()).hexdigest()

            # Check EventLog for prior cache_miss (if event_log provided)
            if event_log:
                # Compute TTL cutoff timestamp
                now = datetime.now(timezone.utc)
                ttl_cutoff = now - timedelta(hours=ttl_hours)
                ttl_cutoff_ts = ttl_cutoff.timestamp()

                # Replay events for this run to find cached result
                for event in event_log.replay(run_id):
                    # Check if event is cache_miss matching operation and key (represents a prior execution)
                    if (
                        event.event_type == "cache_miss"
                        and event.ts >= ttl_cutoff_ts
                        and event.payload.get("operation") == operation
                    ):
                        payload_key = event.payload.get("idempotency_key")
                        if payload_key == key_hash:
                            # Cache hit: return cached result from prior execution
                            logger.debug(
                                f"Idempotent cache hit: {operation} ({key_hash[:8]}...)"
                            )
                            result = event.payload.get("result")
                            return result

            # Cache miss: execute function
            result = func(run_id, *args, **kwargs)

            # Log cache_miss event with result to EventLog
            if event_log:
                try:
                    event_log.append(
                        "cache_miss",
                        run_id,
                        {
                            "operation": operation,
                            "idempotency_key": key_hash,
                            "result": result,
                        },
                    )
                    logger.debug(
                        f"Idempotent cache miss: {operation} ({key_hash[:8]}...) executed and cached"
                    )
                except Exception as e:
                    # Log error but don't fail the operation
                    logger.warning(
                        f"Failed to cache idempotent result: {operation} ({key_hash[:8]}...): {e}"
                    )

            return result

        return wrapper

    return decorator

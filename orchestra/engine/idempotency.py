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


def _resolve_event_log(log):
    """Resolve event_log, calling it if it's a factory function."""
    if callable(log):
        return log()
    return log


def idempotent(
    operation: str,
    key_fn: Optional[Callable[..., str]] = None,
    ttl_hours: int = 48,
    event_log: Optional[EventLog] = None,
    run_id: Optional[str] = None,
    serialize: Optional[Callable[[Any], Any]] = None,
    deserialize: Optional[Callable[[Any], Any]] = None,
    should_cache: Optional[Callable[[Any], bool]] = None,
    bypass_kwarg: Optional[str] = None,
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
        event_log: Optional EventLog instance (or factory) for caching.
                   If None, no caching is performed.
        run_id: Optional fixed run_id. If None, uses first positional arg.
        serialize: Optional function to serialize result before caching.
        deserialize: Optional function to deserialize cached result.
        should_cache: Optional test of a result. Only results that pass are stored, and a
                      stored result that fails it is not replayed (an unfinished outcome
                      must not stand in for a new attempt).
        bypass_kwarg: Optional name of a keyword argument; when the caller passes it truthy
                      the stored result is ignored, the operation runs, and its result
                      replaces the stored one for later calls.

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
        def wrapper(*args, **kwargs) -> Any:
            # Determine run_id: fixed or first positional arg
            if run_id is not None:
                actual_run_id = run_id
                func_args = args
            else:
                actual_run_id = args[0]
                func_args = args[1:]

            # Compute idempotency key
            if key_fn:
                key_str = key_fn(*func_args, **kwargs)
            else:
                key_str = json.dumps(kwargs, sort_keys=True)

            key_hash = hashlib.sha256(key_str.encode()).hexdigest()

            log_instance = _resolve_event_log(event_log)

            bypass = bool(bypass_kwarg and kwargs.get(bypass_kwarg))

            # Check EventLog for prior cache_miss (if event_log provided)
            if log_instance and not bypass:
                # Compute TTL cutoff timestamp
                now = datetime.now(timezone.utc)
                ttl_cutoff = now - timedelta(hours=ttl_hours)
                ttl_cutoff_ts = ttl_cutoff.timestamp()
                # Expired results can never be used: removed here, a store keeps one time-to-live
                # of results instead of every call ever made. Only a store of its own (a fixed
                # run_id): when the key is a real run's id, its events are that run's history.
                if run_id is not None:
                    try:
                        log_instance.prune(actual_run_id, ttl_cutoff_ts)
                    except Exception:
                        logger.warning("Could not prune expired results of %s", operation, exc_info=True)

                # Replay events for this run to find cached result. The newest one wins:
                # a bypassed call stores a second result under the same key.
                latest = None
                for event in log_instance.replay(actual_run_id):
                    # Check if event is cache_miss matching operation and key (represents a prior execution)
                    if (
                        event.event_type == "cache_miss"
                        and event.ts >= ttl_cutoff_ts
                        and event.payload.get("operation") == operation
                    ):
                        payload_key = event.payload.get("idempotency_key")
                        if payload_key == key_hash:
                            latest = event
                if latest is not None:
                    cached = latest.payload.get("result")
                    restored = deserialize(cached) if deserialize else cached
                    if should_cache is None or should_cache(restored):
                        logger.debug(f"Idempotent cache hit: {operation} ({key_hash[:8]}...)")
                        return restored

            # Cache miss: execute function
            result = func(*args, **kwargs)

            # Log cache_miss event with result to EventLog
            if log_instance and (should_cache is None or should_cache(result)):
                try:
                    cache_result = serialize(result) if serialize else result
                    log_instance.append(
                        "cache_miss",
                        actual_run_id,
                        {
                            "operation": operation,
                            "idempotency_key": key_hash,
                            "result": cache_result,
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

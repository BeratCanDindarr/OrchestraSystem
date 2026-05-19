"""Lightweight in-process pub-sub event bus for Orchestra.

Usage:
    from orchestra.events import subscribe, emit

    def my_handler(run_id: str, event: dict) -> None:
        print(run_id, event.get("event"))

    subscribe("run_started", my_handler)   # specific event type
    subscribe_all(my_handler)              # all events
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

Handler = Callable[[str, dict], None]  # (run_id, event_dict) -> None

_lock = threading.Lock()
_subscribers: dict[str, list[Handler]] = defaultdict(list)
_global: list[Handler] = []
_executor: ThreadPoolExecutor | None = None

_bus_logger = logging.getLogger("orchestra.events.bus")


def log_subscriber(run_id: str, event: dict) -> None:
    """Log every emitted event at INFO level."""
    event_type = event.get("event", "unknown")
    _bus_logger.info("Event [%s] run_id=%s %s", event_type, run_id, event)


def _get_executor() -> ThreadPoolExecutor:
    """Lazy-init a bounded thread-pool so slow handlers do not block the emitter."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="orchestra_bus"
        )
    return _executor


def subscribe(event_type: str, handler: Handler) -> None:
    """Subscribe handler to a specific event type."""
    with _lock:
        _subscribers[event_type].append(handler)


def subscribe_all(handler: Handler) -> None:
    """Subscribe handler to every emitted event."""
    with _lock:
        _global.append(handler)


def unsubscribe(event_type: str, handler: Handler) -> None:
    """Remove a specific-type subscription (no-op if not found)."""
    with _lock:
        try:
            _subscribers[event_type].remove(handler)
        except ValueError:
            pass


def unsubscribe_all(handler: Handler) -> None:
    """Remove a global subscription (no-op if not found)."""
    with _lock:
        try:
            _global.remove(handler)
        except ValueError:
            pass


def _run_handler(handler: Handler, run_id: str, event: dict) -> None:
    """Wrapper that catches handler exceptions so one misbehaving subscriber cannot crash others."""
    try:
        handler(run_id, event)
    except Exception:
        pass


def emit(run_id: str, event: dict) -> None:
    """Emit event to all registered handlers.

    Handlers run in a bounded thread-pool so a slow subscriber cannot
    disrupt the main execution path.
    """
    event_type = str(event.get("event", ""))
    with _lock:
        typed = list(_subscribers.get(event_type, []))
        glob = list(_global)
    handlers = typed + glob
    if not handlers:
        return
    executor = _get_executor()
    for handler in handlers:
        executor.submit(_run_handler, handler, run_id, event)


def clear() -> None:
    """Remove all subscribers and tear down the executor. Primarily useful in tests."""
    global _executor
    with _lock:
        _subscribers.clear()
        _global.clear()
    if _executor is not None:
        _executor.shutdown(wait=False)
        _executor = None


# Register logging subscriber at module load time
subscribe_all(log_subscriber)

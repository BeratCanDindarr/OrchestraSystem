"""Lightweight in-process pub-sub event bus for Orchestra.

Usage:
    from orchestra.events import subscribe, emit

    def my_handler(run_id: str, event: dict) -> None:
        print(run_id, event.get("event"))

    subscribe("run_started", my_handler)   # specific event type
    subscribe_all(my_handler)              # all events
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Callable

Handler = Callable[[str, dict], None]  # (run_id, event_dict) -> None

_lock = threading.Lock()
_subscribers: dict[str, list[Handler]] = defaultdict(list)
_global: list[Handler] = []


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


def emit(run_id: str, event: dict) -> None:
    """Emit event to all registered handlers.

    Errors inside handlers are silently swallowed so a misbehaving
    subscriber cannot disrupt the main execution path.
    """
    event_type = str(event.get("event", ""))
    with _lock:
        typed = list(_subscribers.get(event_type, []))
        glob = list(_global)
    for handler in typed + glob:
        try:
            handler(run_id, event)
        except Exception:
            pass


def clear() -> None:
    """Remove all subscribers. Primarily useful in tests."""
    with _lock:
        _subscribers.clear()
        _global.clear()

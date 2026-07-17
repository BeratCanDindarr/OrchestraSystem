from __future__ import annotations
"""Background federation sync daemon.

Periodically syncs the outbox with configured peers and garbage-collects expired
summaries, activating federation end-to-end. Fully DI'd (``sleep_fn`` / ``should_continue``
/ ``now_fn``) so the loop is deterministic and testable without real time or an
infinite asyncio task. Gated by ``config.federation_sync_enabled()`` (default off).
"""
import asyncio
import logging
import time
from typing import Any, Callable, Optional

from orchestra import config

logger = logging.getLogger(__name__)


async def sync_tick(*, now_fn: Callable[[], float] = time.time) -> dict:
    """One sync cycle: push/pull with all peers, then GC expired summaries."""
    from orchestra.federation_server.federation_client import sync_all_peers
    from orchestra.engine.federation import FederationBus
    result = await sync_all_peers()
    cleaned = FederationBus().cleanup_expired(now_fn())
    return {"sync": result, "cleaned": cleaned}


def _warn_peers_needing_refresh(now: float) -> None:
    """WARN-log peers whose auth token is near expiry (operator should rotate)."""
    from orchestra.federation_server.federation_client import _peer_from_config
    margin = config.federation_token_refresh_margin_seconds()
    for idx, url in enumerate(config.federation_config().get("peer_urls", [])):
        peer = _peer_from_config(url, idx)
        if peer.needs_refresh(now, margin):
            logger.warning(
                "Federation peer %s auth token nears expiry — rotate it in config.", peer.peer_id
            )


async def run_sync_loop(
    *,
    interval: Optional[int] = None,
    sleep_fn: Any = None,
    should_continue: Optional[Callable[[], bool]] = None,
    now_fn: Callable[[], float] = time.time,
) -> None:
    """Run sync ticks every ``interval`` seconds until ``should_continue()`` is False.

    A failing tick is logged and swallowed so a transient error never kills the loop.
    """
    if interval is None:
        interval = int(config.federation_config().get("sync_interval_seconds", 300))
    if sleep_fn is None:
        sleep_fn = asyncio.sleep
    if should_continue is None:
        should_continue = lambda: True  # noqa: E731
    while should_continue():
        try:
            await sync_tick(now_fn=now_fn)
            _warn_peers_needing_refresh(now_fn())
        except Exception:
            logger.warning("Federation sync tick failed", exc_info=True)
        await sleep_fn(interval)

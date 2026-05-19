"""aiohttp client for federation peer communication."""
from __future__ import annotations

import logging
from typing import Any

import aiohttp

from orchestra import config
from orchestra.engine.federation import FederationBus, FederationPeer, FederationSummary

logger = logging.getLogger(__name__)


def _headers() -> dict[str, str]:
    token = config.federation_config().get("auth_token", "")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def push_to_peer(peer: FederationPeer, summaries: list[FederationSummary]) -> dict[str, Any]:
    """Push a batch of summaries to a remote peer."""
    if not summaries:
        return {"accepted": 0, "total": 0}
    if peer.is_token_expired():
        logger.warning("Auth token for peer %s has expired; skipping push.", peer.peer_id)
        return {"error": "auth_token_expired"}
    payload = [s.to_dict() for s in summaries]
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{peer.endpoint_url}/federation/push",
                json=payload,
                headers=_headers(),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                return data if resp.status == 200 else {"error": data.get("error", f"HTTP {resp.status}")}
        except Exception as exc:
            logger.warning("Push to peer %s failed: %s", peer.peer_id, exc)
            return {"error": str(exc)}


async def query_peer(peer: FederationPeer, task: str, limit: int = 5) -> list[dict[str, Any]]:
    """Query a remote peer for similar summaries."""
    if peer.is_token_expired():
        logger.warning("Auth token for peer %s has expired; skipping query.", peer.peer_id)
        return [{"error": "auth_token_expired"}]
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                f"{peer.endpoint_url}/federation/query",
                params={"q": task, "limit": limit},
                headers=_headers(),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                if resp.status == 200:
                    return data.get("results", [])
                return [{"error": data.get("error", f"HTTP {resp.status}")}]
        except Exception as exc:
            logger.warning("Query to peer %s failed: %s", peer.peer_id, exc)
            return []


async def sync_with_peer(peer: FederationPeer) -> dict[str, Any]:
    """Bidirectional sync: push outbox, then query remote status."""
    bus = FederationBus()
    outbox = bus.dequeue_outbox()
    result = await push_to_peer(peer, outbox)
    accepted_hashes = []
    if "accepted" in result and result["accepted"] > 0:
        accepted_hashes = [s.content_hash for s in outbox[: result["accepted"]]]
        bus.clear_outbox(accepted_hashes)
    peer.last_sync_at = __import__("time").time()
    return {"pushed": result, "outbox_remaining": len(bus.dequeue_outbox())}


def _peer_from_config(url: str, idx: int) -> FederationPeer:
    """Build a FederationPeer from config, hydrating token metadata."""
    cfg = config.federation_config()
    token = cfg.get("auth_token", "")
    ttl_hours = float(cfg.get("auth_token_ttl_hours", 0.0))
    issued_at = float(cfg.get("auth_token_issued_at", 0.0))
    expires_at = 0.0
    if ttl_hours > 0.0 and issued_at > 0.0:
        expires_at = issued_at + (ttl_hours * 3600)
    return FederationPeer(
        peer_id=f"peer-{idx}",
        endpoint_url=url,
        auth_token=token,
        token_issued_at=issued_at,
        token_expires_at=expires_at,
    )


async def sync_all_peers() -> dict[str, Any]:
    """Sync outbox with all configured peers."""
    peer_urls = config.federation_config().get("peer_urls", [])
    results: dict[str, Any] = {}
    for idx, url in enumerate(peer_urls):
        peer = _peer_from_config(url, idx)
        results[url] = await sync_with_peer(peer)
    return results

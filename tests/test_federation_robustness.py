"""Tests for federation delivery retry + TTL cleanup (Faz 5 Block C)."""
from unittest.mock import AsyncMock, patch

import pytest

from orchestra.engine.federation import FederationBus, FederationSummary, FederationPeer
from orchestra.federation_server import federation_client


def _summary(content_hash, timestamp, ttl_hours=720):
    return FederationSummary(
        content_hash=content_hash, embedding_digest="d",
        timestamp=timestamp, ttl_hours=ttl_hours,
    )


@pytest.mark.asyncio
async def test_push_with_retry_succeeds_after_transient_errors():
    peer = FederationPeer(peer_id="p", endpoint_url="http://x")
    summaries = [_summary("a", 0.0)]
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    push = AsyncMock(side_effect=[{"error": "x"}, {"error": "x"}, {"accepted": 1, "total": 1}])
    with patch.object(federation_client, "push_to_peer", push):
        result = await federation_client.push_with_retry(
            peer, summaries, max_retries=3, sleep_fn=fake_sleep
        )
    assert result == {"accepted": 1, "total": 1}
    assert len(sleeps) == 2  # slept before the two retries


@pytest.mark.asyncio
async def test_push_with_retry_does_not_retry_permanent_errors():
    """auth_token_expired is permanent — return immediately, no backoff waste."""
    peer = FederationPeer(peer_id="p", endpoint_url="http://x")
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    push = AsyncMock(return_value={"error": "auth_token_expired"})
    with patch.object(federation_client, "push_to_peer", push):
        result = await federation_client.push_with_retry(
            peer, [_summary("a", 0.0)], max_retries=5, sleep_fn=fake_sleep
        )
    assert result == {"error": "auth_token_expired"}
    assert sleeps == []               # no retries/backoff
    assert push.await_count == 1


@pytest.mark.asyncio
async def test_push_with_retry_gives_up_after_max():
    peer = FederationPeer(peer_id="p", endpoint_url="http://x")
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    push = AsyncMock(return_value={"error": "down"})
    with patch.object(federation_client, "push_to_peer", push):
        result = await federation_client.push_with_retry(
            peer, [_summary("a", 0.0)], max_retries=2, sleep_fn=fake_sleep
        )
    assert "error" in result
    assert len(sleeps) == 2
    assert push.await_count == 3  # initial + 2 retries


def test_cleanup_expired_removes_only_expired(tmp_path):
    bus = FederationBus(root=tmp_path / "fed")
    now = 1_000_000.0
    bus.enqueue_outbox(_summary("old", timestamp=now - 10 * 3600, ttl_hours=1))  # expired
    bus.enqueue_outbox(_summary("fresh", timestamp=now, ttl_hours=1))            # fresh
    removed = bus.cleanup_expired(now)
    assert removed == 1
    remaining = {s.content_hash for s in bus.dequeue_outbox()}
    assert remaining == {"fresh"}

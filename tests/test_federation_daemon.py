"""Tests for the federation sync daemon + token refresh (Faz 5 completion)."""
from unittest.mock import AsyncMock, patch

import pytest

from orchestra import config
from orchestra.engine import federation_daemon
from orchestra.engine.federation import FederationPeer


@pytest.mark.asyncio
async def test_sync_tick_runs_sync_and_cleanup():
    with patch("orchestra.federation_server.federation_client.sync_all_peers",
               AsyncMock(return_value={"p": 1})), \
         patch("orchestra.engine.federation.FederationBus.cleanup_expired", return_value=2):
        result = await federation_daemon.sync_tick(now_fn=lambda: 0.0)
    assert result == {"sync": {"p": 1}, "cleaned": 2}


@pytest.mark.asyncio
async def test_run_sync_loop_ticks_and_sleeps():
    ticks = {"n": 0}

    def should_continue():
        ticks["n"] += 1
        return ticks["n"] <= 3

    sleeps = []

    async def rec_sleep(delay):
        sleeps.append(delay)

    tick = AsyncMock(return_value={})
    with patch.object(federation_daemon, "sync_tick", tick):
        await federation_daemon.run_sync_loop(interval=5, sleep_fn=rec_sleep, should_continue=should_continue)
    assert tick.await_count == 3
    assert sleeps == [5, 5, 5]


@pytest.mark.asyncio
async def test_run_sync_loop_swallows_tick_error():
    seq = {"n": 0}

    def should_continue():
        seq["n"] += 1
        return seq["n"] <= 2

    async def rec_sleep(delay):
        pass

    tick = AsyncMock(side_effect=[RuntimeError("boom"), {}])
    with patch.object(federation_daemon, "sync_tick", tick):
        await federation_daemon.run_sync_loop(interval=1, sleep_fn=rec_sleep, should_continue=should_continue)
    assert tick.await_count == 2  # loop survived the first tick's error


def test_federation_sync_enabled(monkeypatch):
    monkeypatch.setattr(config, "federation_config", lambda: {"enabled": True, "sync_interval_seconds": 10})
    assert config.federation_sync_enabled() is True
    monkeypatch.setattr(config, "federation_config", lambda: {"enabled": False, "sync_interval_seconds": 10})
    assert config.federation_sync_enabled() is False
    monkeypatch.setattr(config, "federation_config", lambda: {"enabled": True, "sync_interval_seconds": 0})
    assert config.federation_sync_enabled() is False


def test_peer_needs_refresh():
    no_ttl = FederationPeer(peer_id="p", endpoint_url="u", token_expires_at=0.0)
    assert no_ttl.needs_refresh(999.0, 10.0) is False
    ttl = FederationPeer(peer_id="p", endpoint_url="u", token_expires_at=100.0)
    assert ttl.needs_refresh(95.0, 10.0) is True    # 95 > 100 - 10
    assert ttl.needs_refresh(80.0, 10.0) is False   # 80 < 90

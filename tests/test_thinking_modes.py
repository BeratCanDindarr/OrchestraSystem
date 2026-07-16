from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock
from aiohttp.test_utils import TestClient, TestServer

from orchestra import thinking_modes as tm
from orchestra import gateway


def test_split_mode():
    assert tm.split_mode("oll-fast#deep") == ("oll-fast", "deep")
    assert tm.split_mode("oll-fast") == ("oll-fast", None)
    assert tm.split_mode("cdx-deep@windows#plan") == ("cdx-deep@windows", "plan")
    assert tm.split_mode("oll-fast#bogus") == ("oll-fast", None)  # unknown mode ignored


def test_tier_for_alias():
    assert tm.tier_for_alias("oll-fast") == "compact"
    assert tm.tier_for_alias("oll-kimi") == "full"
    assert tm.tier_for_alias("cdx-deep") == "full"
    assert tm.tier_for_alias("oll-fast@backend") == "compact"


def test_mode_prompt_loads_real_file():
    p = tm.mode_prompt("deep", "oll-fast")           # compact tier
    assert p and "step by step" in p.lower()
    pf = tm.mode_prompt("deep", "oll-kimi")           # full tier
    assert pf and "rigor over speed" in pf.lower()
    assert tm.mode_prompt("bogus", "oll-fast") is None


def test_concise_and_calibrated_modes_registered():
    assert tm.split_mode("oll-fast#concise") == ("oll-fast", "concise")
    assert tm.split_mode("cdx-deep#calibrated") == ("cdx-deep", "calibrated")


def test_concise_prompt_loads_real_file():
    p = tm.mode_prompt("concise", "oll-fast")          # compact tier
    assert p and "concise" in p.lower()
    pf = tm.mode_prompt("concise", "oll-kimi")         # full tier
    assert pf and "concise" in pf.lower()


def test_calibrated_prompt_loads_real_file():
    p = tm.mode_prompt("calibrated", "oll-fast")       # compact tier
    assert p and "i don't know" in p.lower()
    pf = tm.mode_prompt("calibrated", "cdx-deep")      # full tier
    assert pf and "i don't know" in pf.lower() and "sycophancy" in pf.lower()


def test_prepend_system():
    assert tm.prepend_system(None, "MODE") == "MODE"
    assert tm.prepend_system("sys", "MODE") == "MODE\n\nsys"
    assert tm.prepend_system([{"type": "text", "text": "s"}], "MODE")[0]["text"] == "MODE"


@pytest.mark.asyncio
async def test_gateway_openai_injects_mode_prompt():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.run_target", new=AsyncMock(return_value="ok")) as mock:
            resp = await client.post("/v1/chat/completions", json={
                "model": "oll-fast#deep",
                "messages": [{"role": "user", "content": "solve X"}],
            })
            assert resp.status == 200
        # the deep-mode system prompt must be folded into the prompt sent to the model
        prompt_arg = mock.call_args.args[1]
        assert "step by step" in prompt_arg.lower()
        assert mock.call_args.args[0] == "oll-fast"           # mode stripped from model id


@pytest.mark.asyncio
async def test_gateway_anthropic_injects_mode_into_system():
    captured = {}
    async def _fake_run_target(model, prompt):
        captured["prompt"] = prompt
        return "ok"
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.run_target", _fake_run_target):
            resp = await client.post("/v1/messages", json={
                "model": "oll-fast#critic", "max_tokens": 32,
                "system": "you are helpful",
                "messages": [{"role": "user", "content": "review this"}],
            })
            assert resp.status == 200
    # mode prompt prepended before the user's own system text
    assert "do not praise" in captured["prompt"].lower()
    assert "you are helpful" in captured["prompt"]


@pytest.mark.asyncio
async def test_gateway_injects_calibrated_mode():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.run_target", new=AsyncMock(return_value="ok")) as mock:
            resp = await client.post("/v1/chat/completions", json={
                "model": "oll-fast#calibrated",
                "messages": [{"role": "user", "content": "is X true?"}],
            })
            assert resp.status == 200
        prompt_arg = mock.call_args.args[1]
        assert "i don't know" in prompt_arg.lower()   # calibrated-mode prompt folded in
        assert mock.call_args.args[0] == "oll-fast"    # mode stripped from model id


@pytest.mark.asyncio
async def test_no_mode_is_unchanged():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.run_target", new=AsyncMock(return_value="ok")) as mock:
            resp = await client.post("/v1/chat/completions", json={
                "model": "oll-fast",
                "messages": [{"role": "user", "content": "hi"}],
            })
            assert resp.status == 200
        prompt_arg = mock.call_args.args[1]
        assert "step by step" not in prompt_arg.lower()  # no mode injected

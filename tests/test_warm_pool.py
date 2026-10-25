"""A Claude agent call can take a CLI process started ahead of time: its start-up (the CLI and its MCP
servers, ~12 s with three servers) is paid while nobody waits. One process answers one task."""
from __future__ import annotations

import asyncio
import json
import sys
import time

import pytest

FAKE_CLI = r'''
import json, os, sys, time
argv = sys.argv[1:]
if "--input-format" in argv:          # warm: the prompt comes on stdin
    prompt = json.loads(sys.stdin.readline())["message"]["content"]
else:                                 # cold: the prompt is the last argument
    time.sleep(1.0)                   # start-up a warm process has already paid
    prompt = argv[-1]
print(json.dumps({"type": "result", "result": f"pid {os.getpid()}: {prompt}", "is_error": False,
                  "usage": {"input_tokens": 1, "output_tokens": 1}}), flush=True)
'''


@pytest.fixture
def pool(monkeypatch, tmp_path):
    from orchestra.providers import claude, warm_pool

    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLI)
    provider = claude.ClaudeProvider()
    monkeypatch.setattr(provider, "build_command", lambda prompt, effort: [sys.executable, str(script), prompt])
    warm_pool.enable()
    yield provider, warm_pool
    warm_pool.disable()


def _ask(provider, prompt):
    return asyncio.run(provider.async_run(prompt, "sonnet", timeout=30))


def test_the_second_call_takes_the_process_the_first_started(pool):
    provider, warm_pool = pool

    first, code = _ask(provider, "one")
    assert code == 0 and first.endswith(": one")
    warm_pool.wait_ready(timeout=10)   # the refill the first call started

    started = time.monotonic()
    second, code = _ask(provider, "two")
    took = time.monotonic() - started

    assert code == 0 and second.endswith(": two")
    assert second.split(":")[0] != first.split(":")[0]   # another process: one task each
    assert took < 0.9                                    # no start-up wait (a cold call sleeps 1 s)


def test_without_the_pool_every_call_starts_its_own_process(pool):
    provider, warm_pool = pool
    warm_pool.disable()

    _ask(provider, "one")

    assert warm_pool.waiting() == 0


def test_a_process_waiting_too_long_is_not_used(pool, monkeypatch):
    provider, warm_pool = pool
    _ask(provider, "one")
    warm_pool.wait_ready(timeout=10)
    monkeypatch.setattr(warm_pool, "MAX_IDLE_SECONDS", 0)

    answer, code = _ask(provider, "two")

    assert code == 0 and answer.endswith(": two")   # answered cold; the stale process was ended

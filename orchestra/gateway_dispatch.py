"""Provider-agnostic dispatch shared by the gateway's OpenAI and Anthropic compat layers:
resolve a model id to a provider/mode and run it — whole (run_target) or streaming (stream_target)."""
from __future__ import annotations

import asyncio
import logging
import threading

from orchestra.providers.fallback import resolve, known_aliases
from orchestra.engine.runners.common import run_dual_async, run_critical_async

logger = logging.getLogger(__name__)
_STREAM_ERROR_TEXT = "[stream error]"   # generic; never leak provider exception text

_MODES = {"dual", "critical", "auto"}


def resolve_target(model: str) -> tuple[str, str]:
    """('mode', <mode>) for orchestra-<mode>, else ('alias', <model>).

    Raises ValueError for an unknown orchestration mode OR an unknown alias
    (base name not in the alias table) — the OpenAI gateway maps this to a 404.
    """
    if model.startswith("orchestra-"):
        mode = model[len("orchestra-"):]
        if mode not in _MODES:
            raise ValueError(f"unknown orchestration mode {mode!r}")
        return "mode", mode
    base = model.split("@", 1)[0]
    if base not in known_aliases():
        raise ValueError(f"unknown model {model!r}")
    return "alias", model


def _reply_from_run(run) -> str:
    """Extract the text reply from an OrchestraRun (or dict) result."""
    summary = getattr(run, "summary", None)
    if summary:
        return summary
    agents = getattr(run, "agents", None) or []
    return agents[-1].stdout_log if agents else ""


def _reply_from_manifest(manifest: dict) -> str:
    """Extract reply text from a run_task manifest dict."""
    agents = manifest.get("agents") or []
    return agents[-1].get("stdout_log", "") if agents else ""


async def run_target(model: str, prompt: str) -> str:
    """Full (non-streaming) reply for a model id."""
    kind, value = resolve_target(model)
    if kind == "mode":
        if value == "dual":
            return _reply_from_run(await run_dual_async(prompt))
        if value == "critical":
            return _reply_from_run(await run_critical_async(prompt))
        from orchestra.service import run_task
        result = await asyncio.to_thread(run_task, mode="auto", task=prompt)
        run_dict = result.get("run", {}) if isinstance(result, dict) else {}
        return run_dict.get("summary") or _reply_from_manifest(run_dict)
    provider, effort = resolve(model)
    stdout, _ = await asyncio.to_thread(provider.run, prompt, effort)
    return stdout


def _start_bg(fn) -> None:
    """Launch fn in a daemon thread."""
    threading.Thread(target=fn, daemon=True).start()


async def stream_target(model: str, prompt: str):
    """Yield reply chunks asynchronously (see openai_compat history for the design).

    Single alias with streaming provider → live tokens via a thread-bridged Queue.
    NotImplementedError (base/CLI/remote) → falls back to run(). Modes → whole reply once.
    The consumer's finally sets a stop Event so a disconnected client abandons the
    background provider stream instead of leaking the thread + a live model call.
    """
    kind, _ = resolve_target(model)
    if kind != "alias":
        yield await run_target(model, prompt)
        return

    provider, effort = resolve(model)
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stop = threading.Event()
    sentinel = object()

    def _produce() -> None:
        try:
            for chunk in provider.stream(prompt, effort):
                if stop.is_set():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except NotImplementedError:
            if not stop.is_set():
                stdout, _ = provider.run(prompt, effort)
                loop.call_soon_threadsafe(queue.put_nowait, stdout)
        except Exception as exc:
            logger.warning("gateway stream provider failed: %s", exc)
            if not stop.is_set():
                loop.call_soon_threadsafe(queue.put_nowait, _STREAM_ERROR_TEXT)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    _start_bg(_produce)
    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                return
            yield item
    finally:
        stop.set()

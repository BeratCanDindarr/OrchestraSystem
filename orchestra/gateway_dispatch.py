"""Provider-agnostic dispatch shared by the gateway's OpenAI and Anthropic compat layers:
resolve a model id to a provider/mode and run it — whole (run_target) or streaming (stream_target)."""
from __future__ import annotations

import asyncio
import logging
import threading

from orchestra import config
from orchestra.engine import artifacts, costing
from orchestra.engine.runners.ask_runner import run_ask
from orchestra.engine.runners.common import run_ask_async, run_dual_async
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus
from orchestra.providers.fallback import resolve, known_aliases

logger = logging.getLogger(__name__)
_STREAM_ERROR_TEXT = "[stream error]"   # generic; never leak provider exception text

_MODES = {"dual", "consensus", "auto"}

# What the gateway serves straight from a provider (streamed text, native tool-call turns) is
# recorded as a step under this name: off `orchestra ps`, on the books for cost and budget.
GATEWAY_PARENT = "gateway"
_RECORDED_TASK_CHARS = 2000
# A reply that is a whole conversation turn: no role prefix, no stored answer, no approval gate.
_PIPELINE_OPTIONS = {"internal": True, "show_live": False, "emit_console": False}


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
        if value == "consensus":
            return _reply_from_run(await run_dual_async(prompt, stances=True))
        from orchestra.service import run_task
        result = await asyncio.to_thread(run_task, mode="auto", task=prompt)
        run_dict = result.get("run", {}) if isinstance(result, dict) else {}
        return run_dict.get("summary") or _reply_from_manifest(run_dict)
    return _reply_or_raise(await run_ask_async(model, prompt, **_PIPELINE_OPTIONS))


def _reply_or_raise(run) -> str:
    """The run's answer; a run that did not complete is an error, never an assistant message."""
    if run.status != RunStatus.COMPLETED:
        reason = getattr(run.failure, "message", None) or f"run {run.run_id} ended {run.status.value}"
        raise RuntimeError(reason)
    return _reply_from_run(run)


def _record(alias: str, provider, effort: str, prompt: str, output: str, reported: dict | None) -> None:
    """Put a reply the gateway served straight from a provider on the books."""
    try:
        run = OrchestraRun(mode="ask", task=prompt[-_RECORDED_TASK_CHARS:], parent_run_id=GATEWAY_PARENT)
        agent = AgentRun(alias=alias, provider=getattr(provider, "name", ""),
                         model=provider.model_label(effort), status=AgentStatus.COMPLETED)
        agent.stdout_log = output
        costing.charge(agent, agent.model, reported if isinstance(reported, dict) else None,
                       prompt=prompt, output=output)
        run.agents = [agent]
        run.total_cost_usd = agent.estimated_cost_usd
        run.status = RunStatus.COMPLETED
        artifacts.write_manifest(run)
        artifacts.append_event(run.run_id, {"event": "run_finished", "status": "completed", "via": "gateway"})
    except Exception:
        logger.warning("could not record a gateway reply for %s", alias, exc_info=True)


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
        sent: list[str] = []
        try:
            for chunk in provider.stream(prompt, effort):
                if stop.is_set():
                    break
                sent.append(chunk)
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
            usage = getattr(provider, "last_usage", lambda: None)()
            _record(model, provider, effort, prompt, "".join(sent), usage)
        except NotImplementedError:
            if not stop.is_set():
                # A provider that cannot stream answers through the run pipeline, in one piece.
                try:
                    reply = _reply_or_raise(run_ask(model, prompt, **_PIPELINE_OPTIONS))
                except Exception as exc:
                    logger.warning("gateway run failed: %s", exc)
                    reply = _STREAM_ERROR_TEXT
                loop.call_soon_threadsafe(queue.put_nowait, reply)
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


def native_capable(model: str) -> bool:
    """True when `model` is a single alias whose provider can run Anthropic-shaped
    payloads natively (structured tool calls). Modes and unknown models -> False."""
    try:
        kind, _ = resolve_target(model)
    except ValueError:
        return False
    if kind != "alias":
        return False
    provider, _ = resolve(model)
    return bool(getattr(provider, "supports_native_messages", False))


async def run_native(model: str, payload: dict) -> dict:
    """Run an Anthropic-shaped payload through a native-messages-capable provider.

    NO fallback chain here: a mid-session swap to a text-only provider would
    silently break the client's tool loop. Non-capable providers raise."""
    provider, effort = resolve(model)
    if not getattr(provider, "supports_native_messages", False):
        raise NotImplementedError(
            f"provider {getattr(provider, 'name', '?')} does not support native messages"
        )
    resolved = {**payload, "model": effort}

    def _call() -> dict:
        # Retry transient failures so a network blip / 5xx doesn't crash the client's tool loop.
        if config.retry_native_enabled():
            from orchestra.providers import native_resilience
            result = native_resilience.call_with_retry(
                lambda: provider.native_messages(resolved),
                max_attempts=config.retry_max_attempts(),
                backoff_ms=config.retry_backoff_ms(),
                multiplier=config.retry_backoff_multiplier(),
            )
        else:
            result = provider.native_messages(resolved)
        _record_native(model, provider, effort, payload, result)
        return result

    return await asyncio.to_thread(_call)


def _record_native(alias: str, provider, effort: str, payload: dict, result: dict) -> None:
    """Record one turn of a native tool loop with the tokens the provider reported for it."""
    usage = result.get("usage") if isinstance(result, dict) else None
    reported = None
    if isinstance(usage, dict) and usage:
        reported = {"prompt_tokens": usage.get("input_tokens"), "completion_tokens": usage.get("output_tokens"),
                    "cached_tokens": usage.get("cache_read_input_tokens")}
    blocks = (result.get("content") or []) if isinstance(result, dict) else []
    output = "".join(block.get("text", "") for block in blocks if isinstance(block, dict))
    messages = payload.get("messages") or []
    _record(alias, provider, effort, str(messages[-1].get("content", "")) if messages else "", output, reported)

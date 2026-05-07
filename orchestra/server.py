"""Orchestra Daemon V4.3: Slots + Toolbox + Direct Tool Execution."""
from __future__ import annotations

import asyncio
import json
import logging
import socket
from pathlib import Path
from typing import Any, Dict, Set

import websockets
from orchestra.engine.runner import run_ask, run_dual, run_planned

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestra.server")

CONNECTED_CLIENTS: Set[websockets.WebSocketServerProtocol] = set()
SERVER_LOOP: asyncio.AbstractEventLoop | None = None

# ── Slot tracking ─────────────────────────────────────────────────────────────
_ALIAS_SLOTS: Dict[str, asyncio.Semaphore] = {}
_SLOT_TASKS:  Dict[str, str] = {}

KNOWN_ALIASES = [
    "cld-fast", "cld-deep",
    "cdx-deep", "cdx-fast",
    "gmn-pro",  "gmn-fast",
    "oll-coder", "oll-analyst", "oll-fast", "oll-mini",
    "tool",
]

_ALIAS_MODEL_LABELS: Dict[str, str] = {
    "cld-fast":    "Claude Sonnet",
    "cld-deep":    "Claude Opus",
    "cdx-deep":    "Codex xhigh",
    "cdx-fast":    "Codex low",
    "gmn-pro":     "Gemini Pro",
    "gmn-fast":    "Gemini Flash",
    "oll-coder":   "qwen2.5-coder:7b",
    "oll-analyst": "deepseek-r1:8b",
    "oll-fast":    "qwen2.5:7b",
    "oll-mini":    "phi4-mini",
    "tool":        "Tool Runner",
}

def _get_slot(alias: str | None) -> asyncio.Semaphore | None:
    if not alias: return None
    if alias not in _ALIAS_SLOTS:
        _ALIAS_SLOTS[alias] = asyncio.Semaphore(1)
    return _ALIAS_SLOTS[alias]

def _slot_states() -> list[dict]:
    return [
        {
            "alias":  a,
            "status": "busy" if a in _SLOT_TASKS else "idle",
            "task":   _SLOT_TASKS.get(a, ""),
            "model":  _ALIAS_MODEL_LABELS.get(a, a),
        }
        for a in KNOWN_ALIASES
    ]

# ── Toolbox discovery ─────────────────────────────────────────────────────────
def _load_tools() -> list[dict]:
    """Discover runnable tools from orchestra/tools/*.py"""
    tools = []
    tools_dir = Path(__file__).parent / "tools"
    if tools_dir.exists():
        for f in sorted(tools_dir.glob("*.py")):
            if f.name == "__init__.py":
                continue
            tools.append({
                "name":  f.stem,
                "task":  str(f),
                "mode":  "tool",
                "alias": "tool",
            })
    return tools

# ── Broadcast helpers ─────────────────────────────────────────────────────────
async def broadcast(payload: Dict[str, Any]):
    if not CONNECTED_CLIENTS:
        return
    msg = json.dumps(payload, ensure_ascii=False)
    for client in list(CONNECTED_CLIENTS):
        try:
            await client.send(msg)
        except Exception:
            CONNECTED_CLIENTS.discard(client)

def _schedule_broadcast(payload: Dict[str, Any]) -> None:
    if SERVER_LOOP is None or not SERVER_LOOP.is_running():
        return
    asyncio.run_coroutine_threadsafe(broadcast(payload), SERVER_LOOP)


def _event_hook(run_id: str, event: Dict[str, Any]) -> None:
    event_name = event.get("event", "")
    if event_name == "agent_retry":
        alias = event.get("alias", "agent")
        model = event.get("model", "")
        attempt = event.get("attempt", 0)
        detail = f"{alias} retry {attempt}"
        if model:
            detail += f" -> {model}"
        _schedule_broadcast({"type": "log", "message": f"[System] {detail}", "alias": None})
        return

    if event_name == "approval_requested":
        _schedule_broadcast({
            "type": "approval_request",
            "run_id": run_id,
            "approval": {
                "actionKind": "Continue critical run",
                "target": run_id,
                "summary": "Round 1 completed and is waiting for user approval before continuing.",
                "diffStat": "",
                "preview": "",
                "risk": "Second-stage synthesis / continuation",
            },
        })
        return

    if event_name in {"run_started", "run_finished", "run_resumed"}:
        _schedule_broadcast({"type": "history", "runs": _recent_runs(10)})

# ── Streaming token buffer ────────────────────────────────────────────────────
import threading as _threading

class _StreamBuffer:
    """Per-alias token accumulator.
    Flushes when a sentence boundary is hit, buffer exceeds MAX_BUF chars,
    or flush() is called explicitly at stream end.

    card_fn registration: call set_card_fn(alias, fn) before streaming starts so
    push() can extract [[kind:ref|label]] annotations and emit result_card payloads.
    push() will hold the buffer when an open [[ has no matching ]] yet.
    """
    MAX_BUF = 120
    _BOUNDARIES = {'.', '!', '?', '\n'}

    def __init__(self):
        self._bufs:     dict[str, list[str]]  = {}
        self._card_fns: dict[str, object]     = {}
        self._lock = _threading.Lock()

    def set_card_fn(self, alias: str, card_fn) -> None:
        with self._lock:
            self._card_fns[alias] = card_fn

    def clear_card_fn(self, alias: str) -> None:
        with self._lock:
            self._card_fns.pop(alias, None)

    @staticmethod
    def _has_open_annotation(text: str) -> bool:
        """True if text contains [[ without a matching ]] after it."""
        idx = text.rfind('[[')
        return idx != -1 and ']]' not in text[idx:]

    def _do_flush(self, alias: str, combined: str, send_fn) -> None:
        """Inner flush: extract annotations if card_fn registered, then broadcast."""
        card_fn = self._card_fns.get(alias)
        if card_fn is not None:
            clean, cards = _extract_cards(combined, alias)
            if clean.strip():
                send_fn(clean)
            for card in cards:
                card_fn(card)
        else:
            send_fn(combined)

    def push(self, chunk: str, alias: str, send_fn) -> None:
        with self._lock:
            buf = self._bufs.setdefault(alias, [])
            buf.append(chunk)
            combined = "".join(buf)
            # Hold buffer if there is an open annotation that has not closed yet
            if self._has_open_annotation(combined):
                return
            if len(combined) >= self.MAX_BUF or any(c in combined for c in self._BOUNDARIES):
                self._bufs[alias] = []
                self._do_flush(alias, combined, send_fn)

    def flush(self, alias: str, send_fn, card_fn=None) -> None:
        with self._lock:
            # Allow caller to override card_fn at flush time (backwards compat)
            if card_fn is not None:
                self._card_fns[alias] = card_fn
            buf = self._bufs.pop(alias, [])
            if buf:
                self._do_flush(alias, "".join(buf), send_fn)

_STREAM_BUFFER = _StreamBuffer()

# ── Annotation parser ─────────────────────────────────────────────────────────
import re as _re

_ANNOTATION_RE = _re.compile(r'\[\[(\w+):([^\]|]+)(?:\|([^\]]*))?\]\]')

def _extract_cards(text: str, alias: str) -> tuple:
    """
    Metinden [[kind:ref|label]] annotation'larini cikarir.
    Returns: (clean_text, list_of_result_card_payloads)
    Annotation flush sirasinda calistirilmali; streaming chunk'larinda degil.
    """
    cards = []
    def _replace(m):
        kind  = m.group(1)
        ref   = m.group(2).strip()
        label = (m.group(3) or m.group(2)).strip()
        line  = 0
        # Script refs may embed line number: Assets/Foo.cs:42 → split out
        if kind == "script" and ":" in ref:
            parts = ref.rsplit(":", 1)
            if parts[1].isdigit():
                ref  = parts[0]
                line = int(parts[1])
        cards.append({
            "type":  "result_card",
            "alias": alias,
            "kind":  kind,
            "ref":   ref,
            "label": label,
            "line":  line,
        })
        return f"[{label}]"
    clean = _ANNOTATION_RE.sub(_replace, text)
    return clean, cards


def _stream_hook(chunk: str, alias: str = "agent"):
    def _send(text: str):
        _schedule_broadcast({"type": "log", "message": text, "alias": alias})
    _STREAM_BUFFER.push(chunk, alias, _send)

def _recent_runs(limit: int = 15) -> list[dict]:
    try:
        from orchestra.engine import artifacts
        return sorted(artifacts.list_runs(), key=lambda x: x.get("created_at", ""), reverse=True)[:limit]
    except Exception:
        return []

# ── Request handler ───────────────────────────────────────────────────────────
async def handle_request(websocket: websockets.WebSocketServerProtocol, request: Dict[str, Any]):
    req_type = request.get("type")
    payload  = request.get("payload", {})

    if req_type == "execute":
        mode  = payload.get("mode", "ask")
        task  = payload.get("task", "")
        alias = payload.get("alias", "cdx-deep")

        # Direct Python tool execution
        if mode == "tool":
            script_path = task
            async def _run_tool():
                tool_name = Path(script_path).stem
                await broadcast({"type": "slot_update", "alias": "tool", "status": "busy",
                                 "task": tool_name, "model": "Tool Runner"})
                await broadcast({"type": "log", "message": f"▶ Running {tool_name}...", "alias": "tool"})
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "python3", script_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                    )
                    async for line in proc.stdout:
                        text = line.decode().rstrip()
                        if text:
                            await broadcast({"type": "log", "message": text, "alias": "tool"})
                    await proc.wait()
                    await broadcast({"type": "log", "message": f"✓ {tool_name} done", "alias": "tool"})
                except Exception as e:
                    await broadcast({"type": "error", "message": f"Tool error: {e}"})
                finally:
                    await broadcast({"type": "slot_update", "alias": "tool", "status": "idle",
                                     "task": "", "model": "Tool Runner"})
            asyncio.create_task(_run_tool())
            await websocket.send(json.dumps({"type": "ack", "message": "Tool running..."}))
            return

        # AI model execution with slot tracking
        async def _run():
            slot = _get_slot(alias if mode == "ask" else None)
            if slot:
                await slot.acquire()
                _SLOT_TASKS[alias] = task[:60]
                _schedule_broadcast({
                    "type": "slot_update", "alias": alias, "status": "busy",
                    "task": task[:60], "model": _ALIAS_MODEL_LABELS.get(alias, alias),
                })
            try:
                loop = asyncio.get_event_loop()
                def _card_send_early(payload: dict): _schedule_broadcast(payload)
                def _flush_send_early(text: str):
                    if text.strip():
                        _schedule_broadcast({"type": "log", "message": text, "alias": alias})
                _STREAM_BUFFER.set_card_fn(alias, _card_send_early)
                def _aliased_stream(chunk): _stream_hook(chunk, alias)
                run = await loop.run_in_executor(None, lambda: (
                    run_planned(task, emit_console=False, install_signal_handlers=False, stream_callback=_aliased_stream) if mode == "planned" else
                    run_dual(task,    emit_console=False, install_signal_handlers=False, stream_callback=_aliased_stream) if mode in ("dual", "critical") else
                    run_ask(alias, task, emit_console=False, install_signal_handlers=False, stream_callback=_aliased_stream)
                ))
                if run and getattr(run, "summary", None):
                    await broadcast({"type": "log", "message": f"🧠 Synthesis:\n{run.summary}", "alias": "synthesizer"})
            except Exception as e:
                await broadcast({"type": "error", "message": str(e)})
            finally:
                # Flush remaining buffer (annotation card_fn already registered via set_card_fn).
                _flush_alias = alias
                def _flush_send(text: str):
                    if text.strip():
                        _schedule_broadcast({"type": "log", "message": text, "alias": _flush_alias})
                _STREAM_BUFFER.flush(alias, _flush_send)
                _STREAM_BUFFER.clear_card_fn(alias)
                if slot:
                    _SLOT_TASKS.pop(alias, None)
                    slot.release()
                    _schedule_broadcast({
                        "type": "slot_update", "alias": alias, "status": "idle",
                        "task": "", "model": _ALIAS_MODEL_LABELS.get(alias, alias),
                    })
                # C2: broadcast updated run list after each run completes
                _schedule_broadcast({"type": "history", "runs": _recent_runs(10)})

        asyncio.create_task(_run())
        await websocket.send(json.dumps({"type": "ack", "message": f"Queued {mode}"}))

    elif req_type == "ping":
        await websocket.send(json.dumps({"type": "pong"}))

# ── Connection handler ────────────────────────────────────────────────────────
async def server_loop(websocket: websockets.WebSocketServerProtocol):
    CONNECTED_CLIENTS.add(websocket)
    try:
        await websocket.send(json.dumps({"type": "history", "runs": _recent_runs()},  ensure_ascii=False))
        await websocket.send(json.dumps({"type": "slots",   "slots": _slot_states()}, ensure_ascii=False))
        await websocket.send(json.dumps({"type": "toolbox", "tools": _load_tools()},  ensure_ascii=False))
        async for message in websocket:
            try:
                await handle_request(websocket, json.loads(message))
            except Exception:
                pass
    finally:
        CONNECTED_CLIENTS.discard(websocket)

# ── Ollama warmup ─────────────────────────────────────────────────────────────
async def _warmup_ollama():
    def _sync():
        try:
            from orchestra.providers.ollama import OllamaProvider
            return OllamaProvider.warmup("qwen2.5-coder:7b", keep_alive="60m")
        except Exception:
            return False

    _schedule_broadcast({
        "type": "slot_update", "alias": "oll-coder", "status": "warming",
        "task": "Loading model...", "model": "qwen2.5-coder:7b",
    })
    ok = await asyncio.get_event_loop().run_in_executor(None, _sync)
    status = "idle" if ok else "error"
    msg    = "✓ Ollama ready (qwen2.5-coder:7b)" if ok else "⚠ Ollama warmup skipped"
    logger.info(msg)
    _schedule_broadcast({"type": "slot_update", "alias": "oll-coder", "status": status, "task": "", "model": "qwen2.5-coder:7b"})
    _schedule_broadcast({"type": "thought", "message": msg})

# ── Entry point ───────────────────────────────────────────────────────────────
async def start_server(host: str = "127.0.0.1", port: int = 8765):
    global SERVER_LOOP
    SERVER_LOOP = asyncio.get_running_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        logger.error(f"Port {port} already in use. Is a daemon already running?")
        return
    logger.info(f"Orchestra Daemon V4.3 ready at {host}:{port}")
    async with websockets.serve(server_loop, sock=sock):
        asyncio.create_task(_warmup_ollama())
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(start_server())

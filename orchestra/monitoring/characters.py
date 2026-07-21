from __future__ import annotations
"""The model fleet as a roster of characters.

Turns Orchestra's alias table into a cast of named personas — each model a
character with a human name, a role archetype, and a deterministic generated
avatar — so the dashboard can present the fleet as a team you dispatch work to.
Personas are auto-derived from the alias/provider/tier and overridable via
``[characters.<alias>]`` config.
"""
import hashlib
from dataclasses import dataclass
from typing import Optional

from orchestra import config

# A pool of human first names, assigned deterministically so each model keeps a
# stable identity across restarts.
_NAME_POOL = [
    "Ada", "Bo", "Cy", "Dex", "Echo", "Fin", "Gale", "Hana", "Iris", "Jax",
    "Kit", "Lena", "Mika", "Nova", "Orin", "Pax", "Quinn", "Rex", "Sage", "Tao",
    "Uma", "Vera", "Wren", "Xan", "Yuki", "Zed", "Ari", "Bex", "Cleo", "Dane",
]

# provider -> role archetype
_ARCHETYPE = {
    "codex": "Coder",
    "openai_sdk": "Coder",
    "openai": "Coder",
    "azure_openai": "Coder",
    "gemini": "Researcher",
    "claude": "Advisor",
    "claude_sdk": "Advisor",
    "ollama": "Local Specialist",
    "deepseek": "Reasoner",
    "perplexity": "Scout",
    "remote": "Remote Agent",
}

# effort/model tokens that hint at a "deep" vs "fast" tier
_DEEP_HINTS = ("deep", "xhigh", "high", "pro", "opus", "reasoner", "reasoning", "o3", "llama3", "kimi", "minimax")
_FAST_HINTS = ("fast", "low", "flash", "mini", "lite", "sonar", "coder")


@dataclass
class Character:
    id: str          # the alias
    name: str        # human display name
    role: str        # role archetype + tier, e.g. "Deep Coder"
    provider: str    # provider name, e.g. "codex"
    model: str       # effort/model token
    available: bool  # provider.is_available()
    avatar: str      # inline SVG


def _tier(effort: str) -> str:
    e = str(effort).lower()
    if any(h in e for h in _DEEP_HINTS):
        return "Deep"
    if any(h in e for h in _FAST_HINTS):
        return "Fast"
    return ""


def _persona_for(alias: str, provider: str, effort: str, *, name_index: Optional[int] = None) -> dict:
    """Auto-derive a persona, overridable via ``[characters.<alias>]`` config.

    ``name_index`` (the alias's sorted position, supplied by build_roster) assigns a
    unique name from the pool; without it we fall back to a hash (may collide).
    """
    overrides = config.section("characters").get(alias, {})
    if not isinstance(overrides, dict):
        overrides = {}
    archetype = _ARCHETYPE.get(provider, "Agent")
    tier = _tier(effort)
    role = f"{tier} {archetype}".strip()
    idx = name_index if name_index is not None else _stable_index(alias)
    name = _NAME_POOL[idx % len(_NAME_POOL)]
    return {"name": overrides.get("name", name), "role": overrides.get("role", role)}


def _stable_index(alias: str) -> int:
    """Deterministic small integer from an alias (stable across processes)."""
    return int(hashlib.sha256(alias.encode("utf-8")).hexdigest()[:6], 16)


def avatar_svg(seed: str, *, size: int = 44) -> str:
    """Deterministic geometric avatar for a character seed (no dependencies)."""
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    hue = int(h[:2], 16) * 360 // 255
    hue2 = (hue + 140) % 360
    eye_y = size * 0.4
    mouth_w = size * (0.25 + (int(h[2:4], 16) / 255) * 0.2)
    cx = size / 2
    return (
        f'<svg class="avatar" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{cx}" cy="{cx}" r="{size/2}" fill="hsl({hue},55%,45%)"/>'
        f'<circle cx="{size*0.35:.1f}" cy="{eye_y:.1f}" r="{size*0.07:.1f}" fill="hsl({hue2},70%,88%)"/>'
        f'<circle cx="{size*0.65:.1f}" cy="{eye_y:.1f}" r="{size*0.07:.1f}" fill="hsl({hue2},70%,88%)"/>'
        f'<rect x="{cx-mouth_w/2:.1f}" y="{size*0.62:.1f}" width="{mouth_w:.1f}" height="{size*0.06:.1f}" '
        f'rx="{size*0.03:.1f}" fill="hsl({hue2},70%,88%)"/>'
        f'</svg>'
    )


def remote_characters(*, fetch_card_fn=None) -> list[Character]:
    """Characters for configured remote nodes' agents (``alias@node``).

    Best-effort: fetches each node's A2A Agent Card for its advertised skills
    (aliases); unreachable nodes are skipped. ``fetch_card_fn`` is injectable for
    testing (default: ``agent_card.fetch_agent_card``).
    """
    from orchestra import remote_nodes
    if fetch_card_fn is None:
        from orchestra.engine.agent_card import fetch_agent_card
        fetch_card_fn = fetch_agent_card
    out: list[Character] = []
    for node, cfg in remote_nodes.list_nodes().items():
        try:
            card = fetch_card_fn(cfg.get("url", ""), cfg.get("token", ""))
        except Exception:
            continue  # node unreachable — skip
        for skill in (card or {}).get("skills", []):
            alias = skill.get("id", "")
            if not alias:
                continue
            cid = f"{alias}@{node}"
            persona = _persona_for(cid, "remote", alias)
            out.append(Character(
                id=cid, name=persona["name"], role=persona["role"],
                provider=f"remote:{node}", model=alias, available=True,
                avatar=avatar_svg(cid),
            ))
    return out


def build_roster(*, include_remote: Optional[bool] = None) -> list[Character]:
    """Build the character roster from the alias table (+ remote nodes when enabled)."""
    from orchestra.providers.fallback import ALIAS_TABLE
    # sorted position → a unique name from the pool (stable across restarts)
    positions = {alias: i for i, alias in enumerate(sorted(ALIAS_TABLE))}
    roster: list[Character] = []
    for alias, (provider, effort) in ALIAS_TABLE.items():
        pname = getattr(provider, "name", type(provider).__name__)
        try:
            available = bool(provider.is_available())
        except Exception:
            available = False
        persona = _persona_for(alias, pname, str(effort), name_index=positions[alias])
        roster.append(Character(
            id=alias, name=persona["name"], role=persona["role"],
            provider=pname, model=str(effort), available=available,
            avatar=avatar_svg(alias),
        ))
    if include_remote is None:
        include_remote = config.fleet_include_remote_nodes()
    if include_remote:
        try:
            roster.extend(remote_characters())
        except Exception:
            import logging
            logging.getLogger(__name__).warning("remote roster build failed", exc_info=True)
    return roster


def _dispatch_output(result) -> tuple:
    """Extract (output_text, ok) from a dispatch result (OrchestraRun or str)."""
    if isinstance(result, str):
        return result, True
    summary = getattr(result, "summary", None)
    if not summary:
        agents = getattr(result, "agents", None) or []
        summary = agents[0].stdout_log if agents else ""
    status = getattr(result, "status", None)
    ok = True if status is None else getattr(status, "value", str(status)).lower() == "completed"
    return str(summary or "(no output)"), ok


def _default_dispatch_stream(alias: str, task: str):
    """Sync generator streaming a model's output for a task (for live dispatch)."""
    from orchestra.providers.fallback import resolve
    provider, effort = resolve(alias)
    try:
        yield from provider.stream(task, effort)
    except NotImplementedError:
        # provider can't stream (base/CLI/remote) → run whole, yield once
        from orchestra.engine.runners.ask_runner import run_ask
        yield _dispatch_output(run_ask(alias, task, show_live=False))[0]


def dispatch_task(character_id: str, task: str, *, context: str = "", room_id=None,
                  dispatch_fn=None, stream_fn=None, inbox=None) -> dict:
    """Assign a task to a model (character), running it in the background.

    Streams the model's output live into the inbox (so the terminal viewer shows it
    token-by-token) and marks the dispatch ``completed``/``failed`` when done. Returns
    immediately with the dispatch id. If ``dispatch_fn`` is given it is used whole
    (no streaming). ``stream_fn``/``inbox`` are injectable for testing.

    ``context`` (a room's role prompt + rules) is prepended to the task sent to the
    model, while the inbox keeps the short ``task`` for display. ``room_id`` tags the
    dispatch so the board can group it under its room.
    """
    import logging
    import threading
    logger = logging.getLogger(__name__)
    if not character_id or not task:
        return {"ok": False, "error": "character_id and task are required"}
    if inbox is None:
        from orchestra.monitoring.dispatch_inbox import get_inbox
        inbox = get_inbox()

    dispatch_id = inbox.add(character_id, task, room_id=room_id)
    prompt = f"{context}\n\n---\nTASK:\n{task}" if context and context.strip() else task

    def _run():
        try:
            if dispatch_fn is not None:  # whole-result path (tests / non-streaming)
                output, ok = _dispatch_output(dispatch_fn(character_id, prompt))
                inbox.complete(dispatch_id, output, ok=ok)
                return
            gen = stream_fn if stream_fn is not None else _default_dispatch_stream
            chunks = []
            for chunk in gen(character_id, prompt):
                if chunk:
                    chunks.append(chunk)
                    inbox.append_output(dispatch_id, chunk)  # live terminal
            inbox.complete(dispatch_id, "".join(chunks) or "(no output)", ok=True)
        except Exception:
            logger.warning("Fleet dispatch to %s failed", character_id, exc_info=True)
            inbox.complete(dispatch_id, "dispatch failed — see server logs", ok=False)

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "character_id": character_id, "dispatch_id": dispatch_id}


def _default_suggest(task: str, candidates: list[str]):
    from orchestra.router.semantic import suggest_best_alias
    return suggest_best_alias(task, candidates)


def _default_tier(task: str) -> str:
    from orchestra.router.classifier import classify_to_tier
    return classify_to_tier(task)


def auto_cast(task: str, roster: list, *, suggest_fn=None, tier_fn=None) -> dict:
    """Pick the best eligible character for a task and explain why.

    ``roster`` is a list of (Character, presence-state) pairs. Prefers models that
    are available/idle (not busy/offline). Uses learned routing (past performance on
    similar tasks) first, then a tier heuristic. Returns
    ``{character_id, name, reason}`` or ``{error}`` when nothing is eligible.
    """
    eligible = [(c, s) for c, s in roster if s in ("available", "idle")]
    if not eligible:
        eligible = [(c, s) for c, s in roster if s != "offline"]  # allow busy as a last resort
    if not eligible:
        return {"error": "no models are available right now"}

    by_id = {c.id: c for c, _ in eligible}
    cand_ids = list(by_id)

    # 1) learned routing — the model with the best track record on similar tasks
    suggest_fn = suggest_fn or _default_suggest
    try:
        picked = suggest_fn(task, cand_ids)
    except Exception:
        picked = None
    if picked in by_id:
        c = by_id[picked]
        return {"character_id": c.id, "name": c.name,
                "reason": f"{c.name} has the best track record on similar tasks"}

    # 2) tier heuristic — complex tasks → a Deep character, simple → a Fast one
    tier_fn = tier_fn or _default_tier
    try:
        tier = str(tier_fn(task)).lower()
    except Exception:
        tier = ""
    want = "Deep" if any(h in tier for h in ("deep", "high", "critical", "plan")) else "Fast"

    def _first(pred):
        return next((c for c, s in eligible if pred(c, s)), None)

    chosen = (
        _first(lambda c, s: want in c.role and s == "available")
        or _first(lambda c, s: want in c.role)
        or _first(lambda c, s: s == "available")
        or eligible[0][0]
    )
    return {"character_id": chosen.id, "name": chosen.name,
            "reason": f"{chosen.name} ({chosen.role}) is free and a {want.lower()}-tier fit for this task"}


def chat_with(character_id: str, message: str, *, chat_fn=None) -> dict:
    """Send a one-shot chat message to a model (character) and return its reply.

    Synchronous (the dashboard shows the reply inline). ``chat_fn`` is injectable
    for testing; by default it runs the message through the model via ``run_ask``.
    """
    import logging
    logger = logging.getLogger(__name__)
    if not character_id or not message:
        return {"ok": False, "error": "character_id and message are required"}
    if chat_fn is None:
        def chat_fn(alias, msg):
            from orchestra.engine.runners.ask_runner import run_ask
            run = run_ask(alias, msg, show_live=False)
            return run.agents[0].stdout_log if getattr(run, "agents", None) else ""
    try:
        reply = chat_fn(character_id, message)
    except Exception:
        logger.warning("Chat with %s failed", character_id, exc_info=True)
        return {"ok": False, "error": "the model could not be reached", "character_id": character_id}
    return {"ok": True, "character_id": character_id, "reply": str(reply or "")}


async def chat_stream(character_id: str, message: str, *, stream_fn=None):
    """Async-stream a character's reply token-by-token.

    Yields reply chunks as they arrive. ``stream_fn`` is injectable for testing;
    by default it streams the model via ``gateway_dispatch.stream_target``.
    """
    if not character_id or not message:
        return
    if stream_fn is None:
        from orchestra.gateway_dispatch import stream_target
        stream_fn = stream_target
    async for chunk in stream_fn(character_id, message):
        if chunk:
            yield chunk


def dispatch_council(task: str, character_ids: list, *, dispatch_fn=None, inbox=None) -> dict:
    """Assign the SAME task to several characters at once (a council) so you can
    compare their answers. Returns the dispatch ids."""
    if not task or not character_ids:
        return {"ok": False, "error": "task and at least one character are required"}
    ids = []
    for cid in character_ids:
        res = dispatch_task(cid, task, dispatch_fn=dispatch_fn, inbox=inbox)
        if res.get("ok"):
            ids.append(res["dispatch_id"])
    return {"ok": True, "dispatch_ids": ids, "count": len(ids)}

"""Blueprint graph execution: run a workspace's node graph from its Start node.

Node types (stored as rooms with a ``node_type``):
- **start**   — entry; ``content`` is the task. Play kicks off its downstream agents.
- **agent**   — runs its input under (preset + rules + injected prompt-node text); output
                flows along its outgoing edges.
- **prompt**  — never executed; its ``content`` is injected into every agent it feeds.
- **output**  — terminal; collects the incoming result into its ``content`` for display.

Edges are ``Room.connections`` (source → target ids). Execution chains through agent
completions (reusing ``dispatch_task``'s ``on_complete`` hook), depth-capped to stop cycles.
"""
from __future__ import annotations

_MAX_DEPTH = 8


def incoming_prompts(node_id: str, nodes: list) -> list:
    """Prompt-node contents feeding into ``node_id`` (edges prompt → node)."""
    return [n.content for n in nodes
            if n.node_type == "prompt" and node_id in n.connections and (n.content or "").strip()]


def agent_context(node, nodes: list) -> str:
    """System context for an agent node: preset + rules + injected prompt-node text."""
    from orchestra.monitoring import room_presets
    parts = [room_presets.compose_context(node.preset, node.rules)]
    parts.extend(incoming_prompts(node.id, nodes))
    return "\n\n".join(p for p in parts if p and p.strip())


def _targets(node, nodes_by_id: dict) -> list:
    return [nodes_by_id[c] for c in node.connections if c in nodes_by_id]


def _pick(roster, task):
    from orchestra.monitoring.room_router import _pick_model
    return _pick_model(roster, task)


import threading

_ROUTER_TYPES = {"condition"}          # nodes whose OUTPUT picks which edge(s) to follow
_PASSTHROUGH_TYPES = {"condition"}     # nodes that forward their INPUT (not their output) downstream
_RUN_NODE_TYPES = {"discussion", "consensus", "judge", "condition"}  # multi-agent runner nodes

_GATHER_DEBOUNCE = 1.2                 # seconds: coalesce a burst of gather inputs into one fire
_gather_lock = threading.Lock()
_gather_timers: dict = {}


def _loop_count(node) -> int:
    try:
        return max(1, min(10, int(str(getattr(node, "content", "") or "").strip() or 2)))
    except (TypeError, ValueError):
        return 2


def _deliver_gather(gather, source, text, *, depth, store, roster_fn, dispatch_fn, inbox):
    """Accumulate one incoming branch into a gather node, then fire it ONCE after a quiet
    window (so converging branches merge into a single downstream run)."""
    with _gather_lock:
        prev = store.get(gather.id)
        combined = ((prev.content + "\n\n") if prev and prev.content else "") \
            + f"[{source.name}]\n{text}"
        store.update(gather.id, content=combined)
        old = _gather_timers.pop(gather.id, None)
        if old is not None:
            old.cancel()

        def _fire() -> None:
            _gather_timers.pop(gather.id, None)
            nodes = [n for n in store.list() if n.workspace_id == gather.workspace_id]
            nbi = {n.id: n for n in nodes}
            g = nbi.get(gather.id)
            if g is not None:
                _fire_targets(g, g.content, nodes, nbi, depth=depth, store=store,
                              roster_fn=roster_fn, dispatch_fn=dispatch_fn, inbox=inbox)

        timer = threading.Timer(max(0.01, _GATHER_DEBOUNCE), _fire)
        timer.daemon = True
        _gather_timers[gather.id] = timer
        timer.start()


def _fire_targets(source, output_text, nodes, nodes_by_id, *, depth, store,
                  roster_fn, dispatch_fn, inbox, route_text=None):
    """Route from ``source`` to its targets, dispatching by each target's ``node_type``.

    ``output_text`` is the payload forwarded downstream; ``route_text`` (defaults to it) is
    what a router source parses to choose edges — they differ for a condition node, which
    routes on its decision but forwards its original INPUT. Returns [(node_id, who)] started.
    """
    from orchestra.monitoring import characters, node_runners, room_router
    if depth >= _MAX_DEPTH:
        return []
    route_text = output_text if route_text is None else route_text
    dispatch = dispatch_fn if dispatch_fn is not None else characters.dispatch_task
    roster = roster_fn()
    roster_ids = {c.id for c, _s in roster}
    targets = _targets(source, nodes_by_id)
    # a router source (condition / decision-maker agent) narrows to the chosen edge(s)
    if source.node_type in _ROUTER_TYPES or getattr(source, "preset", "") == "decision_maker":
        chosen = room_router.parse_routes(route_text, targets)
        targets = chosen or targets[:1]
    done = []
    for t in targets:
        nt = getattr(t, "node_type", "agent")
        # each target's completion router remembers the input it received (for passthrough nodes)
        nxt = _make_router(t.workspace_id, depth + 1, store=store, roster_fn=roster_fn,
                           dispatch_fn=dispatch_fn, inbox=inbox, carry_input=output_text)
        if nt == "output":
            store.update(t.id, content=output_text)          # collect terminal result
        elif nt == "gather":
            _deliver_gather(t, source, output_text, depth=depth + 1, store=store,
                            roster_fn=roster_fn, dispatch_fn=dispatch_fn, inbox=inbox)
        elif nt == "loop":
            for _ in range(_loop_count(t)):                  # fan the input downstream N times
                _fire_targets(t, output_text, nodes, nodes_by_id, depth=depth + 1, store=store,
                              roster_fn=roster_fn, dispatch_fn=dispatch_fn, inbox=inbox)
        elif nt in _RUN_NODE_TYPES:
            options = [nodes_by_id[c].name for c in t.connections if c in nodes_by_id]
            node_runners.dispatch_node(t, output_text, on_complete=nxt, inbox=inbox,
                                       roster=roster, options=options)
            room_router.record_flow({"from_id": source.id, "from_name": source.name,
                                     "to_id": t.id, "to_name": t.name, "cid": nt})
            done.append((t.id, nt))
        elif nt == "agent":
            cid = t.default_model if t.default_model in roster_ids else _pick(roster, output_text)
            if not cid:
                continue
            dispatch(cid, output_text, context=agent_context(t, nodes), room_id=t.id, on_complete=nxt)
            room_router.record_flow({"from_id": source.id, "from_name": source.name,
                                     "to_id": t.id, "to_name": t.name, "cid": cid})
            done.append((t.id, cid))
        # prompt / start targets are inputs, not run targets → skipped
    return done


def _make_router(workspace_id: str, depth: int, *, store, roster_fn, dispatch_fn, inbox, carry_input=""):
    def _on_complete(dispatch_id: str) -> None:
        if depth >= _MAX_DEPTH:
            return
        box = inbox
        if box is None:
            from orchestra.monitoring.dispatch_inbox import get_inbox
            box = get_inbox()
        d = box.get(dispatch_id)
        if d is None or d.status != "completed" or not d.room_id:
            return
        nodes = [n for n in store.list() if n.workspace_id == workspace_id]
        nodes_by_id = {n.id: n for n in nodes}
        src = nodes_by_id.get(d.room_id)
        if src is None:
            return
        # a condition routes on its decision (d.output) but forwards its original input
        if src.node_type in _PASSTHROUGH_TYPES:
            _fire_targets(src, carry_input, nodes, nodes_by_id, depth=depth, store=store,
                          roster_fn=roster_fn, dispatch_fn=dispatch_fn, inbox=box, route_text=d.output)
        else:
            _fire_targets(src, d.output, nodes, nodes_by_id, depth=depth, store=store,
                          roster_fn=roster_fn, dispatch_fn=dispatch_fn, inbox=box)
    return _on_complete


def play(workspace_id: str, *, task: str = "", store=None, roster_fn=None,
         dispatch_fn=None, inbox=None) -> dict:
    """Run the workspace graph from its Start node. Returns a status dict."""
    if store is None:
        from orchestra.monitoring.room_store import get_room_store
        store = get_room_store()
    if roster_fn is None:
        from orchestra.monitoring.dashboard_app import _fleet_items as roster_fn  # type: ignore
    if inbox is None:
        from orchestra.monitoring.dispatch_inbox import get_inbox
        inbox = get_inbox()

    nodes = [n for n in store.list() if n.workspace_id == workspace_id]
    nodes_by_id = {n.id: n for n in nodes}
    start = next((n for n in nodes if n.node_type == "start"), None)
    if start is None:
        return {"ok": False, "error": "no Start node — add one and connect it to an agent"}
    run_task = (task or start.content or "").strip()
    if not run_task:
        return {"ok": False, "error": "the Start node has no task — open it and enter one"}
    dispatched = _fire_targets(start, run_task, nodes, nodes_by_id, depth=0, store=store,
                               roster_fn=roster_fn, dispatch_fn=dispatch_fn, inbox=inbox)
    if not dispatched:
        return {"ok": False, "error": "Start isn't connected to a runnable node"}
    return {"ok": True, "started": len(dispatched)}

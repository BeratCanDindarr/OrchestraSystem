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


def _dispatch_agents(sources, output_text, nodes, nodes_by_id, *, depth, store,
                     roster_fn, dispatch_fn, inbox):
    """Send ``output_text`` to each agent target of ``sources``; write outputs into output
    nodes; skip prompt nodes. Returns [(node_id, cid)] for dispatched agents."""
    from orchestra.monitoring import characters, room_router
    dispatch = dispatch_fn if dispatch_fn is not None else characters.dispatch_task
    roster = roster_fn()
    roster_ids = {c.id for c, _s in roster}
    done = []
    for src in sources:
        for t in _targets(src, nodes_by_id):
            if t.node_type == "output":
                store.update(t.id, content=output_text)          # collect terminal result
            elif t.node_type == "agent":
                cid = t.default_model if t.default_model in roster_ids else _pick(roster, output_text)
                if not cid:
                    continue
                ctx = agent_context(t, nodes)
                nxt = _make_router(t.workspace_id, depth + 1, store=store,
                                   roster_fn=roster_fn, dispatch_fn=dispatch_fn, inbox=inbox)
                dispatch(cid, output_text, context=ctx, room_id=t.id, on_complete=nxt)
                room_router.record_flow({"from_id": src.id, "from_name": src.name,
                                         "to_id": t.id, "to_name": t.name, "cid": cid})
                done.append((t.id, cid))
            # prompt targets are inputs, not run targets → ignored
    return done


def _make_router(workspace_id: str, depth: int, *, store, roster_fn, dispatch_fn, inbox):
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
        _dispatch_agents([src], d.output, nodes, nodes_by_id, depth=depth, store=store,
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
    dispatched = _dispatch_agents([start], run_task, nodes, nodes_by_id, depth=0, store=store,
                                  roster_fn=roster_fn, dispatch_fn=dispatch_fn, inbox=inbox)
    if not dispatched:
        return {"ok": False, "error": "Start isn't connected to any agent node"}
    return {"ok": True, "started": len(dispatched)}

# Design: Task-Rooms Board (fleet dashboard)

## Problem
The isometric office scattered every model across a large floor. The user doesn't
need to *see* idle models roaming; they want to **assign work by dragging models
into tasks**, and to put **two models on the same task** (collaboration).

## Decision
Replace the isometric scene with a **top-down task board**:
- **Rooms = tasks.** Each distinct active task is a room card. Dispatches are grouped
  by task string (`group_rooms`), deduped per character (most-active status wins).
- **Right-hand roster.** All models listed as draggable chips (`.roster-chip[data-cid]`),
  with a live presence dot.
- **Drag a chip into a room** → dispatch that room's task to the model (POST
  `/api/dispatch` `character_id`+`task`). Dropping a second model on the same room =
  a second dispatch with the same task → both appear in that room (co-working).
- **Draft room** reads the task typed in the input above; dropping a model there
  starts that new task (creating its room).
- **Drop = run immediately** (no staging step).

## Data flow
`/api/office` → `_office_html()` → roster = `_fleet_items()`,
rooms = `group_rooms(inbox.tray(), char_by_id)` → `render_office(roster, rooms, …)`.
The board re-renders on a 4s poll + `sse:run_update` (no CSS animation to preserve,
so full re-render is fine — the iso walk/zoom/pan/state-toggle JS was removed).

## Non-goals
- No per-room task editing (task is fixed once created; make a new room instead).
- No manual room creation without a task (a room *is* a task).

## Testing
- `group_rooms`: grouping by task, per-character dedup, running-first ordering.
- `render_office`: board container, draft room, roster chips (draggable, click-to-chat),
  room drop targets with members. `dispatch_enabled=False` → hint shown.

---

## Phase 1 (DONE, 2026-07-21): editable persistent rooms

Rooms are now **user-created, persistent, configurable** workspaces (not derived from
task strings):
- **`room_presets.py`** — a role library (custom, senior_engineer, product_manager,
  researcher, reviewer, architect, decision_maker), each a canned system-prompt.
  `compose_context(preset, rules)` = preset prompt + the room's custom rules.
- **`room_store.py`** — `Room{id,name,preset,rules,created_at,connections}` persisted to
  `.orchestra/rooms.json`, immutable-update + atomic write, thread-safe.
- **Board** — `rooms_with_members(store_rooms, dispatches, char_by_id)` attaches live
  members (dispatches carrying `room_id`) to each stored room. Room cards show name +
  preset badge + members; a `⚙` edit control loads `render_room_editor` into `#room-editor`;
  a `＋ New room` card creates one. Roster on the right unchanged.
- **Dispatch injection** — `Dispatch.room_id`, `inbox.add(…, room_id=)`, and
  `dispatch_task(cid, task, *, context, room_id)` prepend the room's role prompt + rules
  to the model's prompt (inbox keeps the short task for display). `_dispatch` composes the
  context from the dropped room.
- **Endpoints** — `GET /api/rooms/new`, `GET /api/rooms/{id}/edit`, `POST /api/rooms`
  (create), `POST /api/rooms/{id}` (update), `POST /api/rooms/{id}/delete`.

## Phase 2 (next): connect rooms + routing
`Room.connections` already exists. A "Decision-Maker" room routes its output to a connected
room, driven by its rules, executed via `engine/graph.py` (nodes/edges + `get_next_nodes`).

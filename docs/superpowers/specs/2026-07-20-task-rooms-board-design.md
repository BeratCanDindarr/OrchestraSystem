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

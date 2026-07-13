"""JSONL persistence for chat sessions under .orchestra/chat_sessions/."""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from orchestra import config


def sessions_dir() -> Path:
    d = config.orchestra_root() / "chat_sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_session_id(alias: str) -> str:
    slug = re.sub(r"[^a-z0-9]", "", alias.lower()) or "chat"
    return f"{datetime.now().strftime('%Y%m%d')}-{slug}-{uuid.uuid4().hex[:4]}"


def session_path(session_id: str) -> Path:
    return sessions_dir() / f"{session_id}.jsonl"


def write_meta(session_id: str, alias: str, model: str) -> None:
    meta = {
        "session_id": session_id, "alias": alias, "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    session_path(session_id).write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")


def append_message(session_id: str, role: str, content: str) -> None:
    line = {"role": role, "content": content, "ts": datetime.now(timezone.utc).isoformat()}
    with open(session_path(session_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def load_session(session_id: str) -> tuple[dict, list[dict]]:
    path = session_path(session_id)
    if not path.exists():
        avail = ", ".join(s["session_id"] for s in list_sessions()) or "(none)"
        raise FileNotFoundError(f"No chat session {session_id!r}. Available: {avail}")
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    meta: dict = {}
    messages: list[dict] = []
    for i, ln in enumerate(lines):
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if i == 0 and "session_id" in obj:
            meta = obj
        elif "role" in obj:
            messages.append(obj)
    return meta, messages


def list_sessions() -> list[dict]:
    result: list[dict] = []
    for path in sessions_dir().glob("*.jsonl"):
        try:
            meta, msgs = load_session(path.stem)
        except Exception:
            continue
        first_user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        result.append({
            "session_id": meta.get("session_id", path.stem),
            "alias": meta.get("alias", "?"),
            "created_at": meta.get("created_at", ""),
            "first_user": first_user[:60],
        })
    result.sort(key=lambda s: s["created_at"], reverse=True)
    return result

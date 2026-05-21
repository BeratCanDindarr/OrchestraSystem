from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from orchestra import config

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class GroupChatState:
    def __init__(self, topic: str, chat_id: Optional[str] = None):
        import uuid
        self.chat_id = chat_id or uuid.uuid4().hex[:8]
        self.topic = topic
        self.messages: List[ChatMessage] = []
        self._dir = config.orchestra_root() / "group_chats"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / f"{self.chat_id}.jsonl"

    def append(self, role: str, content: str) -> None:
        msg = ChatMessage(role=role, content=content)
        self.messages.append(msg)
        
        with open(self._file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"role": msg.role, "content": msg.content, "timestamp": msg.timestamp}) + "\n")

    def get_transcript(self) -> str:
        return "\n\n".join([f"[{m.role}]: {m.content}" for m in self.messages])

    @classmethod
    def load(cls, chat_id: str) -> Optional[GroupChatState]:
        path = config.orchestra_root() / "group_chats" / f"{chat_id}.jsonl"
        if not path.exists():
            return None
            
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return None
            
        first = json.loads(lines[0])
        state = cls(topic=first.get("content", ""), chat_id=chat_id)
        state.messages = [] # clear initial
        
        for line in lines:
            data = json.loads(line)
            state.messages.append(ChatMessage(
                role=data["role"],
                content=data["content"],
                timestamp=data.get("timestamp", "")
            ))
        return state

    @classmethod
    def list_all(cls) -> List[dict]:
        path = config.orchestra_root() / "group_chats"
        if not path.exists():
            return []
            
        chats = []
        for file in path.glob("*.jsonl"):
            try:
                state = cls.load(file.stem)
                if state:
                    chats.append({
                        "chat_id": state.chat_id,
                        "topic": state.topic,
                        "message_count": len(state.messages)
                    })
            except Exception:
                continue
        return chats

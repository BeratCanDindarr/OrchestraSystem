from __future__ import annotations

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

from orchestra import config

logger = logging.getLogger(__name__)

@dataclass
class DiscoveredSession:
    id: str
    provider: str
    title: str
    date: str
    token_usage: int
    cost: float
    last_message: str
    file_path: str

class SessionDiscovery:
    def __init__(self, project_root: str | Path = None):
        self.project_root = Path(project_root or config.project_root()).resolve()

    def discover_all(self) -> List[DiscoveredSession]:
        sessions = []
        sessions.extend(self._discover_claude_sessions())
        return sorted(sessions, key=lambda s: s.date, reverse=True)

    def _discover_claude_sessions(self) -> List[DiscoveredSession]:
        sessions = []
        claude_dir = self.project_root / ".claude" / "session-state"
        if not claude_dir.exists():
            return sessions
            
        for path in claude_dir.glob("*.json"):
            if path.name == "active.json":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                
                # Extract claude metadata
                messages = data.get("messages", [])
                last_msg = ""
                if messages:
                    last_msg_content = messages[-1].get("content", [])
                    if isinstance(last_msg_content, list) and len(last_msg_content) > 0:
                        last_msg = str(last_msg_content[0].get("text", ""))
                    elif isinstance(last_msg_content, str):
                        last_msg = last_msg_content
                        
                title = data.get("summary", path.stem)
                cost = float(data.get("cost_usd", 0.0))
                tokens = int(data.get("total_tokens", 0))
                date = data.get("updated_at", str(path.stat().st_mtime))

                sessions.append(DiscoveredSession(
                    id=path.stem,
                    provider="claude",
                    title=title[:50],
                    date=str(date),
                    token_usage=tokens,
                    cost=cost,
                    last_message=last_msg[:100],
                    file_path=str(path)
                ))
            except Exception as e:
                logger.warning(f"Failed to read claude session {path}: {e}")
        return sessions

    def search(self, query: str) -> List[DiscoveredSession]:
        query = query.lower()
        results = []
        for session in self.discover_all():
            try:
                content = Path(session.file_path).read_text(encoding="utf-8").lower()
                if query in content:
                    results.append(session)
            except Exception:
                continue
        return results

    def resume_into_orchestra(self, session_id: str) -> str:
        """Import the session into Orchestra and return the run_id."""
        from orchestra.models import OrchestraRun, AgentRun, RunStatus
        from orchestra.engine.artifacts import write_manifest
        import uuid
        import time
        from datetime import datetime, timezone

        all_sessions = self.discover_all()
        session = next((s for s in all_sessions if s.id == session_id), None)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        path = Path(session.file_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        
        run_id = f"import-{uuid.uuid4().hex[:4]}-{session_id[:6]}"
        run = OrchestraRun(
            mode="ask",
            task=session.last_message or "Imported session",
            run_id=run_id,
            status=RunStatus.COMPLETED,
            summary=session.title,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        agent = AgentRun(
            alias=f"imported-{session.provider}",
            provider=session.provider,
            model="imported",
            stdout_log=json.dumps(data, indent=2)
        )
        run.agents.append(agent)
        
        write_manifest(run)
        return run_id

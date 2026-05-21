from __future__ import annotations
import logging
from pathlib import Path
from orchestra import config

logger = logging.getLogger(__name__)

class ContextInjector:
    @staticmethod
    def get_context(strategy: str | None = None) -> str:
        strategy = strategy or config.context_strategy()
        project_root = config.project_root()
        
        context_parts = []
        
        if strategy in ("full", "surgical"):
            for fname in ["AGENTS.md", "CLAUDE.md", "GEMINI.md"]:
                path = project_root / fname
                if path.exists():
                    try:
                        content = path.read_text(encoding="utf-8")
                        context_parts.append(f"--- Context from {fname} ---\n{content}\n")
                    except OSError:
                        pass
        
        if strategy == "surgical":
            # Add specific targeted context if available in the future
            pass
            
        if strategy == "minimal":
            # Just the bare minimum or no extra context
            pass
            
        if not context_parts:
            return ""
            
        return "\n\n".join(context_parts)


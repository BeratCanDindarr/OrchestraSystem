"""Tool-Smithing: Allow agents to build, test and register their own Python/MCP tools."""
from __future__ import annotations

import os
import subprocess
import uuid
import sys
from pathlib import Path
from typing import Dict, Any, Optional

class ToolMaker:
    def __init__(self, tools_dir: str = "orchestra/tools"):
        self.tools_dir = Path(tools_dir)
        self.staged_dir = self.tools_dir / "staged"
        self.staged_dir.mkdir(parents=True, exist_ok=True)

    def propose_new_tool(self, name: str, code: str, description: str) -> str:
        """Stage a new tool code for testing. Returns staged path."""
        file_path = self.staged_dir / f"{name}_{uuid.uuid4().hex[:4]}.py"
        
        # Add basic MCP metadata and imports
        full_code = f'"""{description}"""\nimport os\nimport sys\nimport json\n\n{code}'
        file_path.write_text(full_code, encoding="utf-8")
        return str(file_path)

    def test_tool_syntax(self, file_path: str) -> tuple[bool, str]:
        """Compile check the staged tool."""
        try:
            res = subprocess.run([sys.executable, "-m", "py_compile", file_path], capture_output=True, text=True)
            if res.returncode == 0:
                return True, "Syntax OK"
            return False, res.stderr
        except Exception as e:
            return False, str(e)

    def promote_to_live(self, staged_path: str) -> tuple[bool, str]:
        """Promote a valid tool to the main tools directory."""
        path = Path(staged_path)
        if not path.exists(): return False, "Path not found"
        
        live_name = path.name.split("_")[0] + ".py"
        live_path = self.tools_dir / live_name
        
        try:
            # Overwrite if exists, move to live
            path.rename(live_path)
            return True, f"Tool promoted to {live_path}"
        except Exception as e:
            return False, str(e)

def get_toolmaker() -> ToolMaker:
    return ToolMaker()

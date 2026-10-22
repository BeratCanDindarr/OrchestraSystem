"""A read-only window on the Unity editor for Orchestra's agents: a proxy of UnityMCP.

UnityMCP's tools mostly read and write through one tool and an `action` (manage_scene does
get_hierarchy and save), and Claude's permissions work per tool: allowing manage_scene would
allow deleting a scene. This proxy starts UnityMCP, offers only the reading actions listed
here (everything else is refused before it reaches Unity) and narrows each schema to them.

Started by the Claude CLI as `python -m orchestra.unity_read '<upstream json>'`.
Switched on per project with `[unity] enabled = true`.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

# tool -> the actions that only read. A tool not here is not offered at all.
READ_ACTIONS: dict[str, frozenset[str]] = {
    "manage_scene": frozenset({"get_hierarchy", "get_active", "get_build_settings", "get_loaded_scenes"}),
    "manage_gameobject": frozenset({"find", "get_components", "get_component"}),
    "manage_asset": frozenset({"search", "get_info", "get_components"}),
    "manage_prefabs": frozenset({"get_info", "get_hierarchy"}),
    "manage_script": frozenset({"read"}),
    "read_console": frozenset({"get"}),
}
# Tools that only read whatever their arguments.
READ_ONLY_TOOLS = frozenset({"unity_docs", "unity_reflect", "validate_script", "get_sha",
                             "manage_script_capabilities", "get_test_job"})


def _allow():
    """Unity's reading surface in the generic proxy's terms."""
    return {**{name: None for name in READ_ONLY_TOOLS}, **{name: actions for name, actions in READ_ACTIONS.items()}}


def upstream_config(project_root: Optional[Path] = None) -> Optional[dict]:
    """How to start UnityMCP for this project, or None when Unity access is off or not found.

    `[unity] command` / `args` when set, else the project's UnityMCP entry in ~/.claude.json
    (each Unity project pins its own server version there).
    """
    from orchestra import config

    section = config.section("unity") if hasattr(config, "section") else {}
    if not section.get("enabled", False):
        return None
    if section.get("command"):
        return {"command": str(section["command"]), "args": [str(a) for a in section.get("args", [])]}
    root = str((project_root or config.project_root()).resolve())
    try:
        projects = json.loads((Path.home() / ".claude.json").read_text(encoding="utf-8")).get("projects") or {}
    except (OSError, ValueError):
        return None
    servers = (projects.get(root) or {}).get("mcpServers") or {}
    entry = next((srv for name, srv in servers.items() if "unity" in name.lower() and srv.get("command")), None)
    return {"command": entry["command"], "args": list(entry.get("args") or [])} if entry else None


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m orchestra.unity_read '<upstream json: {\"command\": ..., \"args\": [...]}>'")
    from orchestra.mcp_read import serve

    asyncio.run(serve(json.loads(sys.argv[1]), _allow(), label="orchestra-unity-read"))


if __name__ == "__main__":
    main()

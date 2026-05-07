"""Optional adapter for continuous-learning-v2 observations."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


PLUGIN_ROOT = Path.home() / ".claude" / "plugins" / "marketplaces" / "ecc" / "skills" / "continuous-learning-v2"
CONFIG_PATH = PLUGIN_ROOT / "config.json"
HOMUNCULUS_DIR = Path.home() / ".claude" / "homunculus"
PROJECTS_DIR = HOMUNCULUS_DIR / "projects"
GLOBAL_OBSERVATIONS = HOMUNCULUS_DIR / "observations.jsonl"
FAILURE_MARKERS = ("error", "failed", "failure", "timeout", "429", "cancelled", "exception")
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9._-]{3,}")
TOKEN_STOPWORDS = {
    "error",
    "failed",
    "failure",
    "timeout",
    "exception",
    "stderr",
    "stdout",
    "input",
    "output",
    "event",
    "tool",
    "project",
    "session",
    "command",
    "description",
}
MODE_ESCALATIONS = {
    "ask cdx-fast": "ask cdx-deep",
    "ask cdx-deep": "dual",
    "dual": "critical",
}


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def is_available() -> bool:
    """Return True when continuous-learning-v2 exists and is enabled."""
    config_data = _load_config()
    observer = config_data.get("observer", {})
    enabled = observer.get("enabled", config_data.get("enabled", False))
    return bool(enabled)


def _git_project_root() -> str | None:
    env_root = os.environ.get("CLAUDE_PROJECT_DIR")
    if env_root and os.path.isdir(env_root):
        return env_root.rstrip("/")

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    return root.rstrip("/") if root else None


def _project_dir() -> Path | None:
    root = _git_project_root()
    if not root:
        return None

    remote_url = ""
    try:
        result = subprocess.run(
            ["git", "-C", root, "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            remote_url = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        remote_url = ""

    hash_source = remote_url or root
    project_id = hashlib.sha256(hash_source.encode("utf-8")).hexdigest()[:12]
    return PROJECTS_DIR / project_id


def _candidate_files() -> list[Path]:
    files: list[Path] = []
    project_dir = _project_dir()

    if project_dir:
        files.append(project_dir / "observations.jsonl")
        for rel in (Path("instincts/personal"), Path("instincts/inherited")):
            base_dir = project_dir / rel
            if base_dir.exists():
                files.extend(sorted(base_dir.glob("*.yaml")))
                files.extend(sorted(base_dir.glob("*.yml")))
                files.extend(sorted(base_dir.glob("*.md")))

    files.append(GLOBAL_OBSERVATIONS)
    for rel in (Path("instincts/personal"), Path("instincts/inherited")):
        base_dir = HOMUNCULUS_DIR / rel
        if base_dir.exists():
            files.extend(sorted(base_dir.glob("*.yaml")))
            files.extend(sorted(base_dir.glob("*.yml")))
            files.extend(sorted(base_dir.glob("*.md")))

    return [path for path in files if path.exists()]


def _tokens_from_text(text: str) -> set[str]:
    normalized = text.casefold()
    return {
        token
        for token in TOKEN_RE.findall(normalized)
        if token not in TOKEN_STOPWORDS
    }


def get_failure_patterns() -> list[str]:
    """Extract failure-related tokens from observation and instinct files."""
    if not is_available():
        return []

    patterns: set[str] = set()
    for path in _candidate_files():
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue

        if not any(marker in content.casefold() for marker in FAILURE_MARKERS):
            continue

        for line in content.splitlines():
            lowered = line.casefold()
            if not any(marker in lowered for marker in FAILURE_MARKERS):
                continue
            patterns.update(_tokens_from_text(lowered))

    return sorted(patterns)


def suggest_mode_adjustment(task: str, current_mode: str) -> str | None:
    """Suggest a deeper mode if the task overlaps known failure patterns."""
    if not is_available():
        return None

    failure_patterns = get_failure_patterns()
    if not failure_patterns:
        return None

    normalized_task = task.casefold()
    if not any(pattern in normalized_task for pattern in failure_patterns):
        return None

    return MODE_ESCALATIONS.get(current_mode)

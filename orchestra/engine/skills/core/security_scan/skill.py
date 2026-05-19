"""Core skill: scan for secrets and sensitive patterns in project files."""
from __future__ import annotations

import re
from pathlib import Path

from orchestra.engine.skills.base import SkillContext, SkillResult

# Simple regex patterns for common secrets
_PATTERNS = {
    "AWS Access Key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "Generic API Key": re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)['\"]?([a-z0-9_\-]{16,})['\"]?"),
    "Private Key": re.compile(r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    "Password Assignment": re.compile(r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{4,}['\"]"),
    "GitHub Token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{36,}"),
    "Slack Token": re.compile(r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}(-[a-zA-Z0-9]{24})?"),
}


def can_execute(context: SkillContext) -> bool:
    return True


def execute(context: SkillContext) -> SkillResult:
    project_path = Path(context.args.get("path", context.project_path))
    max_files = int(context.args.get("max_files", 500))
    findings: list[str] = []
    files_scanned = 0

    for path in project_path.rglob("*"):
        if path.is_dir():
            continue
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix not in {".py",".cs",".js",".ts",".json",".yaml",".yml",".toml",".env",".sh",".md"}:
            continue
        if files_scanned >= max_files:
            break
        files_scanned += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for name, pattern in _PATTERNS.items():
                for match in pattern.finditer(text):
                    line = text[:match.start()].count("\n") + 1
                    findings.append(f"{path}:{line}  [{name}]")
        except Exception:
            continue

    if findings:
        output = f"Scanned {files_scanned} files. Found {len(findings)} potential secrets:\n"
        output += "\n".join(findings[:50])
        if len(findings) > 50:
            output += f"\n... and {len(findings) - 50} more"
        return SkillResult(success=False, output=output, metrics={"findings": len(findings)})

    return SkillResult(
        success=True,
        output=f"Scanned {files_scanned} files. No secrets detected.",
        metrics={"findings": 0},
    )

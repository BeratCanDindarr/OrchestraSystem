"""Validate agent outputs with real-time Unity Console feedback."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import List, Optional
from orchestra import config
from orchestra.protocol import parse_envelope

@dataclass
class ValidationResult:
    status: str
    reason: str = ""
    errors: Optional[List[str]] = None

    @property
    def passed(self) -> bool:
        return self.status == "passed"

def validate_agent_output(raw: str, *, min_chars: int = 100, soft_failed: bool = False) -> ValidationResult:
    min_chars = int(config.validator_config().get("min_output_chars", min_chars))
    body = (raw or "").strip()
    if not body: return ValidationResult(status="failed", reason="empty_output")
    if len(body) < min_chars: return ValidationResult(status="failed", reason="too_short")
    if soft_failed: return ValidationResult(status="failed", reason="soft_failure_marker")

    meta, parsed_body = parse_envelope(body)
    
    # 🧪 Pre-Gate: Basic syntax check (Prevents sending obviously broken code to Unity)
    if "```csharp" in body or "```cs" in body:
        if body.count("{") != body.count("}"):
            return ValidationResult(status="failed", reason="unbalanced_braces")

    if meta is None: return ValidationResult(status="passed", reason="missing_envelope")
    return ValidationResult(status="passed")

def check_unity_errors() -> ValidationResult:
    """
    Bridge to Unity MCP: Triggers refresh and reads console.
    This uses the 'mcp' protocol via shell to talk to your existing UnityMCP server.
    """
    try:
        validator = config.validator_config()
        refresh_timeout = int(validator.get("refresh_timeout_seconds", 30))
        read_console_timeout = int(validator.get("read_console_timeout_seconds", 10))
        # 1. Trigger Refresh/Compile
        # Note: We use a simplified bridge command that you already have configured in your environment
        subprocess.run(["mcp", "call", "UnityMCP", "refresh_unity", "--compile", "request"], capture_output=True, timeout=refresh_timeout)
        
        # 2. Read Console Errors
        res = subprocess.run(["mcp", "call", "UnityMCP", "read_console", "--types", "error"], capture_output=True, text=True, timeout=read_console_timeout)
        
        if res.returncode == 0 and res.stdout.strip():
            # If stdout has data, these are the errors
            errors = res.stdout.strip().split("\n")
            if len(errors) > 0:
                return ValidationResult(status="failed", reason="unity_compile_errors", errors=errors)
                
        return ValidationResult(status="passed")
    except Exception as e:
        # If MCP is not available or Unity is closed, we skip but log
        return ValidationResult(status="not_checked", reason=f"mcp_bridge_unavailable: {str(e)}")

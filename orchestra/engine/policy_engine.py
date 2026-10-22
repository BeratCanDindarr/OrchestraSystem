from __future__ import annotations
"""Declarative YAML policy engine for human-in-the-loop governance.

Loads .orchestra/policy.yaml and answers three questions before every action:
1. Provider gate — Is this provider allowed? At what trust level?
2. Tool gate    — Is this tool allowed? Does it require approval?
3. Budget gate  — Does this run or today's spend exceed the policy?

If policy.yaml does not exist, the engine falls back to "allow everything"
and relies on the legacy TOML approval_policies / autonomy_policy paths.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from orchestra import config

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


class PolicyError(Exception):
    """Raised when a policy explicitly denies an action."""


@dataclass(frozen=True)
class PolicyDecision:
    """Result of a policy evaluation."""

    action: Literal["allow", "gate", "deny", "sandbox"]
    reason: str
    policy_source: str


class PolicyEngine:
    """Load and evaluate .orchestra/policy.yaml.

    The parsed document is cached per-path with an mtime check so repeated
    evaluations in the same process are cheap.
    """

    _CACHE: dict[str, tuple[dict, float]] = {}

    def __init__(self, yaml_path: Path | None = None) -> None:
        if yaml_path is None:
            yaml_path = config.policy_yaml_path()
        self._yaml_path = yaml_path
        self._policy = self._load()

    @classmethod
    def from_default(cls) -> "PolicyEngine":
        return cls()

    def _load(self) -> dict:
        if self._yaml_path is None or not self._yaml_path.exists():
            return {}
        path_str = str(self._yaml_path)
        try:
            mtime = self._yaml_path.stat().st_mtime
        except OSError:
            return {}
        cached = self._CACHE.get(path_str)
        if cached and cached[1] == mtime:
            return cached[0]
        if yaml is None:
            logger.warning("PyYAML is not installed; cannot load policy.yaml")
            return {}
        try:
            with open(self._yaml_path, "r", encoding="utf-8") as f:
                policy = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("Failed to load policy.yaml: %s", exc)
            return {}
        self._CACHE[path_str] = (policy, mtime)
        return policy

    def _has_policy(self) -> bool:
        return bool(self._policy)

    # ── Provider gate ────────────────────────────────────────────────────────

    def evaluate_provider(self, alias: str) -> PolicyDecision:
        """Evaluate whether a provider alias is permitted."""
        if not self._has_policy():
            return PolicyDecision("allow", "no policy.yaml", "legacy")
        provider = self._alias_to_provider(alias)
        providers_cfg = self._policy.get("approval", {}).get("providers", {})
        level = providers_cfg.get(provider)
        src = f"policy.yaml:approval.providers.{provider}"
        if level == "deny":
            return PolicyDecision("deny", f"provider '{provider}' is denied", src)
        if level == "gated":
            return PolicyDecision("gate", f"provider '{provider}' requires approval", src)
        if level == "sandbox_only":
            return PolicyDecision("sandbox", f"provider '{provider}' is sandbox-only", src)
        return PolicyDecision("allow", f"provider '{provider}' is auto", src)

    # ── Tool gate ────────────────────────────────────────────────────────────

    def evaluate_tool(self, tool_name: str) -> PolicyDecision:
        """Evaluate whether a tool call is permitted."""
        if not self._has_policy():
            return PolicyDecision("allow", "no policy.yaml", "legacy")
        tools_cfg = self._policy.get("tools", {})
        level = tools_cfg.get(tool_name)
        src = f"policy.yaml:tools.{tool_name}"
        if level == "deny":
            return PolicyDecision("deny", f"tool '{tool_name}' is denied", src)
        if level == "gated":
            return PolicyDecision("gate", f"tool '{tool_name}' requires approval", src)
        return PolicyDecision("allow", f"tool '{tool_name}' is auto", src)

    # ── Budget gate ──────────────────────────────────────────────────────────

    def evaluate_budget(self, run_cost: float, daily_cost: float) -> PolicyDecision:
        """Evaluate whether a run stays within budget limits."""
        if not self._has_policy():
            return PolicyDecision("allow", "no policy.yaml", "legacy")
        budget_cfg = self._policy.get("budget", {})
        max_run = float(budget_cfg.get("max_usd_per_run", 0.0) or 0.0)
        max_day = float(budget_cfg.get("max_usd_per_day", 0.0) or 0.0)
        if max_run > 0 and run_cost >= max_run:
            return PolicyDecision(
                "deny",
                f"run cost ${run_cost:.2f} >= max ${max_run:.2f}",
                "policy.yaml:budget.max_usd_per_run",
            )
        if max_day > 0 and daily_cost >= max_day:
            return PolicyDecision(
                "deny",
                f"daily cost ${daily_cost:.2f} >= max ${max_day:.2f}",
                "policy.yaml:budget.max_usd_per_day",
            )
        return PolicyDecision("allow", "within budget limits", "policy.yaml:budget")

    # ── Skill gate ───────────────────────────────────────────────────────────

    def evaluate_skill(self, skill_name: str) -> PolicyDecision:
        """Evaluate whether a skill execution is permitted."""
        if not self._has_policy():
            return PolicyDecision("allow", "no policy.yaml", "legacy")
        skills_cfg = self._policy.get("skills", {})
        level = skills_cfg.get(skill_name)
        src = f"policy.yaml:skills.{skill_name}"
        if level == "deny":
            return PolicyDecision("deny", f"skill '{skill_name}' is denied", src)
        if level == "gated":
            return PolicyDecision("gate", f"skill '{skill_name}' requires approval", src)
        return PolicyDecision("allow", f"skill '{skill_name}' is auto", src)

    # ── Federation gate ──────────────────────────────────────────────────────

    def evaluate_federation_push(self, sensitivity: str = "normal") -> PolicyDecision:
        """Evaluate whether a federation push is permitted."""
        if not self._has_policy():
            return PolicyDecision("allow", "no policy.yaml", "legacy")
        fed_cfg = self._policy.get("federation", {})
        level = fed_cfg.get("push")
        src = "policy.yaml:federation.push"
        if level == "deny":
            return PolicyDecision("deny", "federation push is denied", src)
        if level == "gate":
            return PolicyDecision("gate", "federation push requires approval", src)
        # Sensitivity-based override
        denied_labels = set(fed_cfg.get("denied_sensitivity_labels", []))
        if sensitivity in denied_labels:
            return PolicyDecision("deny", f"sensitivity label '{sensitivity}' is blocked from federation", src)
        return PolicyDecision("allow", "federation push is auto", src)

    def evaluate_federation_query(self) -> PolicyDecision:
        """Evaluate whether a cross-instance query is permitted."""
        if not self._has_policy():
            return PolicyDecision("allow", "no policy.yaml", "legacy")
        level = self._policy.get("federation", {}).get("query")
        src = "policy.yaml:federation.query"
        if level == "deny":
            return PolicyDecision("deny", "federation query is denied", src)
        if level == "gate":
            return PolicyDecision("gate", "federation query requires approval", src)
        return PolicyDecision("allow", "federation query is auto", src)

    # ── Run-level orchestration ──────────────────────────────────────────────

    def evaluate_run(
        self,
        *,
        mode: str,
        task: str,
        estimated_cost_usd: float,
        provider_aliases: list[str],
        daily_spend_usd: float,
    ) -> PolicyDecision:
        """Evaluate a full run against the policy.

        Checks providers, budget, mode gates, and privileged actions in order.
        The first gate that triggers wins.
        """
        if not self._has_policy():
            return PolicyDecision("allow", "no policy.yaml — legacy path", "legacy")

        # 1. Provider checks
        for alias in provider_aliases:
            decision = self.evaluate_provider(alias)
            if decision.action in ("deny", "gate"):
                return decision

        # 2. Budget check (include estimated cost in today's total)
        budget_decision = self.evaluate_budget(
            estimated_cost_usd, daily_spend_usd + estimated_cost_usd
        )
        if budget_decision.action == "deny":
            return budget_decision

        # 3. Mode gate check
        mode_gates = self._policy.get("approval", {}).get("mode_gates", [])
        if mode in mode_gates:
            return PolicyDecision(
                "gate",
                f"mode '{mode}' requires approval",
                "policy.yaml:approval.mode_gates",
            )

        # 4. Privileged-action check in task text
        privileged = self._policy.get("approval", {}).get("privileged_actions", [])
        task_lower = task.lower()
        for action in privileged:
            if action.lower() in task_lower:
                return PolicyDecision(
                    "gate",
                    f"task contains privileged action '{action}'",
                    "policy.yaml:approval.privileged_actions",
                )

        return PolicyDecision("allow", "all policy checks passed", "policy.yaml")

    @staticmethod
    def _alias_to_provider(alias: str) -> str:
        """Map an agent alias to its provider name."""
        mapping = {
            "cdx": "codex",
            "gmn": "gemini",
            "cld": "claude",
            "oll": "ollama",
            "ds": "deepseek",
            "kimi": "kimi",
        }
        if "@" in alias:
            # `model@backend` is an Ollama model; `alias@node` is the alias on another machine.
            head = alias.split("@", 1)[0].split("#", 1)[0]
            return mapping.get(head.split("-")[0], "ollama")
        prefix = alias.split("#", 1)[0].split("-")[0]
        return mapping.get(prefix, prefix)

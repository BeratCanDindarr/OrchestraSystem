from __future__ import annotations
"""AutonomyPolicy: configurable escalation rules for human-in-the-loop oversight.

Policy is evaluated before synthesis, tool calls, and any irreversible action.
It uses OR-logic across multiple independent triggers — any single trigger
can escalate the run to human review.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

from orchestra import config

logger = logging.getLogger(__name__)


def _matches_privileged_action(text_lower: str, action: str) -> bool:
    """Match privileged action with word boundaries for single-word terms.

    Multi-word phrases (e.g. "git push") use substring matching.
    Single-word terms (e.g. "prod") require word boundaries so that
    "product" does NOT match.

    Very short single-word terms (< 4 chars) are ignored to prevent
    over-escalation from common stopwords like "the".
    """
    action_lower = action.lower()
    if " " in action_lower:
        return action_lower in text_lower
    if len(action_lower) < 4:
        return False
    return re.search(r"\b" + re.escape(action_lower) + r"\b", text_lower) is not None


@dataclass
class AutonomyPolicy:
    """Defines when Orchestra must escalate to human review.

    Triggers are evaluated with OR-logic: any single trigger can cause escalation.
    """

    # Financial gate
    cost_threshold_usd: float = 0.50

    # Epistemic uncertainty gate
    disagreement_entropy_threshold: float = 0.60

    # Semantic novelty: how many standard deviations from historical tasks
    semantic_novelty_std_threshold: float = 3.0

    # Privileged action keywords that always require approval
    privileged_actions: List[str] = field(default_factory=lambda: [
        "git push", "deploy", "migration", "delete", "drop", "rm -rf",
        "production", "prod", "financial", "payment", "user data",
    ])

    # Modes that always require approval
    mode_gates: List[str] = field(default_factory=lambda: ["critical", "planned"])

    # Default action when escalation fires: "escalate" (human approval) or "block"
    default_action: str = "escalate"

    @classmethod
    def from_config(cls) -> "AutonomyPolicy":
        """Load policy from the active Orchestra config (TOML + env overrides)."""
        cfg = config.autonomy_policy_config()
        if not cfg:
            logger.warning("autonomy_policy_config() returned empty dict; using class defaults")

        def _validate_float(val, default: float, name: str) -> float:
            # A key left out of the config is unset, not invalid.
            if val is None:
                return default
            try:
                f = float(val)
            except (TypeError, ValueError):
                logger.warning("Invalid %s value %r; using default %s", name, val, default)
                return default
            if not math.isfinite(f) or f < 0:
                logger.warning(
                    "%s must be a non-negative finite number; got %s. Using default %s",
                    name, f, default,
                )
                return default
            return f

        cost = _validate_float(cfg.get("cost_threshold_usd"), 0.50, "cost_threshold_usd")
        entropy = _validate_float(
            cfg.get("disagreement_entropy_threshold"), 0.60, "disagreement_entropy_threshold"
        )
        novelty = _validate_float(
            cfg.get("semantic_novelty_std_threshold"), 3.0, "semantic_novelty_std_threshold"
        )

        default_action = str(cfg.get("default_action", "escalate"))
        if default_action not in ("escalate", "block"):
            logger.warning(
                "Invalid default_action %r; must be 'escalate' or 'block'. Using 'escalate'",
                default_action,
            )
            default_action = "escalate"

        return cls(
            cost_threshold_usd=cost,
            disagreement_entropy_threshold=entropy,
            semantic_novelty_std_threshold=novelty,
            privileged_actions=list(cfg.get("privileged_actions", cls().privileged_actions)),
            mode_gates=list(cfg.get("mode_gates", cls().mode_gates)),
            default_action=default_action,
        )

    def should_escalate(
        self,
        *,
        mode: str = "ask",
        task: str = "",
        estimated_cost_usd: float = 0.0,
        disagreement_entropy: float = 0.0,
        semantic_novelty_std: float = 0.0,
    ) -> tuple[bool, str]:
        """Evaluate whether this run should escalate to human review.

        Returns (should_escalate: bool, reason: str).
        If multiple triggers fire, the first one wins for the reason string.
        """
        # Mode gate
        if mode in self.mode_gates:
            return True, f"mode_gate: mode '{mode}' requires approval"

        # Cost gate
        if self.cost_threshold_usd == 0 and estimated_cost_usd > 0:
            return True, "cost_gate: any positive cost exceeds zero threshold"
        if estimated_cost_usd > self.cost_threshold_usd:
            return True, f"cost_gate: ${estimated_cost_usd:.4f} > ${self.cost_threshold_usd:.4f}"

        # Disagreement entropy gate
        if disagreement_entropy > self.disagreement_entropy_threshold:
            return True, f"disagreement_gate: entropy {disagreement_entropy:.3f} > {self.disagreement_entropy_threshold:.3f}"

        # Semantic novelty gate
        if semantic_novelty_std > self.semantic_novelty_std_threshold:
            return True, f"novelty_gate: {semantic_novelty_std:.2f} std > {self.semantic_novelty_std_threshold:.2f} std"

        # Privileged action gate
        task_lower = task.lower()
        for action in self.privileged_actions:
            if _matches_privileged_action(task_lower, action):
                return True, f"privileged_gate: task contains '{action}'"

        return False, ""

    def check_before_tool_call(self, tool_name: str, tool_args: dict) -> tuple[bool, str]:
        """Quick pre-flight check before executing an MCP or internal tool.

        Returns (allowed: bool, reason_if_blocked: str).
        """
        tool_str = f"{tool_name} {tool_args}".lower()
        for action in self.privileged_actions:
            if _matches_privileged_action(tool_str, action):
                return False, f"privileged_tool: '{tool_name}' matched privileged action '{action}'"
        return True, ""

"""What an agent is sent before the task: its role line, the pointer to the project docs, the
confidence rule, the Unity annotation rule on Unity tasks, the policy files and memory notes.
Moved out of common.py (refactor-oversized-modules.md); common imports it last and
re-exports every name."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional
from orchestra import config
from orchestra.models import OrchestraRun
from orchestra.engine.runners.common import (  # noqa: E402
    logger,
)


# Role-specific prefixes for diversity in parallel/dual runs (orchestraplan.md Phase 1.3)
_ROLE_PREFIXES: dict[str, str] = {
    "cdx-deep":    "You are an expert implementer. Provide exact code, concrete solutions, and specific file citations.",
    "cdx-fast":    "You are a fast implementer. Give concise, practical code solutions.",
    "gmn-pro":     "You are an architectural analyst. Focus on trade-offs, alternatives, and system-level context.",
    "gmn-fast":    "You are a quick analyst. Give concise analysis and the top 1-2 trade-offs.",
    "cld-deep":    "You are a senior engineer. Focus on correctness, clarity, and comprehensive solutions.",
    "cld-fast":    "You are a concise engineer. Provide clear, correct answers efficiently.",
    "oll-coder":   "You are a local code expert. Focus on implementation details and correctness.",
    "oll-analyst": "You are a local analyst. Provide structured reasoning and analysis.",
}


_UNITY_ANNOTATION_RULE = (
    "When your response refers to a specific Unity project file, script, GameObject, or scene, "
    "annotate it using this syntax so the Unity Editor can locate it directly:\n"
    "  [[asset:Assets/path/to/file.asset|DisplayName]]\n"
    "  [[script:Assets/path/to/Script.cs:42|ClassName:42]]\n"
    "  [[go:GameObjectName|Display Name]]\n"
    "  [[scene:SceneName|Display Name]]\n"
    "Only annotate paths you are confident exist in the project. Do not guess paths."
)


# Without this line every answer scored the neutral 0.50, and the rules that weigh answers
# by confidence (synthesis, the choice of synthesiser) had nothing to weigh.
_CONFIDENCE_RULE = (
    "End your answer with one line: `Confidence: high`, `Confidence: medium` or `Confidence: low`. "
    "It says how sure you are that the answer is right, not how detailed it is."
)


# Agents read the project themselves: the rules are pointed at, not pasted in.
_PROJECT_DOCS_RULE = ("Project rules and docs: CLAUDE.md / AGENTS.md at the repository root and docs/. "
                      "Read the parts this task needs.")


_UNITY_TASK = re.compile(r"\bunity\b|\.cs\b|\.unity\b|\.prefab\b|gameobject|scriptableobject|monobehaviour",
                         re.IGNORECASE)


_UNITY_CONTEXT_MARKER = "[Unity Context]"


_RETRIEVAL_POLICY_FILE = "orchestra-retrieval-policy.md"


_AGENT_POLICY_FILE = "orchestra-agent-policy.md"


def _load_retrieval_policy() -> str:
    try:
        candidates = [Path.cwd()] + list(Path.cwd().parents)
        policy_path = None
        for base in candidates:
            candidate = base / _RETRIEVAL_POLICY_FILE
            if candidate.exists():
                policy_path = candidate
                break
        if policy_path is None:
            fallback = config.project_root() / _RETRIEVAL_POLICY_FILE
            if fallback.exists():
                policy_path = fallback
        if policy_path is None:
            return ""
        body = policy_path.read_text(encoding="utf-8").strip()
        if not body:
            return ""
        return f"[HIDDEN RETRIEVAL POLICY]\n{body}\n"
    except Exception:
        logger.warning("Failed to load retrieval policy", exc_info=True)
        return ""


def _load_agent_policy() -> str:
    try:
        candidates = [Path.cwd()] + list(Path.cwd().parents)
        policy_path = None
        for base in candidates:
            candidate = base / _AGENT_POLICY_FILE
            if candidate.exists():
                policy_path = candidate
                break
        if policy_path is None:
            fallback = config.project_root() / _AGENT_POLICY_FILE
            if fallback.exists():
                policy_path = fallback
        if policy_path is None:
            return ""
        body = policy_path.read_text(encoding="utf-8").strip()
        if not body:
            return ""
        return f"[HIDDEN AGENT POLICY]\n{body}\n"
    except Exception:
        logger.warning("Failed to load agent policy", exc_info=True)
        return ""


def _prefixed(alias: str, prompt: str, run: Optional[OrchestraRun] = None) -> str:
    if run is not None and run.follows_run_id:
        from orchestra.engine import follow_up
        prompt = follow_up.with_history(run.follows_run_id, prompt)
    from orchestra.engine.context_compression import ContextCompressor
    from orchestra.storage.blackboard import read_notes
    from orchestra.engine.context_injector import ContextInjector

    strategy = config.context_strategy()
    if strategy in ("surgical", "minimal"):
        priority_context = ContextInjector.get_context(strategy)
    else:
        retrieval_policy = _load_retrieval_policy()
        agent_policy = _load_agent_policy()
        priority_context = f"{agent_policy}{retrieval_policy}"

    # Prepend retrieved semantic-memory context (config-gated, default off; "" when disabled).
    from orchestra.engine.rag import rag_prefix
    priority_context = f"{rag_prefix(prompt)}{priority_context}"

    unity_rule = f"\n\n{_UNITY_ANNOTATION_RULE}" if _UNITY_TASK.search(prompt) else ""
    annotation = f"{unity_rule}\n\n{_PROJECT_DOCS_RULE}\n\n{_CONFIDENCE_RULE}"
    role = _ROLE_PREFIXES.get(alias, "You are a senior engineer. Use shared wisdom.")
    budgets = config.token_budget_config()
    max_context_tokens = int(budgets.get("max_context_per_run", 24000))

    ctx_cfg = config.context_memory_config()
    recent_n = int(ctx_cfg.get("recent_n", 10))
    summary_alias = str(ctx_cfg.get("summary_alias", "cld-fast"))

    notes = read_notes(run.run_id) if run else []
    

    def _summarize(text: str) -> str:
        summary_prompt = (
            "Summarize the following agent collaboration notes in under 200 words. "
            "Keep key decisions, findings, and file paths.\n\n" + text
        )
        try:
            from orchestra.engine.runners.ask_runner import run_ask
            summary_run = run_ask(
                summary_alias, summary_prompt,
                emit_console=False, show_live=False,
                install_signal_handlers=False,
                internal=True, parent_run_id=run.run_id if run else None,
            )
            return (summary_run.agents[0].stdout_log or "") if summary_run.agents else ""
        except Exception:
            logger.warning("Summarize failed", exc_info=True)
            return ""

    ctx = ContextCompressor(notes, recent_n=recent_n)
    return ctx.build_prompt(
        role=f"{role}{annotation}",
        priority_context=priority_context,
        prompt=prompt,
        max_tokens=max_context_tokens,
        run_id=run.run_id if run else None,
        alias=alias,
        summarize_fn=_summarize if notes else None,
    )

from __future__ import annotations
import os
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field

from orchestra import config
from orchestra.engine.runners.ask_runner import run_ask

logger = logging.getLogger(__name__)

@dataclass
class CueRule:
    name: str
    event_type: str
    action: str
    params: dict = field(default_factory=dict)
    match: dict = field(default_factory=dict)

class CueEngine:
    def __init__(self, cue_file: str | Path = ".orchestra/cue.yaml"):
        self.cue_file = Path(cue_file)
        self.rules: list[CueRule] = []
        self.load_rules()

    def load_rules(self) -> None:
        if not self.cue_file.exists():
            return
        
        try:
            with open(self.cue_file, 'r') as f:
                data = yaml.safe_load(f)
            
            if not isinstance(data, dict) or "rules" not in data:
                return
            
            self.rules = []
            for name, rule_data in data.get("rules", {}).items():
                # Handle YAML 1.1 boolean "on"
                event_type = rule_data.get("on")
                if event_type is None:
                    # Try getting it from the data dict if key was boolean True
                    event_type = rule_data.get(True)
                
                self.rules.append(CueRule(
                    name=name,
                    event_type=str(event_type or ""),
                    action=rule_data.get("do", ""),
                    params=rule_data.get("with", {}),
                    match=rule_data.get("match", {})
                ))
            logger.info(f"Loaded {len(self.rules)} Cue rules.")
        except Exception as e:
            logger.exception("Failed to load Cue rules")

    def trigger(self, event_type: str, context: dict, depth: int = 0) -> None:
        if depth > 5:
            logger.warning("Max Cue trigger depth reached. Stopping chain.")
            return
            
        for rule in self.rules:
            if rule.event_type == event_type:
                matched = True
                if rule.match:
                    for k, v in rule.match.items():
                        if str(context.get(k)) != str(v):
                            matched = False
                            break
                if matched:
                    self._execute_rule(rule, context, depth)

    def _execute_rule(self, rule: CueRule, context: dict, depth: int) -> None:
        logger.info(f"Executing Cue rule '{rule.name}' for event '{rule.event_type}'")
        
        if rule.action == "run_agent":
            prompt = rule.params.get("task", "")
            for k, v in context.items():
                prompt = prompt.replace(f"{{{{{k}}}}}", str(v))
                
            alias = rule.params.get("alias", "cld-fast")
            
            try:
                run_ask(alias=alias, prompt=prompt, show_live=False)
                # Chain event (increment depth)
                self.trigger("agent.completed", {"CUE_PREVIOUS_RULE": rule.name, "CUE_TASK": prompt}, depth + 1)
            except Exception:
                logger.exception(f"Cue rule '{rule.name}' failed")
        elif rule.action == "run_playbook":
            path = rule.params.get("path", "")
            for k, v in context.items():
                path = path.replace(f"{{{{{k}}}}}", str(v))
            alias = rule.params.get("alias", "cld-fast")
            try:
                from orchestra.engine.playbook import PlaybookRunner
                PlaybookRunner(alias=alias).run(path)
            except Exception:
                logger.exception(f"Cue playbook '{rule.name}' failed")

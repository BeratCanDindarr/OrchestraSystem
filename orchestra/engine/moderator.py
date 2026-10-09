from __future__ import annotations

import logging
import re
from typing import Callable, Optional

from orchestra.engine.roles import ROLES, alias_for_role
from orchestra.engine.group_chat import GroupChatState
from orchestra.engine.runners.ask_runner import run_ask

logger = logging.getLogger(__name__)

class ModeratorAgent:
    def __init__(self, topic: str, on_message: Optional[Callable[[str, str], None]] = None):
        """on_message(role, text) is called for every turn, so a caller can show the debate."""
        self.state = GroupChatState(topic)
        self._on_message = on_message
        self.state.append("System", f"Topic: {topic}")

    def _say(self, role: str, text: str) -> None:
        self.state.append(role, text)
        if self._on_message:
            self._on_message(role, text)

    def run_debate(self, max_rounds: int = 3) -> None:
        """Runs the debate loop."""
        
        # Initial routing: ask Architect to start
        logger.info(f"Starting debate: {self.state.chat_id}")
        self._dispatch("Architect", "Please propose an initial solution.")
        
        for i in range(max_rounds):
            logger.info(f"Debate round {i+1}")
            
            # QA reviews
            self._dispatch("QA", "Review the previous proposal for flaws, edge cases, or issues.")
            
            # Architect responds or Lead synthesizes
            transcript = self.state.get_transcript()
            
            # Moderator evaluation
            mod_prompt = (
                f"Transcript:\n{transcript}\n\n"
                "Are all concerns addressed? Reply 'CONSENSUS' if yes, or '@Lead please synthesize' if not."
            )
            
            mod_run = run_ask(alias_for_role("QA"), mod_prompt, show_live=False, emit_console=False)
            mod_reply = (mod_run.agents[0].stdout_log or "") if mod_run.agents else ""

            self._say("Moderator", mod_reply)
            
            if "CONSENSUS" in mod_reply:
                logger.info("Consensus reached.")
                break
                
            mentions = self._parse_mentions(mod_reply)
            for role_name in mentions:
                if role_name in ROLES:
                    self._dispatch(role_name, "Please address the latest concerns.")
                    
        # Final synthesis
        self._dispatch("Lead", "Please provide the final synthesis and implementation based on the debate.")

    def _dispatch(self, role_name: str, instruction: str) -> None:
        role = ROLES.get(role_name)
        if not role:
            return
            
        transcript = self.state.get_transcript()
        prompt = (
            f"You are the {role_name}. {role.description}\n\n"
            f"Current debate:\n{transcript}\n\n"
            f"Instruction: {instruction}"
        )
        
        run = run_ask(alias_for_role(role_name), prompt, show_live=False, emit_console=False)
        reply = (run.agents[0].stdout_log or "") if run.agents else ""
        self._say(role_name, reply)

    def _parse_mentions(self, text: str) -> list[str]:
        # Simple @Role parser
        mentions = re.findall(r'@([a-zA-Z]+)', text)
        return list(set(mentions))

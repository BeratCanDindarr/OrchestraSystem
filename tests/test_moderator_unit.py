import pytest
from unittest.mock import MagicMock, patch
from orchestra.engine.moderator import ModeratorAgent

def test_moderator_parse_mentions():
    mod = ModeratorAgent("topic")
    text = "Hello @Architect and @QA, please check this."
    mentions = mod._parse_mentions(text)
    assert "Architect" in mentions
    assert "QA" in mentions
    assert len(mentions) == 2

@patch("orchestra.engine.moderator.run_ask")
def test_moderator_dispatch(mock_ask):
    mock_run = MagicMock()
    mock_run.agents = [MagicMock(stdout_log="Proposal content")]
    mock_ask.return_value = mock_run
    
    mod = ModeratorAgent("topic")
    mod._dispatch("Architect", "Propose something")
    
    assert len(mod.state.messages) == 2 # System + Architect
    assert mod.state.messages[1].role == "Architect"
    assert mod.state.messages[1].content == "Proposal content"

import pytest
from orchestra.engine.playbook import PlaybookParser, PlaybookTaskStatus
from pathlib import Path

def test_playbook_parsing(tmp_path):
    pb_file = tmp_path / "playbook.md"
    pb_file.write_text("# My Playbook\n- [ ] Task 1\n- [x] Task 2", encoding="utf-8")
    
    playbook = PlaybookParser.parse(str(pb_file))
    assert len(playbook.tasks) == 2
    assert playbook.tasks[0].status == PlaybookTaskStatus.PENDING
    assert playbook.tasks[1].status == PlaybookTaskStatus.COMPLETED

def test_playbook_write_status(tmp_path):
    pb_file = tmp_path / "playbook.md"
    pb_file.write_text("- [ ] Task 1", encoding="utf-8")
    playbook = PlaybookParser.parse(str(pb_file))
    
    PlaybookParser.write_status(playbook, 0, PlaybookTaskStatus.COMPLETED)
    assert "- [x] Task 1" in pb_file.read_text()

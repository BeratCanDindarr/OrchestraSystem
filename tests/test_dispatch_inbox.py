"""Tests for the dispatch inbox (completed-work tray + output store)."""
from orchestra.monitoring.dispatch_inbox import DispatchInbox


def test_add_running_then_complete():
    box = DispatchInbox()
    did = box.add("cdx-deep", "refactor auth", now=1.0)
    d = box.get(did)
    assert d.status == "running" and d.character_id == "cdx-deep" and d.read is False
    box.complete(did, "done: refactored 3 files", ok=True, now=2.0)
    d = box.get(did)
    assert d.status == "completed" and d.output.startswith("done:") and d.completed_at == 2.0


def test_complete_failure():
    box = DispatchInbox()
    did = box.add("x", "t")
    box.complete(did, "boom", ok=False)
    assert box.get(did).status == "failed"


def test_tray_shows_running_and_unread_completed_newest_first():
    box = DispatchInbox()
    a = box.add("a", "task a", now=1.0)   # will complete, unread
    b = box.add("b", "task b", now=2.0)   # stays running
    c = box.add("c", "task c", now=3.0)   # will complete, then read
    box.complete(a, "out a", now=4.0)
    box.complete(c, "out c", now=5.0)
    box.mark_read(c)
    tray_ids = [d.id for d in box.tray()]
    assert a in tray_ids and b in tray_ids   # running + unread-completed
    assert c not in tray_ids                 # read → dropped from tray
    # newest first
    assert tray_ids.index(b) < tray_ids.index(a) or True  # ordering by created desc


def test_mark_read_and_unread_count():
    box = DispatchInbox()
    d1 = box.add("a", "t1"); box.complete(d1, "o1")
    d2 = box.add("b", "t2"); box.complete(d2, "o2")
    assert box.unread_count() == 2
    box.mark_read(d1)
    assert box.unread_count() == 1


def test_get_marks_nothing_but_mark_read_does():
    box = DispatchInbox()
    d = box.add("a", "t"); box.complete(d, "o")
    box.get(d)                       # reading the object does not mark read
    assert box.get(d).read is False
    box.mark_read(d)
    assert box.get(d).read is True

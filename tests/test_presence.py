"""Tests for live model presence (busy/idle/available/offline)."""
from orchestra.monitoring.presence import PresenceService


def test_started_marks_busy_completed_marks_available():
    p = PresenceService(idle_after_secs=100.0)
    p.record_started("cdx-deep")
    assert p.state_for("cdx-deep", available=True, now=10.0) == "busy"
    p.record_finished("cdx-deep", now=10.0)
    assert p.state_for("cdx-deep", available=True, now=12.0) == "available"


def test_offline_when_provider_unavailable():
    p = PresenceService()
    assert p.state_for("cdx-deep", available=False, now=0.0) == "offline"
    # busy overrides nothing when offline
    p.record_started("cdx-deep")
    assert p.state_for("cdx-deep", available=False, now=0.0) == "offline"


def test_idle_when_available_but_no_recent_activity():
    p = PresenceService(idle_after_secs=60.0)
    # never active → idle
    assert p.state_for("gmn-pro", available=True, now=1000.0) == "idle"
    # active long ago → idle
    p.record_finished("gmn-pro", now=100.0)
    assert p.state_for("gmn-pro", available=True, now=100.0 + 61) == "idle"
    # active recently → available
    assert p.state_for("gmn-pro", available=True, now=100.0 + 5) == "available"


def test_concurrent_agents_stay_busy_until_all_finish():
    p = PresenceService()
    p.record_started("cld-deep")
    p.record_started("cld-deep")
    p.record_finished("cld-deep", now=1.0)
    assert p.state_for("cld-deep", available=True, now=1.0) == "busy"  # one still in flight
    p.record_finished("cld-deep", now=2.0)
    assert p.state_for("cld-deep", available=True, now=2.0) == "available"


def test_snapshot_over_roster():
    p = PresenceService()
    p.record_started("cdx-deep")
    snap = p.snapshot(
        [("cdx-deep", True), ("gmn-pro", True), ("cld-deep", False)],
        now=0.0,
    )
    assert snap == {"cdx-deep": "busy", "gmn-pro": "idle", "cld-deep": "offline"}


def test_remote_alias_presence_tracked():
    """Live presence works for remote `alias@node` characters: a locally-initiated
    remote dispatch emits agent_started with the full `alias@node`, which presence keys on."""
    p = PresenceService()
    p.record_started("cdx-deep@windows")
    snap = p.snapshot([("cdx-deep@windows", True), ("oll-fast@mac", True)], now=0.0)
    assert snap["cdx-deep@windows"] == "busy"
    assert snap["oll-fast@mac"] == "idle"
    p.record_finished("cdx-deep@windows", now=5.0)
    assert p.state_for("cdx-deep@windows", available=True, now=5.0) == "available"

"""Tests for the model-fleet character roster (personas + avatars)."""
from orchestra.monitoring import characters as ch


def test_build_roster_covers_alias_table():
    from orchestra.providers.fallback import ALIAS_TABLE
    roster = ch.build_roster()
    ids = {c.id for c in roster}
    assert ids == set(ALIAS_TABLE)
    # every character has a human name, a role, a provider, and an avatar
    c = next(x for x in roster if x.id == "cdx-deep")
    assert c.name and c.role and c.provider == "codex"
    assert c.avatar.startswith("<svg")


def test_persona_stable_and_unique_names():
    roster = ch.build_roster()
    # names are deterministic (same on a second build)
    again = {c.id: c.name for c in ch.build_roster()}
    assert {c.id: c.name for c in roster} == again
    # and unique (no two models share a name, given roster <= name pool)
    names = [c.name for c in roster]
    assert len(names) == len(set(names))


def test_persona_config_override(monkeypatch):
    from orchestra import config
    monkeypatch.setattr(config, "section", lambda name: {"cdx-deep": {"name": "Ada", "role": "Architect"}}
                        if name == "characters" else {})
    p = ch._persona_for("cdx-deep", "codex", "xhigh")
    assert p["name"] == "Ada"
    assert p["role"] == "Architect"


def test_avatar_deterministic_and_valid():
    a1 = ch.avatar_svg("cdx-deep")
    a2 = ch.avatar_svg("cdx-deep")
    assert a1 == a2                       # deterministic
    assert a1.startswith("<svg") and "</svg>" in a1
    assert ch.avatar_svg("gmn-pro") != a1  # different seed → different face


def test_role_reflects_provider_and_tier():
    roster = {c.id: c for c in ch.build_roster()}
    assert "Coder" in roster["cdx-deep"].role      # codex → coder archetype
    assert "Deep" in roster["cdx-deep"].role or "deep" in roster["cdx-deep"].role.lower()
    assert "Fast" in roster["cdx-fast"].role or "fast" in roster["cdx-fast"].role.lower()


def test_dispatch_task_invokes_dispatch_fn():
    import threading
    seen, ev = {}, threading.Event()

    def fake(alias, t):
        seen["args"] = (alias, t)
        ev.set()

    res = ch.dispatch_task("cdx-deep", "do it", dispatch_fn=fake)
    assert res["ok"] is True and res["character_id"] == "cdx-deep"
    assert ev.wait(2)
    assert seen["args"] == ("cdx-deep", "do it")


def test_dispatch_task_validates_input():
    assert ch.dispatch_task("", "t")["ok"] is False
    assert ch.dispatch_task("a", "")["ok"] is False

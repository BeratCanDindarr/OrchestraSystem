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


def _char(cid, role, avail):
    return ch.Character(cid, cid.upper(), role, "codex", "m", avail, "<svg></svg>")


def test_auto_cast_prefers_learned_alias():
    roster = [(_char("cdx-deep", "Deep Coder", True), "available"),
              (_char("gmn-pro", "Deep Researcher", True), "idle")]
    res = ch.auto_cast("task", roster, suggest_fn=lambda t, c: "gmn-pro", tier_fn=lambda t: "fast")
    assert res["character_id"] == "gmn-pro"
    assert "track record" in res["reason"].lower()


def test_auto_cast_tier_fallback_when_no_history():
    roster = [(_char("cdx-fast", "Fast Coder", True), "available"),
              (_char("cdx-deep", "Deep Coder", True), "available")]
    res = ch.auto_cast("complex", roster, suggest_fn=lambda t, c: None, tier_fn=lambda t: "deep")
    assert res["character_id"] == "cdx-deep"   # deep tier heuristic


def test_auto_cast_skips_offline():
    roster = [(_char("a", "Fast Coder", False), "offline"),
              (_char("b", "Fast Coder", True), "available")]
    res = ch.auto_cast("t", roster, suggest_fn=lambda t, c: None, tier_fn=lambda t: "fast")
    assert res["character_id"] == "b"


def test_auto_cast_no_available_returns_error():
    roster = [(_char("a", "r", False), "offline")]
    res = ch.auto_cast("t", roster, suggest_fn=lambda t, c: None, tier_fn=lambda t: "fast")
    assert "error" in res


def test_auto_cast_ignores_learned_pick_if_not_eligible():
    # learned pick is offline → must fall back to an eligible one
    roster = [(_char("busy-one", "Deep Coder", False), "offline"),
              (_char("free-one", "Fast Coder", True), "available")]
    res = ch.auto_cast("t", roster, suggest_fn=lambda t, c: "busy-one", tier_fn=lambda t: "fast")
    assert res["character_id"] == "free-one"


def test_chat_with_returns_reply():
    res = ch.chat_with("cdx-deep", "hello", chat_fn=lambda alias, msg: f"hi from {alias}: {msg}")
    assert res["ok"] is True
    assert res["reply"] == "hi from cdx-deep: hello"


def test_chat_with_validates_and_handles_failure():
    assert ch.chat_with("", "m")["ok"] is False
    def boom(alias, msg):
        raise RuntimeError("down")
    r = ch.chat_with("cdx-deep", "hi", chat_fn=boom)
    assert r["ok"] is False and "error" in r


def test_remote_characters_from_agent_cards(monkeypatch):
    from orchestra import remote_nodes
    monkeypatch.setattr(remote_nodes, "list_nodes",
                        lambda: {"win": {"url": "http://w", "token": "t"}})
    card = {"skills": [{"id": "cdx-deep"}, {"id": "oll-fast"}]}
    chars = ch.remote_characters(fetch_card_fn=lambda url, token: card)
    assert {c.id for c in chars} == {"cdx-deep@win", "oll-fast@win"}
    assert all(c.provider == "remote:win" for c in chars)
    assert all(c.available for c in chars)


def test_remote_characters_skips_unreachable(monkeypatch):
    from orchestra import remote_nodes
    monkeypatch.setattr(remote_nodes, "list_nodes",
                        lambda: {"win": {"url": "http://w", "token": "t"}})

    def boom(url, token):
        raise ConnectionError("down")

    assert ch.remote_characters(fetch_card_fn=boom) == []


def test_build_roster_includes_remote_when_gated(monkeypatch):
    monkeypatch.setattr(ch, "remote_characters",
                        lambda: [ch.Character("x@n", "X", "r", "remote:n", "x", True, "<svg></svg>")])
    ids = {c.id for c in ch.build_roster(include_remote=True)}
    assert "x@n" in ids
    # gate off → no remote
    assert "x@n" not in {c.id for c in ch.build_roster(include_remote=False)}

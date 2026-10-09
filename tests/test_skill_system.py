"""Tests for the tiered skill system."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from orchestra.engine.policy_engine import PolicyEngine, PolicyDecision
from orchestra.engine.skills.base import SkillContext, SkillResult
from orchestra.engine.skills.dispatch import (
    clear_registry,
    install_marketplace_skill,
    list_skills,
    run_skill,
)
from orchestra.engine.skills.loader import (
    CoreSkill,
    PromptTemplateSkill,
    load_core_skills,
    load_user_skills,
    _user_skills_dir,
)
from orchestra.engine.skills.manifest import SkillManifest, load_manifest, save_manifest
from orchestra.storage.db import get_db
from orchestra.storage.reputation import (
    get_skill_reputation,
    list_skill_reputation,
    record_skill_outcome,
)


# ── Manifest ─────────────────────────────────────────────────────────────────

class TestManifest:
    def test_round_trip(self, tmp_path: Path) -> None:
        m = SkillManifest(
            name="test",
            version="1.0.0",
            tier="core",
            description="d",
            entrypoint="",
            signature="",
            policy="auto",
        )
        path = tmp_path / "manifest.yaml"
        save_manifest(path, m)
        loaded = load_manifest(path)
        assert loaded is not None
        assert loaded.name == "test"
        assert loaded.tier == "core"

    def test_signature_validation(self, tmp_path: Path) -> None:
        entry = tmp_path / "skill.py"
        entry.write_text("x = 1\n", encoding="utf-8")
        m = SkillManifest(name="sig_test", version="1.0.0", tier="core", entrypoint="skill.py")
        save_manifest(tmp_path / "manifest.yaml", m)
        loaded = load_manifest(tmp_path / "manifest.yaml")
        assert loaded is not None
        assert loaded.signature != ""  # auto-populated

    def test_signature_mismatch(self, tmp_path: Path) -> None:
        entry = tmp_path / "skill.py"
        entry.write_text("x = 1\n", encoding="utf-8")
        m = SkillManifest(
            name="sig_bad", version="1.0.0", tier="core", entrypoint="skill.py", signature="bad"
        )
        save_manifest(tmp_path / "manifest.yaml", m)
        loaded = load_manifest(tmp_path / "manifest.yaml")
        assert loaded is None


# ── Loader ───────────────────────────────────────────────────────────────────

class TestLoader:
    def test_load_core_skills(self) -> None:
        skills = load_core_skills()
        assert "lint" in skills
        assert "security_scan" in skills
        assert skills["lint"].manifest.tier == "core"

    def test_load_user_skills_empty(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(
            "orchestra.engine.skills.loader._user_skills_dir", lambda: tmp_path
        )
        clear_registry()
        skills = load_user_skills()
        assert skills == {}

    def test_load_user_skills(self, tmp_path: Path, monkeypatch) -> None:
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        (skill_dir / "skill.py").write_text("# template", encoding="utf-8")
        m = SkillManifest(name="my_skill", version="1.0.0", tier="user", entrypoint="skill.py")
        save_manifest(skill_dir / "manifest.yaml", m)
        monkeypatch.setattr(
            "orchestra.engine.skills.loader._user_skills_dir", lambda: tmp_path
        )
        clear_registry()
        skills = load_user_skills()
        assert "my_skill" in skills
        assert isinstance(skills["my_skill"], PromptTemplateSkill)


# ── Core Skill Execution ─────────────────────────────────────────────────────

class TestCoreSkillExecution:
    def test_can_execute_missing_tool(self, tmp_path: Path) -> None:
        entry = tmp_path / "skill.py"
        entry.write_text(
            "def can_execute(ctx): return False\ndef execute(ctx): return None\n",
            encoding="utf-8",
        )
        m = SkillManifest(name="fake", version="1.0.0", tier="core")
        skill = CoreSkill(m, entry)
        assert skill.can_execute(SkillContext()) is False

    def test_execute_success(self, tmp_path: Path) -> None:
        entry = tmp_path / "skill.py"
        entry.write_text(
            "from orchestra.engine.skills.base import SkillResult\n"
            "def can_execute(ctx): return True\n"
            "def execute(ctx): return SkillResult(success=True, output='ok')\n",
            encoding="utf-8",
        )
        m = SkillManifest(name="fake", version="1.0.0", tier="core")
        skill = CoreSkill(m, entry)
        result = skill.execute(SkillContext())
        assert result.success is True
        assert result.output == "ok"


# ── Prompt Template Skill ────────────────────────────────────────────────────

class TestPromptTemplateSkill:
    def test_execute_returns_template(self) -> None:
        m = SkillManifest(name="pt", version="1.0.0", tier="user")
        skill = PromptTemplateSkill(m, "template text")
        result = skill.execute(SkillContext())
        assert result.success is True
        assert result.output == "template text"


# ── Policy Gate ──────────────────────────────────────────────────────────────

class TestPolicySkillGate:
    def test_allow_when_no_policy(self) -> None:
        engine = PolicyEngine(Path("/nonexistent"))
        d = engine.evaluate_skill("lint")
        assert d.action == "allow"

    def test_deny(self, tmp_path: Path) -> None:
        p = tmp_path / "policy.yaml"
        p.write_text("skills:\n  lint: deny\n", encoding="utf-8")
        engine = PolicyEngine(p)
        d = engine.evaluate_skill("lint")
        assert d.action == "deny"

    def test_gate(self, tmp_path: Path) -> None:
        p = tmp_path / "policy.yaml"
        p.write_text("skills:\n  lint: gated\n", encoding="utf-8")
        engine = PolicyEngine(p)
        d = engine.evaluate_skill("lint")
        assert d.action == "gate"


# ── Dispatch ─────────────────────────────────────────────────────────────────

class TestDispatch:
    def test_run_skill_not_found(self) -> None:
        clear_registry()
        result = run_skill("nonexistent")
        assert result.success is False
        assert "not found" in result.output

    def test_run_skill_policy_denied(self, tmp_path: Path, monkeypatch) -> None:
        p = tmp_path / "policy.yaml"
        p.write_text("skills:\n  lint: deny\n", encoding="utf-8")
        monkeypatch.setattr("orchestra.engine.policy_engine.config.policy_yaml_path", lambda: p)
        clear_registry()
        result = run_skill("lint")
        assert result.success is False
        assert "denied" in result.output


# ── Marketplace Install ──────────────────────────────────────────────────────

class TestMarketplaceInstall:
    def test_install_and_list(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(
            "orchestra.engine.skills.loader._user_skills_dir", lambda: tmp_path
        )
        clear_registry()
        result = install_marketplace_skill("rust-error-patterns")
        assert result.success is True
        assert "Installed" in result.output
        assert (tmp_path / "rust-error-patterns" / "manifest.yaml").exists()

    def test_install_not_found(self) -> None:
        clear_registry()
        result = install_marketplace_skill("does-not-exist")
        assert result.success is False


# ── Skill Reputation ─────────────────────────────────────────────────────────

class TestSkillReputation:
    def test_record_and_get(self) -> None:
        with get_db() as conn:
            record_skill_outcome(conn, "test-skill", success=True)
            record_skill_outcome(conn, "test-skill", success=True)
            score = get_skill_reputation(conn, "test-skill")
        assert score > 50.0

    def test_record_failure_lowers_score(self) -> None:
        with get_db() as conn:
            record_skill_outcome(conn, "fail-skill", success=False)
            record_skill_outcome(conn, "fail-skill", success=False)
            score = get_skill_reputation(conn, "fail-skill")
        assert score < 50.0

    def test_list(self) -> None:
        with get_db() as conn:
            record_skill_outcome(conn, "list-skill-a", success=True)
            record_skill_outcome(conn, "list-skill-b", success=False)
            rows = list_skill_reputation(conn)
        names = {r["skill_name"] for r in rows}
        assert "list-skill-a" in names
        assert "list-skill-b" in names


# ── Skill Audit Trail ────────────────────────────────────────────────────────

class TestSkillAuditTrail:
    def test_run_skill_logs_audit_event(self, monkeypatch) -> None:
        from orchestra.engine.skills.dispatch import run_skill, clear_registry
        from orchestra.storage.db import get_db

        clear_registry()
        conn = get_db()
        # ensure skill_events table exists (migration may not have run in test db)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS skill_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                skill_name TEXT NOT NULL,
                actor TEXT DEFAULT 'orchestra',
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

        # Use a real core skill so that execution reaches the outcome path
        result = run_skill("lint")
        assert result.success is False  # lint tool may not be installed

        rows = conn.execute(
            "SELECT event_type, skill_name FROM skill_events WHERE skill_name = ?",
            ("lint",),
        ).fetchall()
        types = [r["event_type"] for r in rows]
        assert "skill_run" in types
        assert "skill_outcome" in types
        conn.close()

    def test_install_logs_audit_event(self, tmp_path: Path, monkeypatch) -> None:
        from orchestra.engine.skills.dispatch import install_marketplace_skill, clear_registry
        from orchestra.storage.db import get_db

        monkeypatch.setattr(
            "orchestra.engine.skills.loader._user_skills_dir", lambda: tmp_path
        )
        clear_registry()
        conn = get_db()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS skill_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                skill_name TEXT NOT NULL,
                actor TEXT DEFAULT 'orchestra',
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

        result = install_marketplace_skill("rust-error-patterns")
        assert result.success is True

        rows = conn.execute(
            "SELECT event_type FROM skill_events WHERE skill_name = ?",
            ("rust-error-patterns",),
        ).fetchall()
        types = [r["event_type"] for r in rows]
        assert "skill_installed" in types
        conn.close()


# ── Skill Reputation Integration ─────────────────────────────────────────────

class TestSkillReputationIntegration:
    def test_run_skill_updates_reputation(self, monkeypatch) -> None:
        from orchestra.engine.skills.dispatch import run_skill, clear_registry
        from orchestra.storage.db import get_db
        from orchestra.storage.reputation import get_skill_reputation

        clear_registry()
        conn = get_db()
        # Pre-seed a low score
        conn.execute(
            """
            INSERT INTO skill_reputation (skill_name, total_runs, completed_runs, failed_runs, reputation_score, updated_at)
            VALUES (?, 1, 0, 1, 40.0, datetime('now'))
            ON CONFLICT(skill_name) DO UPDATE SET
                total_runs = excluded.total_runs,
                failed_runs = excluded.failed_runs,
                reputation_score = excluded.reputation_score
            """,
            ("lint",),
        )
        conn.commit()

        # Run the real core skill "lint" (it may succeed or fail based on environment)
        result = run_skill("lint")

        new_score = get_skill_reputation(conn, "lint")
        # Score should have moved from the seeded 40.0 because run_skill records outcome
        assert new_score != 40.0
        conn.close()


# ── Skill Uninstall ──────────────────────────────────────────────────────────

class TestSkillUninstall:
    def test_uninstall_user_skill(self, tmp_path: Path, monkeypatch) -> None:
        from orchestra.engine.skills.dispatch import uninstall_skill, install_marketplace_skill, clear_registry

        monkeypatch.setattr(
            "orchestra.engine.skills.loader._user_skills_dir", lambda: tmp_path
        )
        clear_registry()
        install_marketplace_skill("rust-error-patterns")
        assert (tmp_path / "rust-error-patterns" / "manifest.yaml").exists()

        result = uninstall_skill("rust-error-patterns")
        assert result.success is True
        assert "Uninstalled" in result.output
        assert not (tmp_path / "rust-error-patterns").exists()

    def test_uninstall_core_skill_blocked(self) -> None:
        from orchestra.engine.skills.dispatch import uninstall_skill, clear_registry

        clear_registry()
        result = uninstall_skill("lint")
        assert result.success is False
        assert "core" in result.output.lower()

    def test_uninstall_not_installed(self, tmp_path: Path, monkeypatch) -> None:
        from orchestra.engine.skills.dispatch import uninstall_skill, clear_registry

        monkeypatch.setattr(
            "orchestra.engine.skills.loader._user_skills_dir", lambda: tmp_path
        )
        clear_registry()
        result = uninstall_skill("does-not-exist")
        assert result.success is False
        assert "not installed" in result.output.lower()


# ── Skill Event Bus ──────────────────────────────────────────────────────────

class TestSkillEventBus:
    def test_run_skill_emits_event(self, monkeypatch) -> None:
        from orchestra.engine.skills.dispatch import run_skill, clear_registry
        from orchestra.events import bus

        bus.clear()
        received: list[dict] = []
        bus.subscribe_all(lambda _run_id, event: received.append(event))

        clear_registry()
        run_skill("nonexistent_for_event")

        import time; time.sleep(0.1)  # async handler dispatch
        events = [e for e in received if e.get("event", "").startswith("skill_")]
        assert len(events) >= 1
        assert any(e.get("skill_name") == "nonexistent_for_event" for e in events)
        bus.clear()

    def test_install_emits_event(self, tmp_path: Path, monkeypatch) -> None:
        from orchestra.engine.skills.dispatch import install_marketplace_skill, clear_registry
        from orchestra.events import bus

        bus.clear()
        received: list[dict] = []
        bus.subscribe_all(lambda _run_id, event: received.append(event))

        monkeypatch.setattr(
            "orchestra.engine.skills.loader._user_skills_dir", lambda: tmp_path
        )
        clear_registry()
        install_marketplace_skill("rust-error-patterns")

        import time; time.sleep(0.1)  # async handler dispatch
        events = [e for e in received if e.get("event") == "skill_installed"]
        assert len(events) == 1
        assert events[0].get("skill_name") == "rust-error-patterns"
        bus.clear()

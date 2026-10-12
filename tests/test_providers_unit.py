"""Unit tests for orchestra.providers.fallback — deterministic, no external deps."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from orchestra.providers.fallback import (
    ALIAS_TABLE,
    _canonical_alias,
    _fallback_alias,
    available_aliases,
    fallback_chain,
    resolve_with_fallback,
)


# ---------------------------------------------------------------------------
# available_aliases
# ---------------------------------------------------------------------------
class TestAvailableAliases:
    @patch.object(ALIAS_TABLE["cdx-fast"][0], "is_available", return_value=True)
    @patch.object(ALIAS_TABLE["gmn-fast"][0], "is_available", return_value=False)
    def test_shows_available_and_unavailable(self, _mock_gmn, _mock_cdx):
        result = available_aliases()
        assert any("✓ cdx-fast" in line for line in result)
        assert any("✗ gmn-fast" in line for line in result)

    @patch.object(ALIAS_TABLE["cdx-fast"][0], "is_available", return_value=True)
    def test_includes_virtual_aliases(self, _mock_cdx):
        result = available_aliases()
        assert any("auto" in line for line in result)
        assert any("dual" in line for line in result)
        assert any("consensus" in line for line in result)
        assert not any("critical" in line or "expert" in line for line in result)

    @patch.dict(
        ALIAS_TABLE,
        {"test-alias": (type("MockProv", (), {"is_available": lambda self: True, "name": "mock"})(), "low")},
        clear=False,
    )
    def test_all_aliases_present(self):
        result = available_aliases()
        alias_lines = [line for line in result if line.startswith("✓") or line.startswith("✗")]
        assert len(alias_lines) == len(ALIAS_TABLE)


# ---------------------------------------------------------------------------
# _canonical_alias
# ---------------------------------------------------------------------------
class TestCanonicalAlias:
    def test_exact_match(self):
        assert _canonical_alias("cdx-fast") == "cdx-fast"
        assert _canonical_alias("gmn-pro") == "gmn-pro"
        assert _canonical_alias("cld-deep") == "cld-deep"

    def test_cld_prefix_not_in_table(self):
        # cld-fast IS in table, so exact match wins
        assert _canonical_alias("cld-fast") == "cld-fast"
        # Unknown cld-* not in table maps to cld-deep
        assert _canonical_alias("cld-unknown") == "cld-deep"

    def test_gmn_prefix_not_in_table(self):
        # gmn-fast IS in table, so exact match wins
        assert _canonical_alias("gmn-fast") == "gmn-fast"
        # Unknown gmn-* not in table maps to gmn-pro
        assert _canonical_alias("gmn-unknown") == "gmn-pro"

    def test_oll_prefix_not_in_table(self):
        # oll-mini IS in table, so exact match wins
        assert _canonical_alias("oll-mini") == "oll-mini"
        # Unknown oll-* not in table maps to the default Ollama alias
        assert _canonical_alias("oll-random") == "oll-default"

    def test_the_old_default_names_still_resolve_to_the_default_model(self):
        default = ALIAS_TABLE["oll-default"]
        assert ALIAS_TABLE["oll-minimax"] == default
        assert ALIAS_TABLE["oll-dev"] == default
        from orchestra.providers.registry import AliasRegistry

        assert AliasRegistry.get("oll-default") is not None

    def test_oll_coder_exact(self):
        assert _canonical_alias("oll-coder") == "oll-coder"

    def test_oll_dev_exact(self):
        assert _canonical_alias("oll-dev") == "oll-dev"

    def test_unknown_defaults_to_cdx_deep(self):
        assert _canonical_alias("unknown") == "cdx-deep"
        assert _canonical_alias("xyz") == "cdx-deep"

    def test_all_aliases_in_table(self):
        for alias in ALIAS_TABLE:
            canonical = _canonical_alias(alias)
            assert canonical in ALIAS_TABLE


# ---------------------------------------------------------------------------
# _fallback_alias
# ---------------------------------------------------------------------------
class TestFallbackAlias:
    def test_cld_fallback(self):
        assert _fallback_alias("cld-deep") == "kimi-deep"
        assert _fallback_alias("cld-fast") == "kimi-deep"

    def test_oll_fallback(self):
        assert _fallback_alias("oll-minimax") == "cdx-deep"
        assert _fallback_alias("oll-fast") == "cdx-deep"

    def test_gmn_fallback(self):
        assert _fallback_alias("gmn-pro") == "cdx-deep"
        assert _fallback_alias("gmn-fast") == "cdx-deep"

    def test_cdx_fallback(self):
        assert _fallback_alias("cdx-deep") == "cld-deep"
        assert _fallback_alias("cdx-fast") == "cld-deep"

    def test_unknown_fallback(self):
        assert _fallback_alias("unknown") == "cld-deep"


# ---------------------------------------------------------------------------
# fallback_chain
# ---------------------------------------------------------------------------
class TestFallbackChain:
    def test_cdx_deep_chain(self):
        chain = fallback_chain("cdx-deep")
        assert chain == ["cdx-deep", "gmn-pro", "cld-deep", "cdx-fast"]

    def test_gmn_fast_chain(self):
        chain = fallback_chain("gmn-fast")
        assert chain == ["gmn-fast", "cdx-fast", "cld-fast", "cdx-deep"]

    def test_cld_deep_chain(self):
        chain = fallback_chain("cld-deep")
        assert chain == ["cld-deep", "kimi-deep", "cdx-deep", "gmn-pro"]

    def test_oll_fast_chain(self):
        chain = fallback_chain("oll-fast")
        assert chain == ["oll-fast", "oll-flash", "gmn-fast", "cdx-fast"]

    def test_oll_dev_chain(self):
        chain = fallback_chain("oll-dev")
        assert chain == ["oll-dev", "kimi-deep", "cld-fast", "cdx-deep"]

    def test_oll_coder_chain(self):
        chain = fallback_chain("oll-coder")
        assert chain == ["oll-coder", "kimi-code", "cld-fast"]

    def test_unknown_chain(self):
        chain = fallback_chain("random")
        # _canonical_alias('random') -> 'cdx-deep', fallback_chain('cdx-deep') returns 4 items
        assert len(chain) == 4
        assert chain[0] == "cdx-deep"

    def test_first_element_is_canonical(self):
        for alias in ALIAS_TABLE:
            chain = fallback_chain(alias)
            assert chain[0] == _canonical_alias(alias)


# ---------------------------------------------------------------------------
# resolve_with_fallback
# ---------------------------------------------------------------------------
class TestResolveWithFallback:
    @patch.object(ALIAS_TABLE["cdx-fast"][0], "is_available", return_value=True)
    @patch("orchestra.providers.fallback.provider_guard.can_use", return_value=True)
    def test_primary_available(self, _mock_guard, _mock_avail):
        provider, effort = resolve_with_fallback("cdx-fast")
        assert effort == "low"

    @patch.object(ALIAS_TABLE["cdx-fast"][0], "is_available", return_value=False)
    @patch.object(ALIAS_TABLE["gmn-fast"][0], "is_available", return_value=True)
    @patch("orchestra.providers.fallback.provider_guard.can_use", return_value=True)
    def test_fallback_on_attempt_1(self, _mock_guard, _mock_gmn, _mock_cdx):
        provider, effort = resolve_with_fallback("cdx-fast", attempt=1)
        assert effort == "flash"

    @patch.object(ALIAS_TABLE["cdx-fast"][0], "is_available", return_value=False)
    @patch.object(ALIAS_TABLE["gmn-fast"][0], "is_available", return_value=False)
    @patch.object(ALIAS_TABLE["cld-fast"][0], "is_available", return_value=True)
    @patch("orchestra.providers.fallback.provider_guard.can_use", return_value=True)
    def test_fallback_on_attempt_2(self, _mock_guard, _mock_cld, _mock_gmn, _mock_cdx):
        provider, effort = resolve_with_fallback("cdx-fast", attempt=2)
        assert effort == "sonnet"

    @patch.object(ALIAS_TABLE["cdx-fast"][0], "is_available", return_value=False)
    @patch.object(ALIAS_TABLE["gmn-fast"][0], "is_available", return_value=False)
    @patch.object(ALIAS_TABLE["cld-fast"][0], "is_available", return_value=False)
    @patch.object(ALIAS_TABLE["cdx-deep"][0], "is_available", return_value=False)
    @patch("orchestra.providers.fallback.provider_guard.can_use", return_value=True)
    def test_nothing_available_returns_start(self, _mock_guard, _mock_deep, _mock_cld, _mock_gmn, _mock_cdx):
        provider, effort = resolve_with_fallback("cdx-fast", attempt=0)
        assert effort == "low"  # cdx-fast

    @patch.object(ALIAS_TABLE["cdx-fast"][0], "is_available", return_value=False)
    @patch.object(ALIAS_TABLE["gmn-fast"][0], "is_available", return_value=False)
    @patch.object(ALIAS_TABLE["cld-fast"][0], "is_available", return_value=False)
    @patch.object(ALIAS_TABLE["cdx-deep"][0], "is_available", return_value=False)
    @patch("orchestra.providers.fallback.provider_guard.can_use", return_value=True)
    def test_attempt_exceeds_chain_length(self, _mock_guard, _mock_deep, _mock_cld, _mock_gmn, _mock_cdx):
        provider, effort = resolve_with_fallback("cdx-fast", attempt=10)
        # start_index = min(10, 3) = 3, chain[3] = 'cdx-deep', effort = 'xhigh'
        assert effort == "xhigh"

    @patch.object(ALIAS_TABLE["gmn-pro"][0], "is_available", return_value=True)
    @patch("orchestra.providers.fallback.provider_guard.can_use", return_value=True)
    def test_different_alias_chain(self, _mock_guard, _mock_avail):
        provider, effort = resolve_with_fallback("gmn-pro")
        assert effort == "pro"

    @patch("orchestra.providers.fallback.provider_guard.can_use", return_value=False)
    @patch.object(ALIAS_TABLE["cdx-fast"][0], "is_available", return_value=True)
    def test_provider_guard_blocks(self, _mock_avail, _mock_guard):
        # can_use returns False so no provider passes guard
        provider, effort = resolve_with_fallback("cdx-fast", attempt=0)
        assert effort == "low"  # falls to start_index which is cdx-fast, attempt<=0 calls resolve

    @patch.object(ALIAS_TABLE["oll-dev"][0], "is_available", return_value=True)
    @patch("orchestra.providers.fallback.provider_guard.can_use", return_value=True)
    def test_oll_dev_chain(self, _mock_guard, _mock_avail):
        provider, effort = resolve_with_fallback("oll-dev")
        assert effort == "gemma4:31b-cloud"


# ---------------------------------------------------------------------------
# [providers.<name>] enabled = false
# ---------------------------------------------------------------------------
class TestProviderEnabledFlag:
    def test_providers_are_enabled_unless_the_config_says_otherwise(self, monkeypatch):
        from orchestra import config

        monkeypatch.setattr(config, "section", lambda name: {"gemini": {"enabled": False}, "codex": {}})
        assert config.provider_enabled("gemini") is False
        assert config.provider_enabled("codex") is True
        assert config.provider_enabled("claude") is True

    def test_a_disabled_cli_provider_is_unavailable_even_when_installed(self, monkeypatch):
        from orchestra import config
        from orchestra.providers.codex import CodexProvider
        from orchestra.providers.gemini import GeminiProvider

        monkeypatch.setattr(config, "provider_enabled", lambda name: False)
        with patch("orchestra.providers.gemini.shutil.which", return_value="/bin/gemini"), \
                patch("orchestra.providers.codex.shutil.which", return_value="/bin/codex"):
            assert GeminiProvider().is_available() is False
            assert CodexProvider().is_available() is False

    def test_an_alias_of_a_disabled_provider_falls_through_its_chain(self, monkeypatch):
        gemini = ALIAS_TABLE["gmn-fast"][0]
        codex = ALIAS_TABLE["cdx-fast"][0]
        claude = ALIAS_TABLE["cld-fast"][0]
        monkeypatch.setattr(gemini, "is_available", lambda: False)
        monkeypatch.setattr(codex, "is_available", lambda: False)
        monkeypatch.setattr(claude, "is_available", lambda: True)
        with patch("orchestra.providers.fallback.provider_guard.can_use", return_value=True):
            provider, model = resolve_with_fallback("gmn-fast")
        assert provider is claude
        assert model == "sonnet"


# ---------------------------------------------------------------------------
# [providers.claude]: what the Claude CLI agent is allowed to do
# ---------------------------------------------------------------------------
class TestClaudeCommand:
    def _command(self, monkeypatch, section: dict, key: str = "claude") -> list[str]:
        from orchestra import config
        from orchestra.providers.claude import ClaudeProvider

        monkeypatch.setattr(config, "provider_config", lambda name: dict(section) if name == key else {})
        provider = ClaudeProvider() if key == "claude" else ClaudeProvider(config_key=key)
        monkeypatch.setattr(provider, "_binary", lambda: "claude")
        return provider.build_command("do it", "opus")

    def test_without_config_the_agent_may_do_anything_and_leaves_no_session(self, monkeypatch):
        assert self._command(monkeypatch, {}) == [
            "claude", "-p", "--no-session-persistence", "--dangerously-skip-permissions",
            "--output-format", "json", "--model", "opus", "do it",
        ]

    def test_sessions_can_be_kept(self, monkeypatch):
        assert "--no-session-persistence" not in self._command(monkeypatch, {"session_persistence": True})

    def test_permissions_can_be_kept(self, monkeypatch):
        command = self._command(monkeypatch, {"skip_permissions": False})
        assert "--dangerously-skip-permissions" not in command
        assert command[-1] == "do it"

    def test_tools_system_prompt_and_extra_arguments_are_passed(self, monkeypatch):
        command = self._command(monkeypatch, {
            "skip_permissions": False,
            "tools": ["Read", "Grep", "Glob"],
            "disallowed_tools": "Bash",
            "permission_mode": "plan",
            "append_system_prompt": "Answer in English.",
            "extra_args": ["--strict-mcp-config", "--setting-sources", "project"],
        })
        assert command[:2] == ["claude", "-p"]
        assert command[command.index("--tools") + 1] == "Read,Grep,Glob"
        assert command[command.index("--disallowed-tools") + 1] == "Bash"
        assert command[command.index("--permission-mode") + 1] == "plan"
        assert command[command.index("--append-system-prompt") + 1] == "Answer in English."
        assert "--strict-mcp-config" in command and command[command.index("--setting-sources") + 1] == "project"
        assert command[-4:] == ["json", "--model", "opus", "do it"]

    def test_a_second_profile_reads_its_own_section(self, monkeypatch):
        command = self._command(monkeypatch, {"skip_permissions": True}, key="claude_write")
        assert "--dangerously-skip-permissions" in command

    def test_the_writing_alias_uses_the_writing_profile(self):
        from orchestra.providers.claude import ClaudeProvider

        provider, model = ALIAS_TABLE["cld-write"]
        assert isinstance(provider, ClaudeProvider)
        assert provider.name == "claude"
        assert provider.config_key == "claude_write"
        assert model == "opus"
        assert ALIAS_TABLE["cld-deep"][0].config_key == "claude"



class TestCodexLogin:
    """An installed but logged-out Codex CLI cannot run anything: it is not available."""

    def _provider(self, monkeypatch, status_rc: int):
        from orchestra.providers import codex as codex_module

        monkeypatch.setattr(codex_module, "_login_checked_at", 0.0)
        monkeypatch.setattr(codex_module.shutil, "which", lambda name: "/bin/codex")
        monkeypatch.setattr(codex_module.config, "provider_enabled", lambda name: True)
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return type("Done", (), {"returncode": status_rc})()

        monkeypatch.setattr(codex_module.subprocess, "run", fake_run)
        return codex_module.CodexProvider(), calls

    def test_logged_out_is_unavailable(self, monkeypatch):
        provider, _ = self._provider(monkeypatch, status_rc=1)
        assert provider.is_available() is False

    def test_logged_in_is_available(self, monkeypatch):
        provider, _ = self._provider(monkeypatch, status_rc=0)
        assert provider.is_available() is True

    def test_the_login_is_asked_once_not_on_every_check(self, monkeypatch):
        provider, calls = self._provider(monkeypatch, status_rc=0)
        for _ in range(5):
            provider.is_available()
        assert len(calls) == 1


class TestRoutingSkipsWhatCannotRun:
    def test_the_router_picks_an_available_candidate(self, monkeypatch):
        from orchestra.router import classifier

        monkeypatch.setattr(classifier, "_can_run", lambda alias: alias == "cld-fast")
        monkeypatch.setattr(classifier, "suggest_best_alias", lambda task, aliases: aliases[0])

        assert classifier.select_alias_for_candidates("explain pooling", ["cdx-fast", "gmn-fast", "cld-fast"]) == "cld-fast"

    def test_with_nothing_available_the_first_candidate_is_kept(self, monkeypatch):
        from orchestra.router import classifier

        monkeypatch.setattr(classifier, "_can_run", lambda alias: False)
        monkeypatch.setattr(classifier, "suggest_best_alias", lambda task, aliases: aliases[0])

        assert classifier.select_alias_for_candidates("explain pooling", ["cdx-fast", "cld-fast"]) == "cdx-fast"

    def test_role_defaults_follow_the_configured_tiers_and_personas(self, monkeypatch):
        from orchestra import config

        monkeypatch.setattr(config, "get", lambda section, key, default=None: default)
        monkeypatch.setattr(config, "tier_config", lambda: {"light": {"agents": ["oll-fast", "oll-gemma"]}})
        monkeypatch.setattr(config, "personas_config", lambda: {"researcher": "kimi-deep", "synthesizer": "cld-deep"})

        assert config.supervisor_worker_alias() == "oll-fast"
        assert config.supervisor_planner_alias() == "kimi-deep"
        assert config.supervisor_synthesizer_alias() == "cld-deep"


class TestClaudeUsage:
    """The Claude CLI says what a call used and cost; that replaces the estimate."""

    REPLY = ('{"type": "result", "is_error": false, "result": "The answer.", "total_cost_usd": 0.0237,'
             ' "usage": {"input_tokens": 2, "cache_creation_input_tokens": 5706, "cache_read_input_tokens": 4591, "output_tokens": 4}}')

    def test_the_answer_and_the_usage_come_out_of_the_json(self):
        from orchestra.providers.claude import parse_result

        text, usage, failed = parse_result(self.REPLY)

        assert (text, failed) == ("The answer.", False)
        assert usage == {"prompt_tokens": 10299, "completion_tokens": 4, "cost_usd": 0.0237}

    def test_text_that_is_not_the_json_is_the_answer_as_it_is(self):
        from orchestra.providers.claude import parse_result

        assert parse_result("plain text answer") == ("plain text answer", None, False)

    def test_an_error_reported_in_the_json_fails_the_call(self):
        from orchestra.providers.claude import ClaudeProvider

        text, code = ClaudeProvider._answer('{"type": "result", "is_error": true, "result": "Credit balance is too low"}', 0)

        assert code == 1 and "Credit balance is too low" in text

    def test_a_call_leaves_its_usage_for_the_runner(self, monkeypatch):
        from orchestra.providers import base, claude

        monkeypatch.setattr(claude.config, "caching_config", lambda: {})
        monkeypatch.setattr(claude.config, "provider_config", lambda key: {})
        monkeypatch.setattr(base.BaseProvider, "run", lambda self, prompt, model, timeout=180, **kw: (self_reply, 0))
        global self_reply
        self_reply = self.REPLY
        provider = claude.ClaudeProvider()

        assert provider.run("q", "sonnet") == ("The answer.", 0)
        assert provider.last_usage()["cost_usd"] == 0.0237


class TestAnswersAreNotErrors:
    def test_a_real_answer_that_mentions_an_error_word_is_kept(self):
        from orchestra.engine.retries import _should_retry_output

        answer = "Use an overloaded method here. " * 30 + "The cached token expired, so refresh it."
        assert _should_retry_output(answer, "", 0) is False

    def test_a_short_error_message_with_exit_zero_is_retried(self):
        from orchestra.engine.retries import _should_retry_output

        assert _should_retry_output("Credit balance is too low", "", 0) is True
        assert _should_retry_output("", "", 0) is True

    def test_a_failed_call_is_judged_as_before(self):
        from orchestra.engine.retries import _should_retry_output

        assert _should_retry_output("x" * 2000 + " rate limit reached", "", 1) is True

    def test_kimi_gets_time_to_read_the_project(self, monkeypatch):
        from orchestra import config

        monkeypatch.setattr(config, "_cfg", {})
        assert config.timeout_for_alias("kimi-deep") == 600
        assert config.timeout_for_alias("kimi-code") == 300


class TestDiversityUsesAgentsThatCanRun:
    def test_a_fallback_that_cannot_run_is_skipped(self, monkeypatch):
        from orchestra.engine import adversarial

        monkeypatch.setattr(adversarial.config, "adversarial_enabled", lambda: True)
        monkeypatch.setattr(adversarial.config, "adversarial_diversity_enforced", lambda: True)
        monkeypatch.setattr(adversarial, "_provider_name",
                            lambda alias: {"cld": "claude", "kim": "kimi", "cdx": "claude", "gmn": "claude", "oll": "ollama"}[alias[:3]])
        monkeypatch.setattr(adversarial, "_can_run", lambda alias: alias in ("oll-coder", "cld-deep"))

        # kimi-deep, cdx-deep and gmn-pro cannot run here: the Ollama alias is the first real alternative
        assert adversarial.ensure_diversity(["cld-deep", "cld-fast"]) == ["cld-deep", "oll-coder"]


class TestAgentProcesses:
    def test_at_the_deadline_the_cli_and_what_it_started_are_ended(self, tmp_path, run_agent_command):
        import os
        import subprocess
        import sys
        import time

        import pytest

        run_command = run_agent_command

        marker = tmp_path / "child.pid"
        # a "CLI" that starts a helper (as Kimi starts MCP servers) and then never finishes
        script = (f"import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                  f"open({str(marker)!r}, 'w').write(str(p.pid)); time.sleep(60)")
        start = time.time()
        with pytest.raises(subprocess.TimeoutExpired):
            run_command([sys.executable, "-c", script], timeout=2)

        assert time.time() - start < 10
        helper = int(marker.read_text())
        time.sleep(0.3)
        with pytest.raises(ProcessLookupError):
            os.kill(helper, 0)

    def test_a_cli_has_no_terminal_of_its_own(self, run_agent_command):
        import sys

        run_command = run_agent_command

        code = "import os\ntry:\n    os.open('/dev/tty', os.O_RDWR); print('tty')\nexcept OSError:\n    print('no tty')"
        stdout, _, returncode = run_command([sys.executable, "-c", code], timeout=10)

        assert (stdout.strip(), returncode) == ("no tty", 0)


# ---------------------------------------------------------------------------
# require_known: a mistyped alias is an error, not another (expensive) agent
# ---------------------------------------------------------------------------
class TestRequireKnown:
    def test_a_table_alias_is_accepted(self):
        from orchestra.providers.fallback import require_known

        for alias in ALIAS_TABLE:
            require_known(alias)

    def test_a_mode_suffix_and_a_backend_reference_are_accepted(self):
        from orchestra.providers.fallback import require_known

        require_known("cdx-deep#calibrated")
        require_known("qwen2.5:7b@windows-gpu")   # the target is checked when it is resolved

    @pytest.mark.parametrize("typo", ["cdx-deeep", "cld-wrte", "openai-fst", "oll-fats#deep", "nonsense", ""])
    def test_a_name_that_is_not_an_alias_is_refused(self, typo):
        from orchestra.providers.fallback import UnknownAliasError, require_known

        with pytest.raises(UnknownAliasError):
            require_known(typo)

    def test_the_error_names_the_closest_aliases(self):
        from orchestra.providers.fallback import UnknownAliasError, require_known

        with pytest.raises(UnknownAliasError, match="cdx-deep"):
            require_known("cdx-deeep")


def test_run_ask_refuses_a_mistyped_alias_before_anything_runs():
    from orchestra import config
    from orchestra.engine.runners import run_ask
    from orchestra.providers.fallback import UnknownAliasError

    before = set(config.artifact_root().iterdir())
    with patch("orchestra.providers.fallback.resolve") as resolve:
        with pytest.raises(UnknownAliasError):
            run_ask("cdx-deeep", "hi")

    resolve.assert_not_called()
    assert set(config.artifact_root().iterdir()) == before


def test_run_dual_refuses_a_mistyped_agent_before_anything_runs():
    from orchestra.engine.runners import run_dual
    from orchestra.providers.fallback import UnknownAliasError

    with pytest.raises(UnknownAliasError):
        run_dual("hi", agents=["cdx-deep", "kimi-deeep"])


class TestTimeoutForAlias:
    """A mode or a node added to an alias does not change how long that agent may take."""

    def test_a_mode_or_a_node_keeps_the_aliass_timeout(self, monkeypatch):
        from orchestra import config

        monkeypatch.setattr(config, "_cfg", {"orchestra": {"timeout": 180}})
        assert config.timeout_for_alias("cdx-deep#calibrated") == 900
        assert config.timeout_for_alias("cdx-deep@windows") == 900
        assert config.timeout_for_alias("kimi-deep#deep@windows") == 600

    def test_the_configured_timeout_of_the_alias_applies_too(self, monkeypatch):
        from orchestra import config

        monkeypatch.setattr(config, "_cfg", {"timeouts": {"cld-deep": 1200, "cld-deep@windows": 60}})
        assert config.timeout_for_alias("cld-deep#deep") == 1200
        assert config.timeout_for_alias("cld-deep@windows") == 60        # the exact name wins

    def test_a_model_on_a_backend_gets_the_default(self, monkeypatch):
        from orchestra import config

        monkeypatch.setattr(config, "_cfg", {"orchestra": {"timeout": 240}})
        assert config.timeout_for_alias("qwen2.5:7b@windows-gpu") == 240

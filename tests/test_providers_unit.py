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
        assert any("critical" in line for line in result)
        assert any("expert" in line for line in result)

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
        # Unknown oll-* not in table maps to oll-minimax
        assert _canonical_alias("oll-random") == "oll-minimax"

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
        assert _fallback_alias("cld-deep") == "oll-minimax"
        assert _fallback_alias("cld-fast") == "oll-minimax"

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
        assert chain == ["cld-deep", "oll-minimax", "cdx-deep", "gmn-pro"]

    def test_oll_fast_chain(self):
        chain = fallback_chain("oll-fast")
        assert chain == ["oll-fast", "oll-minimax", "gmn-fast", "cdx-fast"]

    def test_oll_dev_chain(self):
        chain = fallback_chain("oll-dev")
        assert chain == ["oll-dev", "oll-minimax", "cld-fast", "cdx-deep"]

    def test_oll_coder_chain(self):
        chain = fallback_chain("oll-coder")
        assert chain == ["oll-coder", "oll-minimax", "cld-fast"]

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
        assert effort == "minimax-m2.7:cloud"

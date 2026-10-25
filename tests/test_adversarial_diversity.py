"""Tests for model-family diversity guard."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from orchestra.engine.adversarial import ensure_diversity


class TestEnsureDiversity:
    """Provider-family diversity enforcement."""

    def test_disabled_when_adversarial_off(self) -> None:
        with patch("orchestra.engine.adversarial.config.adversarial_enabled", return_value=False):
            assert ensure_diversity(["cdx-deep", "cdx-fast"]) == ["cdx-deep", "cdx-fast"]

    def test_disabled_when_diversity_off(self) -> None:
        with patch("orchestra.engine.adversarial.config.adversarial_enabled", return_value=True):
            with patch("orchestra.engine.adversarial.config.adversarial_diversity_enforced", return_value=False):
                assert ensure_diversity(["cdx-deep", "cdx-fast"]) == ["cdx-deep", "cdx-fast"]

    def test_no_change_when_already_diverse(self) -> None:
        with patch("orchestra.engine.adversarial.config.adversarial_enabled", return_value=True):
            with patch("orchestra.engine.adversarial.config.adversarial_diversity_enforced", return_value=True):
                with patch("orchestra.engine.adversarial._provider_name", side_effect=["codex", "gemini"]):
                    assert ensure_diversity(["cdx-deep", "gmn-pro"]) == ["cdx-deep", "gmn-pro"]

    def test_replaces_last_when_same_family(self) -> None:
        with patch("orchestra.engine.adversarial.config.adversarial_enabled", return_value=True):
            with patch("orchestra.engine.adversarial.config.adversarial_diversity_enforced", return_value=True):
                # Both aliases resolve to codex; fallback should replace last with gemini
                with patch("orchestra.engine.adversarial._provider_name", side_effect=["codex", "codex", "gemini"]), \
                        patch("orchestra.engine.adversarial._can_run", return_value=True):
                    result = ensure_diversity(["cdx-deep", "cdx-fast"])
                    assert result[0] == "cdx-deep"
                    assert result[1] == "gmn-pro"

    def test_single_alias_unchanged(self) -> None:
        with patch("orchestra.engine.adversarial.config.adversarial_enabled", return_value=True):
            with patch("orchestra.engine.adversarial.config.adversarial_diversity_enforced", return_value=True):
                assert ensure_diversity(["cdx-deep"]) == ["cdx-deep"]

    def test_empty_list_unchanged(self) -> None:
        with patch("orchestra.engine.adversarial.config.adversarial_enabled", return_value=True):
            with patch("orchestra.engine.adversarial.config.adversarial_diversity_enforced", return_value=True):
                assert ensure_diversity([]) == []

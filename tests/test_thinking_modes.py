from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock

from orchestra import thinking_modes as tm


def test_split_mode():
    assert tm.split_mode("oll-fast#deep") == ("oll-fast", "deep")
    assert tm.split_mode("oll-fast") == ("oll-fast", None)
    assert tm.split_mode("cdx-deep@windows#plan") == ("cdx-deep@windows", "plan")
    assert tm.split_mode("oll-fast#bogus") == ("oll-fast", None)  # unknown mode ignored


def test_tier_for_alias():
    assert tm.tier_for_alias("oll-fast") == "compact"
    assert tm.tier_for_alias("kimi-deep") == "full"
    assert tm.tier_for_alias("cdx-deep") == "full"
    assert tm.tier_for_alias("oll-fast@backend") == "compact"


def test_mode_prompt_loads_real_file():
    p = tm.mode_prompt("deep", "oll-fast")           # compact tier
    assert p and "step by step" in p.lower()
    pf = tm.mode_prompt("deep", "kimi-deep")           # full tier
    assert pf and "rigor over speed" in pf.lower()
    assert tm.mode_prompt("bogus", "oll-fast") is None


def test_concise_and_calibrated_modes_registered():
    assert tm.split_mode("oll-fast#concise") == ("oll-fast", "concise")
    assert tm.split_mode("cdx-deep#calibrated") == ("cdx-deep", "calibrated")


def test_concise_prompt_loads_real_file():
    p = tm.mode_prompt("concise", "oll-fast")          # compact tier
    assert p and "concise" in p.lower()
    pf = tm.mode_prompt("concise", "kimi-deep")         # full tier
    assert pf and "concise" in pf.lower()


def test_calibrated_prompt_loads_real_file():
    p = tm.mode_prompt("calibrated", "oll-fast")       # compact tier
    assert p and "i don't know" in p.lower()
    pf = tm.mode_prompt("calibrated", "cdx-deep")      # full tier
    assert pf and "i don't know" in pf.lower() and "sycophancy" in pf.lower()


def test_prepend_system():
    assert tm.prepend_system(None, "MODE") == "MODE"
    assert tm.prepend_system("sys", "MODE") == "MODE\n\nsys"
    assert tm.prepend_system([{"type": "text", "text": "s"}], "MODE")[0]["text"] == "MODE"



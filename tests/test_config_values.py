"""Unit tests for orchestra.utils.config_values.

Locks the unified superset behavior extracted from the previously diverged
copies in cli.py and service.py.  Every bullet in the reconciliation spec is
covered by at least one parametrized case.
"""
from __future__ import annotations

import json
import unittest
from unittest import TestCase

import pytest

from orchestra.utils.config_values import (
    flatten_toml_lines,
    parse_cli_value,
    toml_literal,
)


# ---------------------------------------------------------------------------
# parse_cli_value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True),
        ("True", True),
        ("false", False),
        ("False", False),
        ("3", 3),
        ("3.5", 3.5),
        ("0", 0),
        ("-1", -1),
        ("-1.5", -1.5),
        ("x", "x"),
        ("hello world", "hello world"),
        ("", ""),
    ],
)
def test_parse_cli_value_scalar_strings(raw, expected):
    assert parse_cli_value(raw) == expected


class TestParseCLIValueStrings(TestCase):
    """String inputs — must reproduce the old cli.py behaviour (scalar + list)."""

    def test_list_string_ints(self):
        assert parse_cli_value("[1, 2]") == [1, 2]

    def test_list_string_mixed(self):
        assert parse_cli_value("[1, a]") == [1, "a"]

    def test_list_string_empty(self):
        assert parse_cli_value("[]") == []

    def test_list_string_booleans(self):
        assert parse_cli_value("[true, false]") == [True, False]


class TestParseCLIValueNonStr(TestCase):
    """Non-string inputs — must pass through unchanged (service.py superset)."""

    def test_int_passthrough(self):
        assert parse_cli_value(42) == 42

    def test_float_passthrough(self):
        assert parse_cli_value(3.14) == 3.14

    def test_bool_true_passthrough(self):
        assert parse_cli_value(True) is True

    def test_bool_false_passthrough(self):
        assert parse_cli_value(False) is False

    def test_list_passthrough(self):
        result = parse_cli_value([1, "a"])
        assert result == [1, "a"]

    def test_dict_passthrough(self):
        d = {"k": 1}
        assert parse_cli_value(d) == d

    def test_zero_int_not_treated_as_false(self):
        # int 0 must pass through as 0 (not be confused with bool False)
        result = parse_cli_value(0)
        assert result == 0
        assert type(result) is int


# ---------------------------------------------------------------------------
# toml_literal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        (True, "true"),
        (False, "false"),
        (3, "3"),
        (3.5, "3.5"),
        (0, "0"),
        ("x", '"x"'),
        ('say "hi"', '"say \\"hi\\""'),
        (["a", 1], '["a", 1]'),
        ([], "[]"),
    ],
)
def test_toml_literal_basic(value, expected):
    assert toml_literal(value) == expected


def test_toml_literal_dict():
    """dict handling is the ONE intentional behaviour improvement over old cli.py.

    The old cli.py _toml_literal had no dict branch — a dict would fall through
    to the quoted str(value) path.  The new unified version uses json.dumps.
    """
    d = {"k": 1}
    result = toml_literal(d)
    assert result == json.dumps(d, ensure_ascii=False)
    # must be valid JSON string
    assert json.loads(result) == d


def test_toml_literal_nested_list():
    # Nested list of mixed types — booleans render as TOML true/false (no quotes)
    assert toml_literal([True, 2, "c"]) == '[true, 2, "c"]'


# ---------------------------------------------------------------------------
# flatten_toml_lines
# ---------------------------------------------------------------------------

class TestFlattenTomlLines(TestCase):
    """Identical behaviour to the pre-existing implementations in both files."""

    def test_flat_dict(self):
        lines = flatten_toml_lines({"a": 1, "b": "x"})
        assert "a = 1" in lines
        assert 'b = "x"' in lines

    def test_nested_dict(self):
        data = {"top": 1, "sub": {"key": "val"}}
        lines = flatten_toml_lines(data)
        assert "top = 1" in lines
        assert "[sub]" in lines
        assert 'key = "val"' in lines

    def test_prefix_produces_section_header(self):
        lines = flatten_toml_lines({"key": 42}, prefix=["a", "b"])
        assert lines[0] == "[a.b]"
        assert "key = 42" in lines

    def test_no_prefix_no_section_header(self):
        lines = flatten_toml_lines({"key": 1})
        assert not any(line.startswith("[") for line in lines)

    def test_blank_line_between_nested_sections(self):
        data = {"s1": {"a": 1}, "s2": {"b": 2}}
        lines = flatten_toml_lines(data)
        # blank line must separate the two section blocks
        assert "" in lines

    def test_scalars_before_nested(self):
        data = {"z_scalar": 9, "a_nested": {"x": 1}}
        lines = flatten_toml_lines(data)
        scalar_idx = next(i for i, l in enumerate(lines) if "z_scalar" in l)
        header_idx = next(i for i, l in enumerate(lines) if "[a_nested]" in l)
        assert scalar_idx < header_idx

    def test_empty_dict(self):
        assert flatten_toml_lines({}) == []

    def test_does_not_mutate_input(self):
        data = {"a": 1, "b": {"c": 2}}
        import copy
        original = copy.deepcopy(data)
        flatten_toml_lines(data)
        assert data == original


# ---------------------------------------------------------------------------
# Import sanity (smoke)
# ---------------------------------------------------------------------------

def test_module_importable():
    """Smoke test: module is importable and exports the three public names."""
    from orchestra.utils import config_values  # noqa: F401
    assert callable(config_values.parse_cli_value)
    assert callable(config_values.toml_literal)
    assert callable(config_values.flatten_toml_lines)


if __name__ == "__main__":
    unittest.main()

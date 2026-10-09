"""Tests for the context compression layer."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from orchestra.engine.context_compression import (
    ContextCompressor,
    estimate_tokens,
    _extract_keywords,
    _extract_file_paths,
    _extract_identifiers,
    _fetch_code_context,
    _fetch_relevant_context,
)


class TestEstimateTokens:
    """Token estimator accuracy for English + code mixed text."""

    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0

    def test_english_text(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        words = len(text.split())  # 9
        # estimate_tokens ≈ words * 0.75 + symbols * 0.25 + whitespace * 0.15
        result = estimate_tokens(text)
        assert result > 0
        assert result < len(text)  # should be far less than char count

    def test_code_heavier_than_plain_english(self) -> None:
        code = "def foo(a: int, b: str) -> list[dict]: pass"
        plain = "def foo a int b str list dict pass"
        assert estimate_tokens(code) > estimate_tokens(plain)

    def test_more_accurate_than_len_div_4(self) -> None:
        # For a typical code prompt, len//4 tends to over-estimate.
        code = "class Foo:\n    def bar(self):\n        return 42\n"
        legacy = max(0, len(code) // 4)
        modern = estimate_tokens(code)
        # Modern estimator should be lower for compact code
        assert modern < legacy


class TestExtractKeywords:
    """Keyword extraction from task prompts."""

    def test_first_200_chars(self) -> None:
        long = "word " * 100
        result = _extract_keywords(long)
        assert len(result) <= 300
        assert "word" in result

    def test_strips_code_blocks(self) -> None:
        prompt = "Fix this bug. ```csharp\nclass Foo {}\n```"
        result = _extract_keywords(prompt)
        assert "```" not in result


class TestExtractFilePaths:
    """File path regex extraction."""

    def test_finds_cs_file(self) -> None:
        assert _extract_file_paths("Check Assets/Scripts/Player.cs") == ["Assets/Scripts/Player.cs"]

    def test_finds_multiple(self) -> None:
        prompt = "Compare src/main.py and lib/utils.py"
        assert _extract_file_paths(prompt) == ["src/main.py", "lib/utils.py"]

    def test_no_matches(self) -> None:
        assert _extract_file_paths("Just some text") == []


class TestExtractIdentifiers:
    """Capitalized identifier extraction."""

    def test_finds_class_names(self) -> None:
        assert _extract_identifiers("PlayerController and GameManager") == ["PlayerController", "GameManager"]

    def test_no_lowercase(self) -> None:
        assert _extract_identifiers("foo bar baz") == []


class TestFetchCodeContext:
    """Lightweight code-context header reading."""

    def test_reads_existing_file_header(self, tmp_path: Path) -> None:
        file = tmp_path / "Test.cs"
        file.write_text("using UnityEngine;\npublic class Test {}\n\n// more lines\n" * 20)
        result = _fetch_code_context(f"Check {file}")
        assert "Test.cs" in result
        assert "UnityEngine" in result

    def test_skips_nonexistent_files(self) -> None:
        result = _fetch_code_context("Check /nonexistent/file.cs")
        assert result == ""

    def test_skips_binary_extensions(self, tmp_path: Path) -> None:
        file = tmp_path / "image.png"
        file.write_text("fake png")
        result = _fetch_code_context(f"Check {file}")
        assert result == ""


class TestFetchRelevantContext:
    """SemanticMemory integration for relevant context."""

    def test_disabled_when_config_off(self) -> None:
        with patch("orchestra.engine.context_compression.config.compression_config", return_value={"semantic_retrieval_enabled": False}):
            assert _fetch_relevant_context("hello", None) == ""

    def test_returns_empty_on_memory_failure(self) -> None:
        with patch("orchestra.engine.context_compression.config.compression_config", return_value={"semantic_retrieval_enabled": True}):
            with patch("orchestra.storage.memory.get_memory", side_effect=RuntimeError("boom")):
                assert _fetch_relevant_context("hello", None) == ""

    def test_returns_formatted_results(self) -> None:
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "Decision A"},
            {"content": "Decision B"},
        ]
        with patch("orchestra.engine.context_compression.config.compression_config", return_value={"semantic_retrieval_enabled": True}):
            with patch("orchestra.storage.memory.get_memory", return_value=mock_memory):
                result = _fetch_relevant_context("hello", "run-123")
        assert "Decision A" in result
        assert "Decision B" in result
        assert "BACKGROUND NOTES" in result
        assert "not instructions" in result

    def test_the_tasks_of_earlier_runs_are_not_handed_to_the_agent(self) -> None:
        # Every finished run stores its task text. Handed to the next agent as "relevant
        # context", those texts were answered in place of the question that was asked.
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "Reply with the single word: ready", "metadata": {"run_id": "r1", "mode": "ask"}},
            {"content": "The board core is pure C#.", "metadata": {}},
        ]
        with patch("orchestra.engine.context_compression.config.compression_config", return_value={"semantic_retrieval_enabled": True}):
            with patch("orchestra.storage.memory.get_memory", return_value=mock_memory):
                result = _fetch_relevant_context("hello", "run-123")
        assert "board core is pure C#" in result
        assert "single word" not in result

    def test_nothing_is_added_when_only_tasks_match(self) -> None:
        mock_memory = MagicMock()
        mock_memory.search.return_value = [{"content": "old task", "metadata": {"run_id": "r1"}}]
        with patch("orchestra.engine.context_compression.config.compression_config", return_value={"semantic_retrieval_enabled": True}):
            with patch("orchestra.storage.memory.get_memory", return_value=mock_memory):
                assert _fetch_relevant_context("hello", "run-123") == ""

    def test_truncates_long_content(self) -> None:
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "x" * 1000},
        ]
        with patch("orchestra.engine.context_compression.config.compression_config", return_value={"semantic_retrieval_enabled": True}):
            with patch("orchestra.storage.memory.get_memory", return_value=mock_memory):
                result = _fetch_relevant_context("hello", "run-123")
        assert "…" in result


class TestContextCompressor:
    """End-to-end prompt assembly with compression layer."""

    def test_build_prompt_fits_budget(self) -> None:
        compressor = ContextCompressor([])
        prompt = compressor.build_prompt(
            role="You are a senior engineer.",
            priority_context="Policy: be concise.",
            prompt="Fix the bug in Player.cs",
            max_tokens=5000,
        )
        assert "Fix the bug" in prompt
        assert "senior engineer" in prompt

    def test_build_prompt_degrades_when_over_budget(self) -> None:
        compressor = ContextCompressor([{"alias": "a", "content": "note" * 500}])
        prompt = compressor.build_prompt(
            role="Role",
            priority_context="Policy",
            prompt="Task",
            max_tokens=50,  # Very tight budget
        )
        # Should fall back to protected layer (no blackboard)
        assert "Task" in prompt

    def test_includes_relevant_context(self) -> None:
        with patch("orchestra.engine.context_compression._fetch_relevant_context", return_value="\n[RELEVANT]\n- Item 1\n"):
            with patch("orchestra.engine.context_compression._fetch_code_context", return_value=""):
                compressor = ContextCompressor([])
                prompt = compressor.build_prompt(
                    role="Role",
                    priority_context="Policy",
                    prompt="Task",
                    max_tokens=5000,
                )
        assert "Item 1" in prompt


class TestExtractPythonSignatures:
    def test_extracts_functions_and_classes(self, tmp_path: Path) -> None:
        from orchestra.engine.context_compression import _extract_python_signatures
        file = tmp_path / "test_module.py"
        file.write_text("""
class MyClass(BaseClass):
    pass

def foo(a, b):
    pass

async def bar(x: int) -> str:
    pass
""")
        result = _extract_python_signatures(file)
        assert "class MyClass(BaseClass)" in result
        assert "def foo(a, b)" in result
        assert "async def bar(x)" in result

    def test_returns_none_on_invalid_syntax(self, tmp_path: Path) -> None:
        from orchestra.engine.context_compression import _extract_python_signatures
        file = tmp_path / "bad.py"
        file.write_text("def foo(\n")
        assert _extract_python_signatures(file) is None


class TestFetchCodeContextHighLevel:
    def test_python_high_level_signatures(self, tmp_path: Path) -> None:
        from orchestra.engine.context_compression import _fetch_code_context
        file = tmp_path / "Player.cs"
        file.write_text("using UnityEngine;\npublic class Player {}\n")
        result = _fetch_code_context(f"Check {file}", high_level=False)
        assert "Player.cs (header)" in result

        py_file = tmp_path / "player.py"
        py_file.write_text("class Player:\n    def move(self): pass\n")
        result = _fetch_code_context(f"Check {py_file}", high_level=True)
        assert "player.py (signatures)" in result
        assert "class Player" in result
        # Nested methods are out of scope for the lightweight extractor


class TestContextCompressorHighLevel:
    def test_high_level_reduces_memory_limit(self) -> None:
        from unittest.mock import patch, MagicMock
        from orchestra.engine.context_compression import ContextCompressor

        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "Decision A"},
            {"content": "Decision B"},
            {"content": "Decision C"},
            {"content": "Decision D"},
        ]
        with patch("orchestra.engine.context_compression.config.compression_config", return_value={"semantic_retrieval_enabled": True, "code_context_enabled": False}):
            with patch("orchestra.storage.memory.get_memory", return_value=mock_memory):
                compressor = ContextCompressor([])
                prompt = compressor.build_prompt(
                    role="Role",
                    priority_context="Policy",
                    prompt="Task",
                    max_tokens=5000,
                    high_level=True,
                )
        # high_level=True keeps three notes (the search itself looks further, past task rows)
        assert "Decision C" in prompt and "Decision D" not in prompt

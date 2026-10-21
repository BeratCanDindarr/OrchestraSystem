"""AST regression test: providers must use self._last_usage, not self.last_usage.

The base class (orchestra/providers/base.py) stores usage in `_last_usage` and
exposes it via the `last_usage()` method.  Assigning to the public name
`self.last_usage` silently loses cost/token tracking because nothing reads it.

This test walks every orchestra/providers/*.py source file via `ast` and asserts
that no assignment to `self.last_usage` exists.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

PROVIDERS_DIR = pathlib.Path(__file__).parent.parent / "orchestra" / "providers"


def _find_bad_assignments(source: str, filename: str) -> list[str]:
    """Return list of 'file:line' strings for every self.last_usage = ... found."""
    tree = ast.parse(source, filename=str(filename))
    violations: list[str] = []
    for node in ast.walk(tree):
        # Both plain Assign (a = b) and augmented assign (a += b) are checked.
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == "last_usage"
                ):
                    violations.append(f"{filename}:{node.lineno}")
    return violations


def collect_violations() -> list[str]:
    all_violations: list[str] = []
    for py_file in sorted(PROVIDERS_DIR.glob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        all_violations.extend(_find_bad_assignments(source, py_file.name))
    return all_violations


def test_no_public_last_usage_assignment() -> None:
    """Every provider must assign self._last_usage, never self.last_usage."""
    violations = collect_violations()
    assert violations == [], (
        "Found assignment(s) to self.last_usage (public, unread) instead of "
        "self._last_usage in the following locations:\n"
        + "\n".join(f"  {v}" for v in violations)
        + "\n\nChange 'self.last_usage = ...' to 'self._last_usage = ...' in each file."
    )


def test_base_provider_last_usage_method_returns_private_attr() -> None:
    """Sanity-check: base.py last_usage() returns self._last_usage."""
    base_file = PROVIDERS_DIR / "base.py"
    source = base_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename="base.py")

    # Look for: return self._last_usage inside a function named last_usage
    found_method = False
    found_return = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "last_usage":
            found_method = True
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Return)
                    and isinstance(child.value, ast.Attribute)
                    and isinstance(child.value.value, ast.Name)
                    and child.value.value.id == "self"
                    and child.value.attr == "_last_usage"
                ):
                    found_return = True
    assert found_method, "base.py: last_usage() method not found"
    assert found_return, "base.py: last_usage() does not return self._last_usage"


# ── usage belongs to the call, not to the provider object ───────────────────
# One provider object serves every agent that uses it, several at once.

import asyncio  # noqa: E402

from orchestra.providers.base import BaseProvider  # noqa: E402


class _Metered(BaseProvider):
    """Reports as many completion tokens as its prompt says; "fail" reports nothing."""
    name = "metered"

    def is_available(self) -> bool:
        return True

    def model_label(self, effort_or_model: str) -> str:
        return "metered/1"

    def build_command(self, prompt: str, effort_or_model: str) -> list[str]:
        return ["true"]

    def run(self, prompt, effort_or_model, timeout=180, **kwargs):
        if prompt == "fail":
            return "[ERROR] no", 1
        self._last_usage = {"completion_tokens": int(prompt)}
        return "ok", 0


@pytest.mark.asyncio
async def test_two_agents_on_one_provider_each_read_their_own_usage():
    provider = _Metered()

    async def agent(tokens: int):
        provider._last_usage = {"completion_tokens": tokens}
        await asyncio.sleep(0.01)                 # the other agent writes in the meantime
        return provider.last_usage()

    assert await asyncio.gather(agent(1), agent(2)) == [{"completion_tokens": 1}, {"completion_tokens": 2}]


@pytest.mark.asyncio
async def test_usage_noted_in_the_worker_thread_reaches_the_task_that_awaited_the_call():
    provider = _Metered()

    await provider.async_run("7", "m")

    assert provider.last_usage() == {"completion_tokens": 7}


@pytest.mark.asyncio
async def test_a_failed_call_does_not_report_the_usage_of_the_call_before_it():
    from orchestra.engine import retries

    provider = _Metered()
    policy = retries.RetryPolicy(max_attempts=1, backoff_seconds=[0.0], retry_on_exit_codes=[])

    await retries.run_with_retry_async(provider, "m", "5", policy)
    assert provider.last_usage() == {"completion_tokens": 5}

    await retries.run_with_retry_async(provider, "m", "fail", policy)
    assert provider.last_usage() is None



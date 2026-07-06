"""Lock-discipline regression test for orchestra/engine/parallel.py.

A ``threading.Lock`` must NEVER be held across an ``await``: while the
coroutine is suspended at the await, the event loop can schedule another
agent coroutine, and when that coroutine reaches its own ``with lock:`` the
non-async ``threading.Lock.acquire()`` blocks the whole event-loop OS thread,
freezing the loop.

This AST-based test parses parallel.py and asserts that no ``await`` appears
anywhere inside a ``with``/``async with`` block whose context expression
references a name/attribute containing "lock" (case-insensitive).
"""

import ast
from pathlib import Path

PARALLEL_PATH = Path(__file__).resolve().parent.parent / "orchestra" / "engine" / "parallel.py"


def _mentions_lock(node: ast.AST) -> bool:
    """True if any Name/Attribute in the subtree contains 'lock' (case-insensitive)."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and "lock" in sub.id.lower():
            return True
        if isinstance(sub, ast.Attribute) and "lock" in sub.attr.lower():
            return True
    return False


def _await_lines(node: ast.AST) -> list[int]:
    """Line numbers of every ``ast.Await`` descendant of ``node``."""
    return sorted(
        sub.lineno
        for sub in ast.walk(node)
        if isinstance(sub, ast.Await)
    )


def test_no_await_held_across_threading_lock() -> None:
    source = PARALLEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(PARALLEL_PATH))

    offenders: list[tuple[int, list[int]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        if not any(_mentions_lock(item.context_expr) for item in node.items):
            continue
        awaits = _await_lines(node)
        if awaits:
            offenders.append((node.lineno, awaits))

    assert not offenders, (
        "await found inside a lock-holding `with` block in parallel.py "
        "(holding threading.Lock across await freezes the event loop): "
        + "; ".join(
            f"with-block at line {with_line} contains await(s) at line(s) {awaits}"
            for with_line, awaits in offenders
        )
    )

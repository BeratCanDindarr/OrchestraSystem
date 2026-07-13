"""Interactive REPL loop with injectable IO for testing."""
from __future__ import annotations

from orchestra.chat import commands


def run_repl(session, input_fn=input, output_fn=print) -> None:
    while True:
        try:
            line = input_fn("you> ")
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return
        if line is None:
            return
        if not line.strip():
            continue
        if commands.is_command(line):
            if commands.handle_command(session, line, output_fn) == "exit":
                return
            continue
        output_fn(f"{session.alias}> ", end="")
        try:
            session.send(line, on_chunk=lambda t: output_fn(t, end=""))
            output_fn("")
        except KeyboardInterrupt:
            output_fn("\n[cancelled]")
        except Exception as exc:
            output_fn(f"\n[error] {exc}")

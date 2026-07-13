"""Slash-command dispatch for the chat REPL."""
from __future__ import annotations

_HELP = (
    "/model <alias>  switch model (history kept)\n"
    "/clear          clear conversation history\n"
    "/new            start a new session\n"
    "/models         list available aliases\n"
    "/help           this help\n"
    "/exit           quit"
)


def is_command(line: str) -> bool:
    return line.strip().startswith("/")


def handle_command(session, line: str, output_fn) -> str:
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0]
    arg = parts[1].strip() if len(parts) > 1 else ""
    if cmd == "/exit":
        return "exit"
    if cmd == "/help":
        output_fn(_HELP)
    elif cmd == "/clear":
        session.history.clear()
        output_fn("history cleared")
    elif cmd == "/model":
        if not arg:
            output_fn("usage: /model <alias>")
        else:
            try:
                session.switch_model(arg)
                output_fn(f"switched to {arg}")
            except Exception as exc:
                output_fn(f"cannot switch to {arg!r}: {exc}")
    elif cmd == "/new":
        from orchestra.chat import store
        try:
            session.session_id = store.new_session_id(session.alias)
            store.write_meta(session.session_id, session.alias, session.provider.model_label(session.effort))
            session.history.clear()
            output_fn(f"new session {session.session_id}")
        except Exception as exc:
            output_fn(f"cannot start a new session: {exc}")
    elif cmd == "/models":
        from orchestra.providers.fallback import available_aliases
        for a in available_aliases():
            output_fn(a)
    else:
        output_fn(f"unknown command {cmd!r}; type /help")
    return "continue"

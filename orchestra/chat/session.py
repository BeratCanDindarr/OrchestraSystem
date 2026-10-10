"""ChatSession: one conversation with a resolved provider, streamed and persisted."""
from __future__ import annotations

from orchestra import config
from orchestra.chat import store
from orchestra.chat.history import render_history
from orchestra.providers.fallback import resolve


class ChatSession:
    def __init__(self, session_id: str, alias: str, provider, effort: str, history: list[dict]):
        self.session_id = session_id
        self.alias = alias
        self.provider = provider
        self.effort = effort
        self.history = history

    @classmethod
    def start(cls, alias: str, session_id: str | None = None) -> "ChatSession":
        provider, effort = resolve(alias)
        if session_id is None:
            session_id = store.new_session_id(alias)
            store.write_meta(session_id, alias, provider.model_label(effort))
        return cls(session_id, alias, provider, effort, [])

    @classmethod
    def resume(cls, session_id: str) -> "ChatSession":
        meta, messages = store.load_session(session_id)
        alias = meta.get("alias") or config.default_light_alias()
        provider, effort = resolve(alias)
        return cls(session_id, alias, provider, effort, messages)

    def switch_model(self, alias: str) -> None:
        self.provider, self.effort = resolve(alias)
        self.alias = alias

    def send(self, message: str, on_chunk=None) -> str:
        self.history.append({"role": "user", "content": message})
        store.append_message(self.session_id, "user", message)
        prompt = render_history(self.history)
        reply = self._generate(prompt, on_chunk)
        self.history.append({"role": "assistant", "content": reply})
        store.append_message(self.session_id, "assistant", reply)
        return reply

    def _generate(self, prompt: str, on_chunk) -> str:
        try:
            parts: list[str] = []
            for chunk in self.provider.stream(prompt, self.effort):
                parts.append(chunk)
                if on_chunk:
                    on_chunk(chunk)
            return "".join(parts)
        except NotImplementedError:
            reply, _ = self.provider.run(prompt, self.effort)
            if on_chunk:
                on_chunk(reply)
            return reply

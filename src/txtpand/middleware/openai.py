"""Transparent proxy for OpenAI client that auto-expands user messages."""

from __future__ import annotations

from typing import Any

from txtpand.core.expander import Expander


def wrap_openai(client: Any, expander: Expander | None = None) -> Any:
    """Wrap an OpenAI client to auto-expand user messages.

    Usage:
        import openai
        import txtpand

        client = txtpand.wrap_openai(openai.OpenAI())
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "cn y hel me"}],
        )
        # User message is expanded to "can you help me" before sending

    Args:
        client: An openai.OpenAI() instance.
        expander: Custom Expander instance (uses default if None).

    Returns:
        Wrapped client.
    """
    exp = expander or Expander()
    return _OpenAIProxy(client, exp)


class _OpenAIProxy:
    """Proxy that intercepts chat.completions.create calls."""

    def __init__(self, client: Any, expander: Expander) -> None:
        self._client = client
        self._expander = expander
        self.chat = _ChatProxy(client.chat, expander)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _ChatProxy:
    def __init__(self, chat: Any, expander: Expander) -> None:
        self._chat = chat
        self._expander = expander
        self.completions = _CompletionsProxy(chat.completions, expander)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _CompletionsProxy:
    def __init__(self, completions: Any, expander: Expander) -> None:
        self._completions = completions
        self._expander = expander

    def create(self, **kwargs: Any) -> Any:
        if "messages" in kwargs:
            kwargs["messages"] = _expand_messages(
                kwargs["messages"], self._expander
            )
        return self._completions.create(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


def _expand_messages(messages: list[dict[str, Any]], expander: Expander) -> list[dict[str, Any]]:
    """Expand user message content."""
    expanded = []
    for msg in messages:
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            new_msg = dict(msg)
            new_msg["content"] = expander.expand(msg["content"])
            expanded.append(new_msg)
        else:
            expanded.append(msg)
    return expanded

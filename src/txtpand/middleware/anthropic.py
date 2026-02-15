"""Transparent proxy for Anthropic client that auto-expands user messages."""

from __future__ import annotations

from typing import Any

from txtpand.core.expander import Expander


def wrap_anthropic(client: Any, expander: Expander | None = None) -> Any:
    """Wrap an Anthropic client to auto-expand user messages.

    Usage:
        import anthropic
        import txtpand

        client = txtpand.wrap_anthropic(anthropic.Anthropic())
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "cn y hel me"}],
            max_tokens=1024,
        )
        # User message is expanded to "can you help me" before sending

    Args:
        client: An anthropic.Anthropic() instance.
        expander: Custom Expander instance (uses default if None).

    Returns:
        Wrapped client.
    """
    exp = expander or Expander()
    return _AnthropicProxy(client, exp)


class _AnthropicProxy:
    """Proxy that intercepts messages.create calls."""

    def __init__(self, client: Any, expander: Expander) -> None:
        self._client = client
        self._expander = expander
        self.messages = _MessagesProxy(client.messages, expander)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _MessagesProxy:
    def __init__(self, messages: Any, expander: Expander) -> None:
        self._messages = messages
        self._expander = expander

    def create(self, **kwargs: Any) -> Any:
        if "messages" in kwargs:
            kwargs["messages"] = _expand_messages(
                kwargs["messages"], self._expander
            )
        return self._messages.create(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


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

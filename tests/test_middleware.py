"""Tests for middleware proxies (mocked SDK clients)."""

from unittest.mock import MagicMock

from txtpand.core.expander import Expander
from txtpand.middleware.anthropic import wrap_anthropic
from txtpand.middleware.openai import _expand_messages, wrap_openai


class TestOpenAIMiddleware:
    def test_expand_messages_helper(self):
        expander = Expander()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hel me"},
        ]
        expanded = _expand_messages(messages, expander)

        # System message should be unchanged
        assert expanded[0]["content"] == "You are helpful."
        # User message should be expanded
        assert "help" in expanded[1]["content"] or "hello" in expanded[1]["content"]
        assert "me" in expanded[1]["content"]

    def test_wrap_openai_proxy(self):
        # Create mock OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Sure, I can help!"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        wrapped = wrap_openai(mock_client)

        wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hel me"}],
        )

        # Verify the mock was called
        mock_client.chat.completions.create.assert_called_once()

        # Verify messages were expanded
        call_kwargs = mock_client.chat.completions.create.call_args
        sent_messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert sent_messages is not None
        user_msg = [m for m in sent_messages if m["role"] == "user"][0]
        # Should be expanded from "hel me"
        assert user_msg["content"] != "hel me"

    def test_system_message_unchanged(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock()

        wrapped = wrap_openai(mock_client)
        wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "hel me"},
                {"role": "user", "content": "hello"},
            ],
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        sent_messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        system_msg = [m for m in sent_messages if m["role"] == "system"][0]
        # System messages should NOT be expanded
        assert system_msg["content"] == "hel me"

    def test_passthrough_attributes(self):
        mock_client = MagicMock()
        mock_client.models.list.return_value = ["gpt-4o"]

        wrapped = wrap_openai(mock_client)
        result = wrapped.models.list()
        assert result == ["gpt-4o"]


class TestAnthropicMiddleware:
    def test_wrap_anthropic_proxy(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Sure!"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        wrapped = wrap_anthropic(mock_client)

        wrapped.messages.create(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "hel me"}],
            max_tokens=1024,
        )

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args
        sent_messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        user_msg = [m for m in sent_messages if m["role"] == "user"][0]
        assert user_msg["content"] != "hel me"

    def test_passthrough_attributes(self):
        mock_client = MagicMock()
        mock_client.count_tokens.return_value = 42

        wrapped = wrap_anthropic(mock_client)
        assert wrapped.count_tokens() == 42

"""LLM provider adapters for OpenAI and Anthropic."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    def complete(self, system: str, user: str, timeout: float) -> str:
        """Send a chat completion request and return the text response."""
        ...


class OpenAIProvider(LLMProvider):
    """OpenAI API provider adapter."""

    def __init__(
        self,
        client: object | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self._client = client
        self._model = model

    def _get_client(self) -> object:
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI()
            except ImportError as e:
                raise ImportError(
                    "openai package required. Install with: pip install txtpand[openai]"
                ) from e
        return self._client

    def complete(self, system: str, user: str, timeout: float) -> str:
        client = self._get_client()
        response = client.chat.completions.create(  # type: ignore[union-attr]
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=200,
            temperature=0.0,
            timeout=timeout,
        )
        return response.choices[0].message.content or ""  # type: ignore[union-attr]


class AnthropicProvider(LLMProvider):
    """Anthropic API provider adapter."""

    def __init__(
        self,
        client: object | None = None,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._client = client
        self._model = model

    def _get_client(self) -> object:
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic()
            except ImportError as e:
                raise ImportError(
                    "anthropic package required. Install with: pip install txtpand[anthropic]"
                ) from e
        return self._client

    def complete(self, system: str, user: str, timeout: float) -> str:
        client = self._get_client()
        response = client.messages.create(  # type: ignore[union-attr]
            model=self._model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=200,
            temperature=0.0,
        )
        return response.content[0].text  # type: ignore[union-attr]

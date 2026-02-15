"""Tests for LLM fallback (mocked)."""

from unittest.mock import MagicMock

import pytest

from txtpand.config import TxtpandConfig
from txtpand.core.expander import Expander
from txtpand.exceptions import LLMFallbackError
from txtpand.llm.fallback import LLMFallback
from txtpand.llm.prompt import (
    DISAMBIGUATION_SYSTEM,
    POLISH_SYSTEM,
    build_disambiguation_prompt,
    build_polish_prompt,
)
from txtpand.llm.providers import LLMProvider
from txtpand.types import MatchTier, ScoredCandidate, TokenResult


class MockProvider(LLMProvider):
    def __init__(self, response: str):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, timeout: float) -> str:
        self.calls.append((system, user))
        return self.response


class TestLLMFallback:
    def test_resolve_single_token(self):
        provider = MockProvider("work")
        fallback = LLMFallback(provider)

        tr = TokenResult(
            original="wo",
            expanded="would",
            confidence=0.7,
            candidates=[
                ScoredCandidate("would", 0.7, MatchTier.PREFIX),
                ScoredCandidate("work", 0.65, MatchTier.PREFIX),
            ],
            ambiguous=True,
        )

        result = fallback.resolve([(2, tr)], "can you would on this")
        assert result == {2: "work"}

    def test_resolve_multiple_tokens(self):
        provider = MockProvider("work\nfew")
        fallback = LLMFallback(provider)

        tr1 = TokenResult(
            original="wo",
            expanded="would",
            confidence=0.7,
            candidates=[
                ScoredCandidate("would", 0.7, MatchTier.PREFIX),
                ScoredCandidate("work", 0.65, MatchTier.PREFIX),
            ],
            ambiguous=True,
        )
        tr2 = TokenResult(
            original="fe",
            expanded="feel",
            confidence=0.6,
            candidates=[
                ScoredCandidate("feel", 0.6, MatchTier.PREFIX),
                ScoredCandidate("few", 0.55, MatchTier.PREFIX),
            ],
            ambiguous=True,
        )

        result = fallback.resolve([(2, tr1), (5, tr2)], "context")
        assert result == {2: "work", 5: "few"}

    def test_resolve_empty(self):
        provider = MockProvider("")
        fallback = LLMFallback(provider)
        assert fallback.resolve([], "context") == {}

    def test_resolve_provider_failure(self):
        provider = MagicMock(spec=LLMProvider)
        provider.complete.side_effect = RuntimeError("API down")
        fallback = LLMFallback(provider)

        tr = TokenResult(
            original="wo",
            expanded="would",
            confidence=0.7,
            candidates=[],
            ambiguous=True,
        )

        with pytest.raises(LLMFallbackError):
            fallback.resolve([(0, tr)], "context")

    def test_provider_called_with_correct_params(self):
        provider = MockProvider("test")
        fallback = LLMFallback(provider, timeout=5.0)

        tr = TokenResult(
            original="te",
            expanded="tell",
            confidence=0.6,
            candidates=[
                ScoredCandidate("tell", 0.6, MatchTier.PREFIX),
                ScoredCandidate("test", 0.55, MatchTier.PREFIX),
            ],
            ambiguous=True,
        )

        fallback.resolve([(0, tr)], "some context")
        assert len(provider.calls) == 1
        system, user = provider.calls[0]
        assert "text expansion" in system.lower()
        assert "te" in user


class TestPolish:
    def test_polish_corrects_sentence(self):
        provider = MockProvider("can you help me work on a few things")
        fallback = LLMFallback(provider)

        result = fallback.polish(
            "cn y hel me wo on a fe thin",
            "can you help me work on a few thin",
        )
        assert result == "can you help me work on a few things"

    def test_polish_strips_quotes(self):
        provider = MockProvider('"can you help me"')
        fallback = LLMFallback(provider)
        result = fallback.polish("cn y hel me", "can you help me")
        assert result == "can you help me"

    def test_polish_returns_original_on_failure(self):
        provider = MagicMock(spec=LLMProvider)
        provider.complete.side_effect = RuntimeError("timeout")
        fallback = LLMFallback(provider)

        with pytest.raises(LLMFallbackError):
            fallback.polish("cn y hel me", "can you help me")

    def test_polish_rejects_wildly_different_response(self):
        provider = MockProvider("a" * 1000)
        fallback = LLMFallback(provider)
        result = fallback.polish("hi", "hello")
        assert result == "hello"  # falls back to original

    def test_polish_called_with_correct_prompts(self):
        provider = MockProvider("can you help me")
        fallback = LLMFallback(provider)
        fallback.polish("cn y hel me", "can you help me")

        assert len(provider.calls) == 1
        system, user = provider.calls[0]
        assert "fix" in system.lower() or "correct" in system.lower()
        assert "cn y hel me" in user
        assert "can you help me" in user

    def test_expander_with_llm_polish(self):
        """Integration: expander uses LLM polish when enabled."""
        provider = MockProvider("can you help me work on a few things")
        fallback = LLMFallback(provider)
        config = TxtpandConfig(llm_enabled=True)
        expander = Expander(config=config, llm_fallback=fallback)

        report = expander.expand_detailed("cn y hel me wo on a fe thin")
        assert report.llm_used is True
        assert "things" in report.expanded

    def test_expander_llm_failure_graceful(self):
        """If LLM fails, dictionary result still works."""
        provider = MagicMock(spec=LLMProvider)
        provider.complete.side_effect = RuntimeError("API down")
        fallback = LLMFallback(provider)
        config = TxtpandConfig(llm_enabled=True)
        expander = Expander(config=config, llm_fallback=fallback)

        result = expander.expand("cn y hel me")
        # Should still get dictionary result, not crash
        assert "can" in result
        assert "help" in result or "hello" in result


class TestPromptBuilding:
    def test_system_prompt_exists(self):
        assert len(DISAMBIGUATION_SYSTEM) > 0

    def test_polish_system_prompt_exists(self):
        assert len(POLISH_SYSTEM) > 0

    def test_build_prompt(self):
        tokens = [("wo", ["work", "would", "world"])]
        prompt = build_disambiguation_prompt(tokens, "can you wo on this")
        assert "wo" in prompt
        assert "work" in prompt
        assert "can you wo on this" in prompt

    def test_build_prompt_multiple(self):
        tokens = [
            ("wo", ["work", "would"]),
            ("fe", ["few", "feel"]),
        ]
        prompt = build_disambiguation_prompt(tokens, "some context")
        assert "wo" in prompt
        assert "fe" in prompt

    def test_build_polish_prompt(self):
        prompt = build_polish_prompt("cn y hel me", "can you help me")
        assert "cn y hel me" in prompt
        assert "can you help me" in prompt

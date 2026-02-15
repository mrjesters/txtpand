"""LLM fallback for disambiguation and full-sentence polish."""

from __future__ import annotations

from typing import TYPE_CHECKING

from txtpand.exceptions import LLMFallbackError
from txtpand.llm.prompt import (
    DISAMBIGUATION_SYSTEM,
    POLISH_SYSTEM,
    build_disambiguation_prompt,
    build_polish_prompt,
)

if TYPE_CHECKING:
    from txtpand.llm.providers import LLMProvider
    from txtpand.types import TokenResult


class LLMFallback:
    """LLM-powered expansion support.

    Two modes:
    - resolve(): batched disambiguation for ambiguous tokens only
    - polish(): full-sentence correction pass after dictionary expansion
    """

    def __init__(
        self,
        provider: LLMProvider,
        timeout: float = 2.0,
    ) -> None:
        self.provider = provider
        self.timeout = timeout

    def polish(self, original: str, expanded: str) -> str:
        """Polish a dictionary-expanded sentence using LLM.

        Sends the original shorthand + dictionary expansion to the LLM,
        which corrects any misexpansions in a single call.

        Args:
            original: The original shorthand input.
            expanded: The dictionary-based first-pass expansion.

        Returns:
            Corrected sentence.

        Raises:
            LLMFallbackError: If the LLM call fails.
        """
        user_prompt = build_polish_prompt(original, expanded)

        try:
            response = self.provider.complete(
                POLISH_SYSTEM, user_prompt, self.timeout
            )
        except Exception as e:
            raise LLMFallbackError(f"LLM polish call failed: {e}") from e

        result = response.strip()
        # Strip quotes if the LLM wrapped the response in them
        if len(result) >= 2 and result[0] == '"' and result[-1] == '"':
            result = result[1:-1]

        # Sanity: if LLM returned something wildly different length, keep original
        if not result or len(result) > len(expanded) * 3:
            return expanded

        return result

    def resolve(
        self,
        ambiguous_tokens: list[tuple[int, TokenResult]],
        context: str,
    ) -> dict[int, str]:
        """Resolve ambiguous tokens using LLM.

        Args:
            ambiguous_tokens: List of (index, TokenResult) for ambiguous tokens.
            context: Full sentence with current best guesses.

        Returns:
            Dict mapping token index → resolved word.
        """
        if not ambiguous_tokens:
            return {}

        # Build prompt with candidates
        prompt_tokens: list[tuple[str, list[str]]] = []
        for _, tr in ambiguous_tokens:
            candidates = [c.word for c in tr.candidates[:5]]
            prompt_tokens.append((tr.original, candidates))

        user_prompt = build_disambiguation_prompt(prompt_tokens, context)

        try:
            response = self.provider.complete(
                DISAMBIGUATION_SYSTEM, user_prompt, self.timeout
            )
        except Exception as e:
            raise LLMFallbackError(f"LLM call failed: {e}") from e

        # Parse response: one word per line
        lines = [
            line.strip()
            for line in response.strip().splitlines()
            if line.strip()
        ]

        results: dict[int, str] = {}
        for i, (idx, _) in enumerate(ambiguous_tokens):
            if i < len(lines):
                word = lines[i].strip().lower()
                # Basic sanity: single word, alphabetic
                if word and word.isalpha():
                    results[idx] = word

        return results

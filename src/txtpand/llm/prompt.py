"""Prompt templates for LLM disambiguation and polish."""

from __future__ import annotations

DISAMBIGUATION_SYSTEM = """\
You are a text expansion assistant. Given abbreviated tokens and their context, \
determine the most likely full English word for each abbreviation.

Rules:
- Return ONLY the expanded words, one per line, in the same order as the input tokens.
- Each line should contain a single word.
- Choose the word that best fits the sentence context.
- If unsure, pick the most common word that starts with the abbreviation.\
"""

POLISH_SYSTEM = """\
You are a text expansion assistant. The user typed shorthand/abbreviated text \
and a dictionary-based system produced a first-pass expansion. Your job is to \
fix any incorrect expansions so the final sentence reads naturally.

Rules:
- Return ONLY the corrected full sentence, nothing else.
- Fix words that were clearly expanded wrong (e.g. "thin" should be "things" if context demands it).
- Do NOT add words, remove words, or rephrase. Only fix individual word expansions.
- If the expansion already looks correct, return it unchanged.
- Keep the same word count — map each expanded token 1:1.
- Preserve punctuation, URLs, code blocks, and quoted strings exactly.\
"""


def build_disambiguation_prompt(
    tokens: list[tuple[str, list[str]]],
    context: str,
) -> str:
    """Build a disambiguation prompt for the LLM.

    Args:
        tokens: List of (abbreviation, top_candidate_words) pairs.
        context: The full sentence context (with current best guesses).

    Returns:
        User prompt string.
    """
    lines = [f"Context: \"{context}\"", "", "Expand these abbreviated tokens:"]
    for abbrev, candidates in tokens:
        cand_str = ", ".join(candidates[:5])
        lines.append(f"  \"{abbrev}\" → candidates: [{cand_str}]")
    lines.append("")
    lines.append("Return one word per line, in order:")
    return "\n".join(lines)


def build_polish_prompt(original: str, expanded: str) -> str:
    """Build a polish prompt for the LLM.

    Args:
        original: The original shorthand input.
        expanded: The dictionary-based first-pass expansion.

    Returns:
        User prompt string.
    """
    return (
        f"Original shorthand: \"{original}\"\n"
        f"Dictionary expansion: \"{expanded}\"\n\n"
        f"Return the corrected sentence:"
    )

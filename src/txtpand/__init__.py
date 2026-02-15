"""txtpand — Shorthand text expansion library.

Type less. Say more.

Usage:
    import txtpand

    # Simple expansion
    txtpand.expand("cn y hel me wo on a fe thin")
    # → "can you help me work on a few things"

    # Spaceless mode
    txtpand.expand("cnyhelmewoonfethin", spaceless=True)

    # Detailed results
    report = txtpand.expand_detailed("cn y hel me")

    # With custom config
    expander = txtpand.Expander()
    expander.add_words({"kubernetes": 5.0})
    expander.add_abbreviations({"k8s": "kubernetes"})
"""

from __future__ import annotations

from txtpand.config import TxtpandConfig
from txtpand.core.expander import Expander
from txtpand.types import ExpansionReport, MatchTier, ScoredCandidate, TokenResult

__version__ = "0.1.0"

__all__ = [
    "expand",
    "expand_detailed",
    "Expander",
    "TxtpandConfig",
    "ExpansionReport",
    "TokenResult",
    "ScoredCandidate",
    "MatchTier",
    "wrap_openai",
    "wrap_anthropic",
]

# Module-level default expander (lazy init)
_default_expander: Expander | None = None


def _get_default() -> Expander:
    global _default_expander
    if _default_expander is None:
        _default_expander = Expander()
    return _default_expander


def expand(text: str, spaceless: bool = False) -> str:
    """Expand shorthand text to full English.

    Args:
        text: Shorthand input like "cn y hel me".
        spaceless: If True, treat input as having no spaces.

    Returns:
        Expanded text string.
    """
    return _get_default().expand(text, spaceless=spaceless)


def expand_detailed(text: str, spaceless: bool = False) -> ExpansionReport:
    """Expand with full diagnostic report.

    Args:
        text: Shorthand input.
        spaceless: If True, treat input as having no spaces.

    Returns:
        ExpansionReport with per-token details.
    """
    return _get_default().expand_detailed(text, spaceless=spaceless)


def wrap_openai(client: object) -> object:
    """Wrap an OpenAI client to auto-expand user messages.

    Args:
        client: An openai.OpenAI() instance.

    Returns:
        Wrapped client that expands shorthand in user messages.
    """
    from txtpand.middleware.openai import wrap_openai as _wrap

    return _wrap(client)


def wrap_anthropic(client: object) -> object:
    """Wrap an Anthropic client to auto-expand user messages.

    Args:
        client: An anthropic.Anthropic() instance.

    Returns:
        Wrapped client that expands shorthand in user messages.
    """
    from txtpand.middleware.anthropic import wrap_anthropic as _wrap

    return _wrap(client)

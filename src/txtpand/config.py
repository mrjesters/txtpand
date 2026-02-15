"""Configuration for txtpand."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TxtpandConfig:
    """Configuration parameters for text expansion."""

    # Scoring weights (must sum to 1.0)
    weight_prefix: float = 0.35
    weight_edit: float = 0.25
    weight_frequency: float = 0.25
    weight_length: float = 0.15

    # Context bonus weight applied on top of base score
    context_bonus_weight: float = 0.20

    # Ambiguity threshold: if top-2 candidates are within this margin, flag as ambiguous
    ambiguity_margin: float = 0.15

    # Minimum score to accept a candidate
    min_confidence: float = 0.20

    # Maximum edit distance for fuzzy matching (fraction of token length)
    max_edit_distance_ratio: float = 0.6

    # Maximum word length to consider in segmenter
    max_word_length: int = 20

    # LLM fallback settings
    llm_timeout_seconds: float = 2.0
    llm_enabled: bool = False

    # Passthrough: tokens that are already known words skip matching
    passthrough_known_words: bool = True

    # Minimum token length for fuzzy matching (shorter tokens use prefix only)
    min_fuzzy_length: int = 2

    # Top-k candidates to keep per tier
    top_k_prefix: int = 10
    top_k_fuzzy: int = 10

    # User model persistence path (None = disabled)
    user_model_path: str | None = None

    # Custom abbreviation overrides
    abbreviations: dict[str, str] = field(default_factory=dict)

    # Custom word additions with frequencies
    custom_words: dict[str, float] = field(default_factory=dict)

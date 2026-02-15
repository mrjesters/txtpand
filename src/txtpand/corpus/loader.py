"""Load bundled word and bigram frequency data."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from txtpand.exceptions import CorpusLoadError

_CORPUS_DIR = Path(__file__).parent


@lru_cache(maxsize=1)
def get_words() -> dict[str, float]:
    """Load word frequencies (word → zipf score).

    Returns cached dictionary on subsequent calls.
    """
    path = _CORPUS_DIR / "words.json"
    try:
        with open(path) as f:
            data: dict[str, float] = json.load(f)
        return data
    except (OSError, json.JSONDecodeError) as e:
        raise CorpusLoadError(f"Failed to load words from {path}: {e}") from e


@lru_cache(maxsize=1)
def get_bigrams() -> dict[str, float]:
    """Load bigram frequencies (word1_word2 → log-probability).

    Returns cached dictionary on subsequent calls.
    """
    path = _CORPUS_DIR / "bigrams.json"
    try:
        with open(path) as f:
            data: dict[str, float] = json.load(f)
        return data
    except (OSError, json.JSONDecodeError) as e:
        raise CorpusLoadError(f"Failed to load bigrams from {path}: {e}") from e

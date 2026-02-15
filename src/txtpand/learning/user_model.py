"""User model for learning from corrections.

Persists user corrections and adjusts candidate scores over time.
"""

from __future__ import annotations

import json
from pathlib import Path


class UserModel:
    """Tracks user corrections to improve future expansions.

    Stores corrections as (abbreviation → preferred word) pairs with counts.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._corrections: dict[str, dict[str, int]] = {}
        if self._path and self._path.exists():
            self._load()

    def _load(self) -> None:
        assert self._path is not None
        try:
            with open(self._path) as f:
                self._corrections = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._corrections = {}

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._corrections, f, indent=2)

    def record_correction(self, abbreviation: str, correct_word: str) -> None:
        """Record that the user corrected an abbreviation to a specific word."""
        abbrev = abbreviation.lower()
        word = correct_word.lower()
        if abbrev not in self._corrections:
            self._corrections[abbrev] = {}
        self._corrections[abbrev][word] = self._corrections[abbrev].get(word, 0) + 1
        self._save()

    def get_preference(self, abbreviation: str) -> str | None:
        """Get the user's preferred expansion for an abbreviation.

        Returns the most-corrected word, or None if no corrections recorded.
        """
        abbrev = abbreviation.lower()
        if abbrev not in self._corrections:
            return None
        prefs = self._corrections[abbrev]
        if not prefs:
            return None
        return max(prefs, key=lambda w: prefs[w])

    def get_boost(self, abbreviation: str, word: str) -> float:
        """Get a score boost for a candidate based on user history.

        Returns a value between 0.0 and 0.3 based on correction frequency.
        """
        abbrev = abbreviation.lower()
        word = word.lower()
        if abbrev not in self._corrections:
            return 0.0
        count = self._corrections[abbrev].get(word, 0)
        total = sum(self._corrections[abbrev].values())
        if total == 0:
            return 0.0
        # Boost proportional to how often user chose this word (max 0.3)
        return min(0.3, count / total * 0.3)

    def get_all_corrections(self) -> dict[str, dict[str, int]]:
        """Get all recorded corrections."""
        return dict(self._corrections)

    def clear(self) -> None:
        """Clear all corrections."""
        self._corrections = {}
        self._save()

"""Exception hierarchy for txtpand."""


class TxtpandError(Exception):
    """Base exception for all txtpand errors."""


class CorpusLoadError(TxtpandError):
    """Failed to load corpus data."""


class SegmentationError(TxtpandError):
    """Failed to segment spaceless input."""


class LLMFallbackError(TxtpandError):
    """LLM fallback call failed."""


class ConfigError(TxtpandError):
    """Invalid configuration."""

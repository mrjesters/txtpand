"""Persistent config file support.

Loads settings from ~/.config/txtpand/config.toml so users
configure their API key and model once.

Example config file:

    [llm]
    provider = "openai"          # "openai" or "anthropic"
    model = "gpt-4o-mini"        # cheap + fast
    api_key = "sk-..."           # or set OPENAI_API_KEY env var
    timeout = 2.0

    [expansion]
    spaceless = false
    passthrough_known_words = true
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Try tomllib (3.11+) then tomli, then fall back to basic parsing
try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


_DEFAULT_CONFIG_DIR = Path.home() / ".config" / "txtpand"
_DEFAULT_CONFIG_FILE = _DEFAULT_CONFIG_DIR / "config.toml"


@dataclass
class LLMSettings:
    """LLM configuration loaded from config file."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    timeout: float = 2.0
    enabled: bool = False


@dataclass
class UserConfig:
    """Full user configuration."""

    llm: LLMSettings
    spaceless: bool = False
    passthrough_known_words: bool = True


def load_config(path: Path | None = None) -> UserConfig:
    """Load config from file, falling back to defaults.

    Checks in order:
    1. Explicit path argument
    2. TXTPAND_CONFIG env var
    3. ~/.config/txtpand/config.toml
    """
    if path is None:
        env_path = os.environ.get("TXTPAND_CONFIG")
        if env_path:
            path = Path(env_path)
        else:
            path = _DEFAULT_CONFIG_FILE

    llm = LLMSettings()
    spaceless = False
    passthrough = True

    if path.exists() and tomllib is not None:
        with open(path, "rb") as f:
            data = tomllib.load(f)

        llm_data = data.get("llm", {})
        if llm_data:
            llm.provider = llm_data.get("provider", llm.provider)
            llm.model = llm_data.get("model", llm.model)
            llm.api_key = llm_data.get("api_key", llm.api_key)
            llm.timeout = float(llm_data.get("timeout", llm.timeout))
            llm.enabled = True

        exp_data = data.get("expansion", {})
        spaceless = exp_data.get("spaceless", spaceless)
        passthrough = exp_data.get("passthrough_known_words", passthrough)

    # Environment variables override config file
    env_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if env_key and not llm.api_key:
        llm.api_key = env_key
        llm.enabled = True
        if os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            llm.provider = "anthropic"

    return UserConfig(llm=llm, spaceless=spaceless, passthrough_known_words=passthrough)


def init_config_file() -> Path:
    """Create a default config file if one doesn't exist. Returns the path."""
    _DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if not _DEFAULT_CONFIG_FILE.exists():
        _DEFAULT_CONFIG_FILE.write_text(
            '# txtpand configuration\n'
            '# See: https://github.com/txtpand/txtpand\n'
            '\n'
            '[llm]\n'
            '# Uncomment and set your API key to enable LLM polish.\n'
            '# Or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var.\n'
            '# provider = "openai"       # "openai" or "anthropic"\n'
            '# model = "gpt-4o-mini"     # cheap + fast\n'
            '# api_key = "sk-..."\n'
            '# timeout = 2.0\n'
            '\n'
            '[expansion]\n'
            'spaceless = false\n'
            'passthrough_known_words = true\n'
        )

    return _DEFAULT_CONFIG_FILE

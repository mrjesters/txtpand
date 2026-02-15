"""Build an Expander from user config (config file + env vars)."""

from __future__ import annotations

import sys

from txtpand.cli.config_file import UserConfig, load_config
from txtpand.config import TxtpandConfig
from txtpand.core.expander import Expander


def build_expander(user_config: UserConfig | None = None) -> Expander:
    """Create a fully configured Expander from user settings."""
    if user_config is None:
        user_config = load_config()

    config = TxtpandConfig(
        llm_enabled=user_config.llm.enabled,
        llm_timeout_seconds=user_config.llm.timeout,
        passthrough_known_words=user_config.passthrough_known_words,
    )

    llm_fallback = None
    if user_config.llm.enabled and user_config.llm.api_key:
        llm_fallback = _build_llm_fallback(user_config)
        if llm_fallback is None:
            config.llm_enabled = False

    return Expander(config=config, llm_fallback=llm_fallback)


def _build_llm_fallback(user_config: UserConfig):
    """Try to build an LLM fallback from config. Returns None on failure."""
    from txtpand.llm.fallback import LLMFallback

    provider = user_config.llm.provider.lower()

    if provider == "openai":
        try:
            import openai

            client = openai.OpenAI(api_key=user_config.llm.api_key)
            from txtpand.llm.providers import OpenAIProvider

            return LLMFallback(
                OpenAIProvider(client=client, model=user_config.llm.model),
                timeout=user_config.llm.timeout,
            )
        except ImportError:
            print("txtpand: openai package not installed. Run: pip install txtpand[openai]", file=sys.stderr)
            return None

    elif provider == "anthropic":
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=user_config.llm.api_key)
            from txtpand.llm.providers import AnthropicProvider

            return LLMFallback(
                AnthropicProvider(client=client, model=user_config.llm.model),
                timeout=user_config.llm.timeout,
            )
        except ImportError:
            print("txtpand: anthropic package not installed. Run: pip install txtpand[anthropic]", file=sys.stderr)
            return None

    else:
        print(f"txtpand: unknown LLM provider '{provider}'. Use 'openai' or 'anthropic'.", file=sys.stderr)
        return None

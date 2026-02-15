"""Shared test fixtures for txtpand."""

from __future__ import annotations

import pytest

from txtpand.config import TxtpandConfig
from txtpand.core.context import ContextResolver
from txtpand.core.matcher import Matcher
from txtpand.core.scorer import Scorer


@pytest.fixture
def config() -> TxtpandConfig:
    return TxtpandConfig()


@pytest.fixture
def matcher(config: TxtpandConfig) -> Matcher:
    m = Matcher(config)
    m.build()
    return m


@pytest.fixture
def context_resolver(config: TxtpandConfig) -> ContextResolver:
    return ContextResolver(config)


@pytest.fixture
def scorer(config: TxtpandConfig) -> Scorer:
    return Scorer(config)

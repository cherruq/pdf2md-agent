"""Tests for the ``resolve_ctx_limit`` priority chain.

The chain, highest priority first:

1. ``PDF2MD_AGENT_CTX_LIMIT`` env var (if set to a positive int)
2. Hardcoded per-model default (e.g. MiniMax-M3 → 524 288)
3. Generic fallback (``_DEFAULT_CTX_LIMIT``)

``resolve_ctx_limit`` is ``lru_cache``d at module level so every test starts
by clearing the cache to avoid cross-test pollution.
"""
from __future__ import annotations

import pytest

from pdf2md_agent import config


@pytest.fixture(autouse=True)
def _reset_ctx_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop ``lru_cache`` state and known env knobs before each test."""
    config.resolve_ctx_limit.cache_clear()
    monkeypatch.delenv("PDF2MD_AGENT_CTX_LIMIT", raising=False)


def test_env_var_wins_over_hardcoded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PDF2MD_AGENT_CTX_LIMIT", "12345")
    result = config.resolve_ctx_limit()
    assert result == 12345


def test_hardcoded_used_when_env_var_unset() -> None:
    result = config.resolve_ctx_limit()
    assert result == config._HARD_CODED_CTX_LIMITS[config.MODEL_NAME]


def test_generic_fallback_when_model_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "MODEL_NAME", "totally-unknown-model-xyz")
    result = config.resolve_ctx_limit()
    assert result == config._DEFAULT_CTX_LIMIT


def test_result_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Second call within the same test must use cached result."""
    first = config.resolve_ctx_limit()
    # Modify MODEL_NAME; cached result should remain unchanged.
    monkeypatch.setattr(config, "MODEL_NAME", "totally-unknown-model-xyz")
    second = config.resolve_ctx_limit()
    assert first == second


def test_invalid_env_var_falls_through_to_hardcoded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-integer env value is ignored (not a hard error)."""
    monkeypatch.setenv("PDF2MD_AGENT_CTX_LIMIT", "not-a-number")
    result = config.resolve_ctx_limit()
    assert result == config._HARD_CODED_CTX_LIMITS[config.MODEL_NAME]


def test_zero_env_var_treated_as_unset_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """``0`` is meaningless for ctx_limit (unlike for retries) — ignore it."""
    monkeypatch.setenv("PDF2MD_AGENT_CTX_LIMIT", "0")
    result = config.resolve_ctx_limit()
    assert result == config._HARD_CODED_CTX_LIMITS[config.MODEL_NAME]


def test_minimax_m3_hardcoded_default_is_512k(monkeypatch: pytest.MonkeyPatch) -> None:
    """The published 512K guarantee is baked in for the default model."""
    monkeypatch.setattr(config, "MODEL_NAME", "MiniMax-M3")
    result = config.resolve_ctx_limit()
    assert result == 524_288
    assert result == 512 * 1024
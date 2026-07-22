"""Tests for the ``resolve_ctx_limit`` priority chain.

The chain, highest priority first:

1. ``PDF2MD_AGENT_CTX_LIMIT`` env var (if set to a positive int)
2. ``probe_ctx_limit(OPENAI_BASE_URL, ..., MODEL_NAME)`` (clamped to 1M)
3. Hardcoded per-model default (e.g. MiniMax-M3 → 524 288)
4. Generic fallback (``_DEFAULT_CTX_LIMIT``)

The probe and hardcoded steps are mocked here so no real network I/O runs.
``resolve_ctx_limit`` is ``lru_cache``d at module level so every test starts
by clearing the cache to avoid cross-test pollution.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from pdf2md_agent import config


@pytest.fixture(autouse=True)
def _reset_ctx_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop ``lru_cache`` state and known env knobs before each test."""
    config.resolve_ctx_limit.cache_clear()
    monkeypatch.delenv("PDF2MD_AGENT_CTX_LIMIT", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_env_var_wins_over_probe_and_hardcoded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PDF2MD_AGENT_CTX_LIMIT", "12345")
    with patch(
        "pdf2md_agent.config.probe_ctx_limit", return_value=999999
    ) as mock_probe:
        result = config.resolve_ctx_limit()
    assert result == 12345
    mock_probe.assert_not_called()


def test_probe_used_when_env_var_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    with patch(
        "pdf2md_agent.config.probe_ctx_limit", return_value=524288
    ) as mock_probe:
        result = config.resolve_ctx_limit()
    assert result == 524288
    mock_probe.assert_called_once()


def test_probe_result_clamped_to_1M(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    with patch(
        "pdf2md_agent.config.probe_ctx_limit", return_value=10_000_000
    ):
        result = config.resolve_ctx_limit()
    assert result == 1_048_576


def test_hardcoded_used_when_probe_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    with patch("pdf2md_agent.config.probe_ctx_limit", return_value=None):
        result = config.resolve_ctx_limit()
    assert result == config._HARD_CODED_CTX_LIMITS[config.MODEL_NAME]


def test_hardcoded_used_when_api_key_missing() -> None:
    # No OPENAI_API_KEY → probe skipped entirely, hardcoded wins.
    result = config.resolve_ctx_limit()
    assert result == config._HARD_CODED_CTX_LIMITS[config.MODEL_NAME]


def test_generic_fallback_when_model_unknown_and_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setattr(config, "MODEL_NAME", "totally-unknown-model-xyz")
    with patch("pdf2md_agent.config.probe_ctx_limit", return_value=None):
        result = config.resolve_ctx_limit()
    assert result == config._DEFAULT_CTX_LIMIT


def test_result_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Second call within the same test must not re-probe."""
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    with patch(
        "pdf2md_agent.config.probe_ctx_limit", return_value=4096
    ) as mock_probe:
        first = config.resolve_ctx_limit()
        second = config.resolve_ctx_limit()
    assert first == second == 4096
    assert mock_probe.call_count == 1


def test_invalid_env_var_falls_through_to_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-integer env value is ignored (not a hard error)."""
    monkeypatch.setenv("PDF2MD_AGENT_CTX_LIMIT", "not-a-number")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    with patch(
        "pdf2md_agent.config.probe_ctx_limit", return_value=4096
    ) as mock_probe:
        result = config.resolve_ctx_limit()
    assert result == 4096
    mock_probe.assert_called_once()


def test_zero_env_var_treated_as_unset_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """``0`` is meaningless for ctx_limit (unlike for retries) — ignore it."""
    monkeypatch.setenv("PDF2MD_AGENT_CTX_LIMIT", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    with patch(
        "pdf2md_agent.config.probe_ctx_limit", return_value=4096
    ) as mock_probe:
        result = config.resolve_ctx_limit()
    assert result == 4096
    mock_probe.assert_called_once()


def test_minimax_m3_hardcoded_default_is_512k(monkeypatch: pytest.MonkeyPatch) -> None:
    """The published 512K guarantee is baked in for the default model."""
    monkeypatch.setattr(config, "MODEL_NAME", "MiniMax-M3")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("pdf2md_agent.config.probe_ctx_limit", return_value=None):
        result = config.resolve_ctx_limit()
    assert result == 524_288
    assert result == 512 * 1024


def test_env_var_takes_precedence_even_over_higher_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """User-supplied env var is authoritative, no probe call made."""
    monkeypatch.setenv("PDF2MD_AGENT_CTX_LIMIT", "8192")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    with patch(
        "pdf2md_agent.config.probe_ctx_limit", return_value=1_000_000
    ) as mock_probe:
        result = config.resolve_ctx_limit()
    assert result == 8192
    mock_probe.assert_not_called()


def test_probe_receives_configured_base_url_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    with patch(
        "pdf2md_agent.config.probe_ctx_limit", return_value=4096
    ) as mock_probe:
        config.resolve_ctx_limit()
    args, kwargs = mock_probe.call_args
    # Either positional or keyword; both are fine.
    assert config.OPENAI_BASE_URL in (args + tuple(kwargs.values()))
    assert config.MODEL_NAME in (args + tuple(kwargs.values()))
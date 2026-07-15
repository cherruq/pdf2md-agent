"""Tests for convertpdf.llm_retry."""
from __future__ import annotations

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from convertpdf.llm_retry import RetryConfig, call_with_retry, is_transient


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", "https://example.test"))


def _request() -> httpx.Request:
    return httpx.Request("GET", "https://example.test")


# --- RetryConfig validation --------------------------------------------------


def test_retry_config_defaults_are_valid() -> None:
    cfg = RetryConfig()
    assert cfg.max_attempts >= 1
    assert cfg.initial_delay >= 0
    assert cfg.backoff >= 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"initial_delay": -0.1},
        {"backoff": 0.5},
        {"max_delay": 0.1, "initial_delay": 1.0},
        {"jitter": -0.1},
        {"jitter": 1.5},
    ],
)
def test_retry_config_rejects_invalid_kwargs(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        RetryConfig(**kwargs)


# --- is_transient classification --------------------------------------------


def test_is_transient_true_for_known_transient() -> None:
    assert is_transient(APITimeoutError(request=_request()))
    assert is_transient(APIConnectionError(request=_request()))
    assert is_transient(
        InternalServerError(message="boom", response=_response(500), body=None)
    )
    assert is_transient(
        RateLimitError(message="slow down", response=_response(429), body=None)
    )


def test_is_transient_true_for_5xx_status() -> None:
    err = InternalServerError(message="bad gateway", response=_response(503), body=None)
    assert is_transient(err)
    assert err.status_code >= 500


def test_is_transient_false_for_4xx_status() -> None:
    err = BadRequestError(message="bad input", response=_response(400), body=None)
    assert not is_transient(err)


def test_is_transient_false_for_authentication() -> None:
    err = AuthenticationError(message="bad key", response=_response(401), body=None)
    assert not is_transient(err)


def test_is_transient_false_for_unrelated_exception() -> None:
    assert not is_transient(ValueError("not an API error"))
    assert not is_transient(RuntimeError("boom"))


# --- call_with_retry: success paths -----------------------------------------


def test_call_with_retry_returns_on_first_success() -> None:
    calls = []

    def fn() -> str:
        calls.append(1)
        return "ok"

    sleeps: list[float] = []
    result = call_with_retry(fn, sleep=sleeps.append)
    assert result == "ok"
    assert len(calls) == 1
    assert sleeps == []


def test_call_with_retry_returns_after_transient_failures() -> None:
    attempts = []

    def fn() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise APITimeoutError(request=_request())
        return "recovered"

    sleeps: list[float] = []
    result = call_with_retry(
        fn,
        config=RetryConfig(max_attempts=4, initial_delay=0.1, backoff=2.0, jitter=0.0),
        sleep=sleeps.append,
    )
    assert result == "recovered"
    assert len(attempts) == 3
    assert sleeps == pytest.approx([0.1, 0.2])


# --- call_with_retry: failure paths -----------------------------------------


def test_call_with_retry_raises_permanent_immediately() -> None:
    attempts = []

    def fn() -> None:
        attempts.append(1)
        raise BadRequestError(message="bad input", response=_response(400), body=None)

    sleeps: list[float] = []
    with pytest.raises(BadRequestError):
        call_with_retry(fn, sleep=sleeps.append)
    assert len(attempts) == 1
    assert sleeps == []


def test_call_with_retry_raises_after_max_attempts() -> None:
    attempts = []

    def fn() -> None:
        attempts.append(1)
        raise APITimeoutError(request=_request())

    sleeps: list[float] = []
    with pytest.raises(APITimeoutError):
        call_with_retry(
            fn,
            config=RetryConfig(max_attempts=3, initial_delay=0.1, backoff=2.0, jitter=0.0),
            sleep=sleeps.append,
        )
    assert len(attempts) == 3
    assert sleeps == pytest.approx([0.1, 0.2])


def test_call_with_retry_jitter_stays_within_bounds() -> None:
    attempts = []

    def fn() -> None:
        attempts.append(1)
        raise APIConnectionError(request=_request())

    sleeps: list[float] = []
    with pytest.raises(APIConnectionError):
        call_with_retry(
            fn,
            config=RetryConfig(
                max_attempts=4,
                initial_delay=1.0,
                backoff=2.0,
                max_delay=30.0,
                jitter=0.25,
            ),
            sleep=sleeps.append,
        )
    assert len(sleeps) == 3
    expected_bases = [1.0, 2.0, 4.0]
    for sleep, base in zip(sleeps, expected_bases):
        assert base * 0.75 <= sleep <= base * 1.25


def test_call_with_retry_label_is_passed_through(caplog) -> None:
    caplog.set_level("WARNING", logger="convertpdf.llm_retry")

    def fn() -> None:
        raise APITimeoutError(request=_request())

    sleeps: list[float] = []
    with pytest.raises(APITimeoutError):
        call_with_retry(
            fn,
            config=RetryConfig(max_attempts=2, initial_delay=0.0, backoff=2.0, jitter=0.0),
            label="page 7",
            sleep=sleeps.append,
        )
    assert any("page 7" in rec.message for rec in caplog.records)
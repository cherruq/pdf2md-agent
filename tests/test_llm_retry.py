"""Tests for pdf2md_agent.llm_retry."""
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

from pdf2md_agent.llm_retry import RetryConfig, call_with_retry, is_transient


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", "https://example.test"))


def _request() -> httpx.Request:
    return httpx.Request("GET", "https://example.test")


# --- RetryConfig validation --------------------------------------------------


def test_retry_config_defaults_are_valid() -> None:
    cfg = RetryConfig()
    assert cfg.max_attempts is None
    assert cfg.max_delay == 900.0
    assert cfg.initial_delay >= 0
    assert 0.0 <= cfg.jitter <= 1.0


def test_retry_config_default_max_attempts_is_none_for_unlimited_retries() -> None:
    assert RetryConfig().max_attempts is None


def test_retry_config_default_max_delay_is_900_seconds() -> None:
    assert RetryConfig().max_delay == 900.0


def test_retry_config_rejects_explicit_zero_max_attempts() -> None:
    # Sentinel ``0`` = unlimited is handled at CLI/env-parsing boundary;
    # the dataclass itself rejects 0 so misuse isn't silently no-op.
    with pytest.raises(ValueError):
        RetryConfig(max_attempts=0)


def test_retry_config_rejects_negative_max_attempts() -> None:
    with pytest.raises(ValueError):
        RetryConfig(max_attempts=-1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_delay": -0.1},
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
        config=RetryConfig(max_attempts=4, initial_delay=0.1, jitter=0.0),
        sleep=sleeps.append,
    )
    assert result == "recovered"
    assert len(attempts) == 3
    # Fibonacci: F[1]=1, F[2]=1 ⇒ sleeps before retry 2 and retry 3
    # are both 0.1×initial_delay = 0.1.
    assert sleeps == pytest.approx([0.1, 0.1])


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
            config=RetryConfig(max_attempts=3, initial_delay=0.1, jitter=0.0),
            sleep=sleeps.append,
        )
    assert len(attempts) == 3
    # Fibonacci before the 2nd and 3rd attempts: F[1]=1, F[2]=1 ⇒ 0.1, 0.1.
    assert sleeps == pytest.approx([0.1, 0.1])


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
                max_delay=30.0,
                jitter=0.25,
            ),
            sleep=sleeps.append,
        )
    assert len(sleeps) == 3
    # Fibonacci multipliers for 3 retries: F[1]=1, F[2]=1, F[3]=2
    expected_bases = [1.0, 1.0, 2.0]
    for sleep, base in zip(sleeps, expected_bases):
        assert base * 0.75 <= sleep <= base * 1.25


def test_call_with_retry_label_is_passed_through(caplog) -> None:
    caplog.set_level("WARNING", logger="pdf2md_agent.llm_retry")

    def fn() -> None:
        raise APITimeoutError(request=_request())

    sleeps: list[float] = []
    with pytest.raises(APITimeoutError):
        call_with_retry(
            fn,
            config=RetryConfig(max_attempts=2, initial_delay=0.0, jitter=0.0),
            label="page 7",
            sleep=sleeps.append,
        )
    assert any("page 7" in rec.message for rec in caplog.records)


# --- Fibonacci backoff -------------------------------------------------------


def test_call_with_retry_fibonacci_sequence_for_5_retries() -> None:
    attempts = []

    def fn() -> None:
        attempts.append(1)
        raise APITimeoutError(request=_request())

    sleeps: list[float] = []
    with pytest.raises(APITimeoutError):
        call_with_retry(
            fn,
            config=RetryConfig(
                max_attempts=7, initial_delay=2.0, max_delay=900.0, jitter=0.0
            ),
            sleep=sleeps.append,
        )
    assert len(attempts) == 7
    # F[1..6] = [1, 1, 2, 3, 5, 8] ⇒ sleeps = initial * multiplier
    expected = [2.0, 2.0, 4.0, 6.0, 10.0, 16.0]
    assert sleeps == pytest.approx(expected)


def test_call_with_retry_fibonacci_caps_at_max_delay() -> None:
    attempts = []

    def fn() -> None:
        attempts.append(1)
        raise APITimeoutError(request=_request())

    sleeps: list[float] = []
    with pytest.raises(APITimeoutError):
        call_with_retry(
            fn,
            config=RetryConfig(
                max_attempts=6, initial_delay=2.0, max_delay=7.0, jitter=0.0
            ),
            sleep=sleeps.append,
        )
    # F[1..5] = [1, 1, 2, 3, 5] × 2 = [2, 2, 4, 6, 10] → last capped at 7.0
    expected = [2.0, 2.0, 4.0, 6.0, 7.0]
    assert sleeps == pytest.approx(expected)


# --- Infinite retries (max_attempts=None) -----------------------------------


def test_call_with_retry_infinite_max_attempts_eventually_succeeds() -> None:
    attempts = []

    def fn() -> str:
        attempts.append(1)
        if len(attempts) < 5:
            raise APITimeoutError(request=_request())
        return "eventually"

    sleeps: list[float] = []
    result = call_with_retry(
        fn,
        config=RetryConfig(max_attempts=None, initial_delay=0.1, jitter=0.0),
        sleep=sleeps.append,
    )
    assert result == "eventually"
    assert len(attempts) == 5
    assert len(sleeps) == 4


def test_call_with_retry_infinite_max_attempts_propagates_non_transient() -> None:
    attempts = []

    def fn() -> None:
        attempts.append(1)
        raise BadRequestError(message="bad", response=_response(400), body=None)

    with pytest.raises(BadRequestError):
        call_with_retry(
            fn,
            config=RetryConfig(max_attempts=None, initial_delay=0.0, jitter=0.0),
            sleep=lambda _w: None,
        )
    assert len(attempts) == 1


def test_call_with_retry_infinite_max_attempts_uses_max_delay_cap_on_fibonacci() -> None:
    attempts = []

    def fn() -> str:
        attempts.append(1)
        if len(attempts) < 4:
            raise APITimeoutError(request=_request())
        return "ok"

    sleeps: list[float] = []
    call_with_retry(
        fn,
        config=RetryConfig(
            max_attempts=None,
            initial_delay=2.0,
            max_delay=3.0,
            jitter=0.0,
        ),
        sleep=sleeps.append,
    )
    # F[1..3] × 2 = [2, 2, 4] → first two pass, third capped at 3.0
    assert sleeps == pytest.approx([2.0, 2.0, 3.0])
"""Retry helper for transient LLM-API failures.

Vision-model calls go through the OpenAI SDK against a custom ``base_url``
(MiniMax-M3). That endpoint, like any HTTP service, can return:

* network-level failures (timeout, connection refused, DNS hiccup)
* transient server errors (HTTP 5xx, gateway 502/503/504)
* rate-limit responses (HTTP 429)

We retry those with exponential backoff + jitter. Permanent failures
(authentication, bad request, permission denied) are NOT retried: re-issuing
the exact same request just burns the budget and re-fails identically.

The runner wraps each per-page ``crew.kickoff()`` in :func:`call_with_retry`
and, on retry exhaustion, hands the page off to a fallback path that emits
markdown from the PDF's native text layer (no vision model required).
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


log = logging.getLogger("convertpdf.llm_retry")

T = TypeVar("T")


@dataclass(frozen=True)
class RetryConfig:
    """Bounded exponential-backoff retry policy.

    Defaults give 4 total attempts over ~7 seconds of total wait, which is
    well under the typical user-perceived "stuck" threshold for a CLI tool
    while still riding out a short provider outage.
    """

    max_attempts: int = 4
    initial_delay: float = 1.0
    backoff: float = 2.0
    max_delay: float = 30.0
    jitter: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.initial_delay < 0:
            raise ValueError("initial_delay must be >= 0")
        if self.backoff < 1.0:
            raise ValueError("backoff must be >= 1.0")
        if self.max_delay < self.initial_delay:
            raise ValueError("max_delay must be >= initial_delay")
        if not 0.0 <= self.jitter <= 1.0:
            raise ValueError("jitter must be in [0.0, 1.0]")


# Concrete transient exception types we always retry. ``APIStatusError`` is
# handled separately (only when the status code is 5xx; 4xx is permanent).
_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    RateLimitError,
)


def is_transient(exc: BaseException) -> bool:
    """Return True if ``exc`` represents a transient failure worth retrying.

    Permanent client errors (400/401/403/404/422) return False — retrying
    them produces an identical failure and wastes the budget.
    """
    if isinstance(exc, _TRANSIENT_EXCEPTIONS):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code >= 500:
        return True
    return False


def call_with_retry(
    fn: Callable[[], T],
    *,
    config: RetryConfig = RetryConfig(),
    label: str = "llm",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` with bounded exponential-backoff retry on transient failures.

    The caller passes a zero-arg callable so each attempt is a fresh call
    (no shared mutable state across attempts). Non-transient exceptions
    propagate immediately without sleeping.

    ``sleep`` is injectable for tests so we can assert retry counts without
    actually waiting.
    """
    delay = config.initial_delay
    last_exc: BaseException | None = None
    for attempt in range(1, config.max_attempts + 1):
        try:
            return fn()
        except BaseException as exc:
            if not is_transient(exc):
                raise
            last_exc = exc
            if attempt >= config.max_attempts:
                log.error(
                    "%s: giving up after %d attempt(s): %s: %s",
                    label,
                    attempt,
                    type(exc).__name__,
                    exc,
                )
                raise
            jittered = delay * (1.0 + random.uniform(-config.jitter, config.jitter))
            wait = max(0.0, min(jittered, config.max_delay))
            log.warning(
                "%s: transient %s on attempt %d/%d (%s); retrying in %.2fs",
                label,
                type(exc).__name__,
                attempt,
                config.max_attempts,
                exc,
                wait,
            )
            sleep(wait)
            delay = min(delay * config.backoff, config.max_delay)
    # Unreachable: the loop always returns or raises.
    assert last_exc is not None  # pragma: no cover
    raise last_exc  # pragma: no cover
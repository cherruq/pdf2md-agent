"""Retry helper for transient LLM-API failures.

Vision-model calls go through the OpenAI SDK against a custom ``base_url``
(MiniMax-M3). That endpoint, like any HTTP service, can return:

* network-level failures (timeout, connection refused, DNS hiccup)
* transient server errors (HTTP 5xx, gateway 502/503/504)
* rate-limit responses (HTTP 429)

We retry those with Fibonacci backoff + jitter, capped at a per-attempt
delay (15 minutes by default). With ``max_attempts=None`` transient failures
are retried indefinitely; permanent failures (authentication, bad request,
permission denied) are NOT retried — re-issuing the exact same request
just burns the budget and re-fails identically.

The runner wraps each per-page ``crew.kickoff()`` in :func:`call_with_retry`
and, on retry exhaustion, hands the page off to a fallback path that emits
markdown from the PDF's native text layer (no vision model required).
"""
from __future__ import annotations

import logging
import secrets
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Callable, TypeVar

# secrets.SystemRandom (vs random) so retry backoffs cannot sync across clients.
_RNG = secrets.SystemRandom()

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


def _safe_exc_summary(exc: BaseException) -> str:
    """Return a redacted summary of ``exc`` safe to write to logs.

    For ``APIStatusError`` we emit only the exception class name, HTTP
    status code, and ``str(exc)`` (which is the OpenAI SDK's own
    redacted message — it deliberately excludes ``exc.body``). This
    prevents provider response payloads (which can contain user
    content, internal stack traces, or other sensitive data) from
    landing in log files.
    """
    if isinstance(exc, APIStatusError):
        return f"{type(exc).__name__}: status={exc.status_code}: {exc}"
    return f"{type(exc).__name__}: {exc}"


log = logging.getLogger("pdf2md_agent.llm_retry")

T_co = TypeVar("T_co")


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Bounded Fibonacci-backoff retry policy.

    Defaults retry transient failures indefinitely (``max_attempts=None``)
    with a Fibonacci growth schedule capped at ``max_delay`` (15 minutes by
    default). The CLI's ``--max-retries`` flag accepts ``0`` as a synonym for
    unlimited; pass an explicit integer to bound the budget.

    ``initial_delay`` must be strictly positive: a zero value disables
    backoff entirely, and combined with ``max_attempts=None`` devolves into
    a busy-spin on transient failures.
    """

    max_attempts: int | None = None
    initial_delay: float = 1.0
    max_delay: float = 900.0
    jitter: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts is not None:
            if self.max_attempts < 1:
                raise ValueError(
                    "max_attempts must be None (unlimited) or >= 1; use 0 at the CLI/env boundary to mean unlimited"
                )
        if self.initial_delay <= 0:
            raise ValueError(
                "initial_delay must be > 0; a zero (or negative) value disables backoff "
                "and combined with max_attempts=None devolves into a busy-spin"
            )
        if self.max_delay < self.initial_delay:
            raise ValueError("max_delay must be >= initial_delay")
        if not 0.0 <= self.jitter <= 1.0:
            raise ValueError("jitter must be in [0.0, 1.0]")


def _fibonacci_multipliers() -> Iterator[int]:
    """Yield Fibonacci numbers 1, 1, 2, 3, 5, 8, ... ad infinitum.

    Used by :func:`call_with_retry` to scale each retry delay: each
    sleep = ``initial_delay * next(fibonacci)``, then capped at ``max_delay``.
    """
    a, b = 1, 1
    while True:
        yield a
        a, b = b, a + b


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
    fn: Callable[[], T_co],
    *,
    config: RetryConfig = RetryConfig(),
    label: str = "llm",
    sleep: Callable[[float], None] = time.sleep,
    timeout_seconds: float | None = None,
) -> T_co:
    """Call ``fn`` with Fibonacci-backoff retry on transient failures.

    The caller passes a zero-arg callable so each attempt is a fresh call
    (no shared mutable state across attempts). Non-transient exceptions
    propagate immediately without sleeping.

    ``sleep`` is injectable for tests so we can assert retry counts without
    actually waiting.

    ``timeout_seconds`` is a wall-clock guard for each attempt: when the
    call exceeds the budget, an :class:`APITimeoutError` is raised. The
    guard is layered on top of the SDK's own ``timeout`` so hangs inside
    crewAI's internal pipelines are also bounded.

    Per-retry sleeps grow by the Fibonacci sequence
    (1, 1, 2, 3, 5, 8, 13, ...) scaled by ``initial_delay`` and capped at
    ``max_delay``. With ``max_attempts=None`` (the default) transient
    failures are retried indefinitely; non-transient failures always
    propagate immediately, regardless of the cap.
    """
    bound = (
        str(config.max_attempts) if config.max_attempts is not None else "\u221e"
    )
    last_exc: Exception | None = None
    fib_multipliers = _fibonacci_multipliers()
    attempt = 0
    # Only exits: ``return`` (success) or ``raise`` (terminal transient
    # exhaustion / non-transient propagation). Infinite when max_attempts=None.
    while config.max_attempts is None or attempt < config.max_attempts:
        attempt += 1
        log.info(
            "%s: attempt %d/%s started",
            label,
            attempt,
            bound,
        )
        try:
            if timeout_seconds is None:
                return fn()
            return _call_with_timeout(fn, timeout_seconds)
        except _TimeoutCause as exc:
            _ = exc
            log.warning(
                "%s: attempt %d/%s timed out after %.1fs; treating as transient",
                label,
                attempt,
                bound,
                timeout_seconds,
            )
            last_exc = APITimeoutError(request=_dummy_request())
            if config.max_attempts is not None and attempt >= config.max_attempts:
                log.error(
                    "%s: giving up after %d attempt(s): %s",
                    label,
                    attempt,
                    _safe_exc_summary(last_exc),
                )
                raise last_exc
            wait = _compute_fibonacci_wait(config, next(fib_multipliers))
            log.info(
                "%s: retrying after transient %s on attempt %d/%s (%s); sleeping %.2fs",
                label,
                "Timeout",
                attempt,
                bound,
                _safe_exc_summary(last_exc),
                wait,
            )
            sleep(wait)
        except Exception as exc:  # noqa: BLE001 — predicate is `is_transient` below
            if not is_transient(exc):
                raise
            last_exc = exc
            if config.max_attempts is not None and attempt >= config.max_attempts:
                log.error(
                    "%s: giving up after %d attempt(s): %s",
                    label,
                    attempt,
                    _safe_exc_summary(exc),
                )
                raise
            wait = _compute_fibonacci_wait(config, next(fib_multipliers))
            log.info(
                "%s: retrying after transient %s on attempt %d/%s (%s); sleeping %.2fs",
                label,
                type(exc).__name__,
                attempt,
                bound,
                _safe_exc_summary(exc),
                wait,
            )
            sleep(wait)
    if last_exc is None:
        raise RuntimeError("unreachable: retry loop must set last_exc")
    raise last_exc  # pragma: no cover


def _compute_fibonacci_wait(config: RetryConfig, multiplier: int) -> float:
    uncapped = config.initial_delay * multiplier
    jittered = uncapped * (1.0 + _RNG.uniform(-config.jitter, config.jitter))
    return max(0.0, min(jittered, config.max_delay))


class _TimeoutCause(Exception):
    """Internal marker: distinguishes a timeout-guard hit from caller raises."""


def _call_with_timeout(
    fn: Callable[[], T_co],
    timeout_seconds: float,
) -> T_co:
    """Run ``fn()`` on a daemon thread; raise :class:`_TimeoutCause` on overrun.

    A previous implementation wrapped a one-shot
    :class:`concurrent.futures.ThreadPoolExecutor` in a ``with`` block. The
    block's ``__exit__`` calls ``executor.shutdown(wait=True)`` which
    blocks the caller until the worker thread completes — so when ``fn``
    hangs the timeout-guard raised its marker exception only **after** the
    hung call had already finished, defeating the whole point of the
    wall-clock guard. We now spawn a ``daemon=True`` thread per call and
    ``join(timeout=...)`` on it. The caller returns as soon as the timeout
    fires; the abandoned worker keeps running but is killed on process
    exit (daemon=True). The SDK's own ``timeout`` argument forwarded to
    the LLM call eventually unblocks the inner I/O, so the orphan
    thread is short-lived in practice.
    """
    import threading

    holder: list[object] = [None, None, False]

    def _runner() -> None:
        try:
            holder[0] = fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised by the joiner below
            holder[1] = exc
        finally:
            holder[2] = True

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    if not holder[2]:
        raise _TimeoutCause()
    if holder[1] is not None:
        raise holder[1]
    return holder[0]


def _dummy_request() -> object:
    import httpx

    return httpx.Request("GET", "https://example.test/")


__all__ = [
    "RetryConfig",
    "call_with_retry",
    "is_transient",
]
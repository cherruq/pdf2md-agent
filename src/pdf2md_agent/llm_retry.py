"""用于瞬时 LLM-API 故障的重试辅助工具。

视觉模型调用通过 OpenAI SDK 发送到自定义 ``base_url``
(MiniMax-M3)。与任何 HTTP 服务一样，该端点可能会返回：

* 网络级故障（超时、拒绝连接、DNS 异常）
* 瞬时服务器错误（HTTP 5xx、网关 502/503/504）
* 速率限制响应（HTTP 429）

我们使用斐波那契退避算法（Fibonacci backoff）+ 抖动（jitter）来重试这些请求，单次延迟有上限
（默认为 15 分钟）。如果 ``max_attempts=None``，则会无限期地重试瞬时故障；
而永久性故障（认证失败、错误请求、无权限）不会被重试 —— 重新发送完全相同的请求
只会白白消耗预算并得到同样的失败结果。

运行器（runner）将每页的 ``crew.kickoff()`` 包装在 :func:`call_with_retry` 中，
并且在重试耗尽后，会将该页面转交给降级/回退路径，该路径从 PDF 的原生文本层
生成 markdown（无需使用视觉模型）。

公共 API：:class:`RetryConfig`, :func:`is_transient`, :func:`call_with_retry`.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
import httpx
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Callable, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


# 使用 ``secrets.SystemRandom``（而不是 ``random``），这样重试退避不会在
# 并行的客户端之间同步。
_RNG = secrets.SystemRandom()

T_co = TypeVar("T_co")

log = logging.getLogger("pdf2md_agent.llm_retry")


# --- Exception classification ----------------------------------------------


def _safe_exc_summary(exc: BaseException) -> str:
    """返回可以安全写入日志的已删减的 ``exc`` 摘要。

    对于 ``APIStatusError``，我们仅输出异常类名、HTTP 状态码和
    ``str(exc)``（这是 OpenAI SDK 自带的经过删减的信息 —— 它刻意排除了 ``exc.body``）。
    这可以防止提供商响应的负载（可能包含用户内容、内部堆栈跟踪或其他敏感数据）
    落入日志文件中。
    """
    if isinstance(exc, APIStatusError):
        return f"{type(exc).__name__}: status={exc.status_code}: {exc}"
    return f"{type(exc).__name__}: {exc}"


_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    RateLimitError,
)
"""我们总是会重试的具体瞬时异常类型。``APIStatusError``
会单独处理（仅当状态码为 5xx 时重试；4xx 是永久性的）。"""


def is_transient(exc: BaseException) -> bool:
    """如果 ``exc`` 表示值得重试的瞬时故障，则返回 True。

    永久性客户端错误（400/401/403/404/422）返回 False —— 重试
    它们会产生相同的失败结果并浪费预算。
    """
    if isinstance(exc, _TRANSIENT_EXCEPTIONS):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code >= 500:
        return True
    return False


# --- Retry configuration ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """有界斐波那契退避（Fibonacci-backoff）重试策略。

    默认无限期重试瞬时故障（``max_attempts=None``），并以斐波那契增长计划作为退避策略，
    上限为 ``max_delay``（默认为 15 分钟）。CLI 的 ``--max-retries`` 标志接受 ``0`` 作为
    无限次数的同义词；传入一个明确的整数来限制预算。

    ``initial_delay`` 必须严格为正数：零值会完全禁用退避算法，并且如果结合
    ``max_attempts=None``，会在发生瞬时故障时退化为忙等待循环（busy-spin）。
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


# --- Backoff math ----------------------------------------------------------


def _fibonacci_multipliers() -> Iterator[int]:
    """无限生成斐波那契数列 1, 1, 2, 3, 5, 8, ...

    被 :func:`call_with_retry` 用于缩放每个重试延迟：每次
    sleep = ``initial_delay * next(fibonacci)``，然后受 ``max_delay`` 的限制。
    """
    a, b = 1, 1
    while True:
        yield a
        a, b = b, a + b


def _compute_fibonacci_wait(config: RetryConfig, multiplier: int) -> float:
    """计算单个具有上限和抖动的退避延迟。

    ``uncapped`` 是按斐波那契缩放的延迟；``jittered`` 对其加上 ``±jitter`` 抖动扰动；
    结果被限制在 ``[0, max_delay]`` 的范围内。
    """
    uncapped = config.initial_delay * multiplier
    jittered = uncapped * (1.0 + _RNG.uniform(-config.jitter, config.jitter))
    return max(0.0, min(jittered, config.max_delay))


# --- Timeout guard ---------------------------------------------------------


class _TimeoutCause(Exception):
    """内部标记：用于区分触发超时保护还是调用者抛出异常。"""


def _dummy_request() -> object:
    """用于满足 ``APITimeoutError(request=…)`` 参数要求的占位符 httpx request。"""
    return httpx.Request("GET", "https://example.test/")


def _call_with_timeout(
    fn: Callable[[], T_co],
    timeout_seconds: float,
) -> T_co:
    """在守护线程中运行 ``fn()``；若超时，抛出 :class:`_TimeoutCause` 异常。

    早期的实现是将一次性的 :class:`concurrent.futures.ThreadPoolExecutor` 包装
    在 ``with`` 块中。该块的 ``__exit__`` 会调用 ``executor.shutdown(wait=True)``，这会
    阻塞调用者直到工作线程完成 —— 因此，当 ``fn`` 发生挂起时，超时保护机制
    只有在挂起的调用已经完成 **之后** 才会抛出其标记异常，完全失去了强制
    挂钟（wall-clock）超时的初衷。现在我们每次调用都会生成一个 ``daemon=True`` 线程，
    并对其调用 ``join(timeout=...)``。只要超时触发，调用者就会立即返回；被遗弃的
    工作线程会继续运行，但会在进程退出时被杀掉（daemon=True）。传递给 LLM 调用的 SDK
    自带的 ``timeout`` 参数最终会解除内部 I/O 的阻塞状态，所以在实际情况中，
    孤儿线程往往是短暂存在的。
    """
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


# --- Public entry point ----------------------------------------------------


def call_with_retry(
    fn: Callable[[], T_co],
    *,
    config: RetryConfig = RetryConfig(),
    label: str = "llm",
    sleep: Callable[[float], None] = time.sleep,
    timeout_seconds: float | None = None,
) -> T_co:
    """在遇到瞬时故障时，调用 ``fn`` 并使用斐波那契退避重试。

    调用者需传入无参的可调用对象，这样每次尝试都是一次全新的调用
    （每次尝试之间不共享可变状态）。非瞬时异常会立即抛出而不会等待。

    ``sleep`` 支持为了测试而注入，以便我们无需实际等待即可断言重试次数。

    ``timeout_seconds`` 是针对每次尝试的挂钟（wall-clock）保护：当调用
    超过设定时间时，抛出 :class:`APITimeoutError` 异常。该保护机制叠加
    在 SDK 自身的 ``timeout`` 之上，因此 crewAI 内部管道的挂起也会受到限制。

    每次重试的睡眠时间按斐波那契数列（1, 1, 2, 3, 5, 8, 13, ...）以
    ``initial_delay`` 进行缩放，最大不超过 ``max_delay``。在设置 ``max_attempts=None``
    （默认值）时，瞬时故障将被无限重试；而非瞬时故障，无论上限设为多少，都会立刻抛出异常。
    """
    bound = str(config.max_attempts) if config.max_attempts is not None else "\u221e"
    last_exc: Exception | None = None
    fib_multipliers = _fibonacci_multipliers()
    attempt = 0
    while config.max_attempts is None or attempt < config.max_attempts:
        attempt += 1
        log.info("%s: attempt %d/%s started", label, attempt, bound)
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
            _sleep_and_continue(label, attempt, bound, last_exc, config, fib_multipliers, sleep)
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
            _sleep_and_continue(label, attempt, bound, exc, config, fib_multipliers, sleep)
    if last_exc is None:
        raise RuntimeError("unreachable: retry loop must set last_exc")
    raise last_exc  # pragma: no cover


def _sleep_and_continue(
    label: str,
    attempt: int,
    bound: str,
    exc: BaseException,
    config: RetryConfig,
    fib_multipliers: Iterator[int],
    sleep: Callable[[float], None],
) -> None:
    """记录瞬时重试的日志，进行休眠并继续循环。"""
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


__all__ = [
    "RetryConfig",
    "call_with_retry",
    "is_transient",
]

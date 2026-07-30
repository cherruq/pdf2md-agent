"""Project configuration loaded from the environment at import time.

``config.py`` is split into four concern-grouped sections:

1. **Env helpers** (:func:`_env`, :func:`_env_int`, :func:`_env_float`,
   :func:`_env_bool`, :func:`_env_int_or_unlimited`,
   :func:`_env_positive_float`) — strict, typed accessors with clear
   error messages.
2. **Endpoint + auth** (``OPENAI_BASE_URL``, ``MODEL_NAME``,
   :func:`require_api_key`) — the OpenAI-compatible base URL, the
   configured model, and the API-key guard.
3. **Token budget + image downscale** (``TOKEN_BUDGET_SAFETY``,
   ``IMAGE_LONG_SIDE``, ``IMAGE_JPEG_QUALITY``, ``IMAGE_MIN_LONG_SIDE``,
   :func:`resolve_ctx_limit`) — knobs the
   per-call planner reads.
4. **LLM retry + fallback** (``RETRY_*``, ``FALLBACK_TO_TEXT``) —
   knobs the retry/backoff layer reads.

``load_dotenv()`` runs at import time; any module that imports
``config`` transitively will read ``.env`` exactly once.
"""
from __future__ import annotations

import functools
import logging
import os
from typing import Final

from dotenv import load_dotenv

from pdf2md_agent.ctx_probe import probe_ctx_limit


log = logging.getLogger("pdf2md_agent.config")

load_dotenv()


# --- Env helpers ------------------------------------------------------------


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, default)
    return value.strip() if value else default


# Only essential environment helper utilities retained for core config items.


def _env_int_or_unlimited(name: str) -> int | None:
    """Read an integer-or-unlimited env knob.

    Empty/unset → ``None`` (default = unlimited retries); ``"0"`` → ``None``
    (explicit unlimited); positive integers → bounded attempt count.
    Negative integers and non-numeric strings raise.
    """
    raw = _env(name)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value == 0:
        return None
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r} (use 0 for unlimited)")
    return value


def _env_positive_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return value


# --- Endpoint + auth --------------------------------------------------------


OPENAI_BASE_URL: Final[str] = _env("OPENAI_BASE_URL", "https://api.minimaxi.com/v1")
MODEL_NAME: Final[str] = _env("PDF2MD_AGENT_MODEL", "MiniMax-M3")


def require_api_key() -> str:
    """Return the OpenAI API key from the environment, or raise with guidance."""
    value = _env("OPENAI_API_KEY")
    if not value:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return value


# --- Token budget + image downscale -----------------------------------------

# ``resolve_ctx_limit`` consults env → ``/v1/models`` probe → hardcoded
# default; the 0.85 safety margin keeps us off the cliff edge while a
# paginate is in flight.

_MAX_CTX_LIMIT: Final[int] = 1_048_576  # 1M, the published MiniMax-M3 ceiling.
_DEFAULT_CTX_LIMIT: Final[int] = 128_000  # safe fallback for unrecognised models.

# Override by setting ``PDF2MD_AGENT_CTX_LIMIT`` or letting the runtime
# probe succeed against ``OPENAI_BASE_URL``.
_HARD_CODED_CTX_LIMITS: Final[dict[str, int]] = {
    "MiniMax-M3": 524_288,  # 512K, the published guarantee.
    "MiniMax-Text-01": 1_000_000,  # 1M ceiling, per the MSA spec sheet.
    "MiniMax-VL-01": 524_288,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "claude-3-5-sonnet-latest": 200_000,
    "claude-3-opus-latest": 200_000,
}


@functools.lru_cache(maxsize=1)
def resolve_ctx_limit() -> int:
    """Resolve the model's context-window token budget.

    Priority, highest first:

    1. ``PDF2MD_AGENT_CTX_LIMIT`` env var (positive int)
    2. ``probe_ctx_limit(OPENAI_BASE_URL, api_key, MODEL_NAME)`` clamped to
       ``_MAX_CTX_LIMIT``; the probe is skipped entirely if
       ``OPENAI_API_KEY`` is unset
    3. Hardcoded default for the active ``MODEL_NAME``; falls back to
       ``_DEFAULT_CTX_LIMIT`` if the model is unknown

    Result is cached at module level; tests clear the cache via
    ``resolve_ctx_limit.cache_clear()``.
    """
    raw = _env("PDF2MD_AGENT_CTX_LIMIT")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                log.info(
                    "ctx_limit: %d (from PDF2MD_AGENT_CTX_LIMIT env var)", value
                )
                return value
        except ValueError:
            pass  # fall through to probe; the caller should fix the typo

    api_key = _env("OPENAI_API_KEY")
    if api_key:
        probed = probe_ctx_limit(OPENAI_BASE_URL, api_key, MODEL_NAME)
        if probed is not None and probed > 0:
            clamped = min(probed, _MAX_CTX_LIMIT)
            log.info(
                "ctx_limit: %d (probed from %s/models for %s)",
                clamped, OPENAI_BASE_URL, MODEL_NAME,
            )
            return clamped

    hardcoded = _HARD_CODED_CTX_LIMITS.get(MODEL_NAME)
    if hardcoded is not None:
        log.info("ctx_limit: %d (hardcoded for %s)", hardcoded, MODEL_NAME)
        return hardcoded

    log.warning(
        "ctx_limit: %d (generic fallback; model %r is unknown — "
        "set PDF2MD_AGENT_CTX_LIMIT to silence this warning)",
        _DEFAULT_CTX_LIMIT, MODEL_NAME,
    )
    return _DEFAULT_CTX_LIMIT


TOKEN_BUDGET_SAFETY: Final[float] = 0.85
IMAGE_LONG_SIDE: Final[int] = 1536
IMAGE_JPEG_QUALITY: Final[int] = 85
IMAGE_MIN_LONG_SIDE: Final[int] = 768

REQUEST_TIMEOUT_SECONDS: Final[float] = _env_positive_float(
    "PDF2MD_AGENT_REQUEST_TIMEOUT", 60.0
)


# --- LLM retry + fallback ---------------------------------------------------
# Defaults: unlimited transient retries with Fibonacci backoff.
# Set PDF2MD_AGENT_MAX_RETRIES (or pass --max-retries) to a positive integer to bound the budget.

RETRY_MAX_ATTEMPTS: Final[int | None] = _env_int_or_unlimited(
    "PDF2MD_AGENT_MAX_RETRIES"
)
RETRY_INITIAL_DELAY: Final[float] = 1.0
RETRY_MAX_DELAY: Final[float] = 900.0
RETRY_JITTER: Final[float] = 0.25
FALLBACK_TO_TEXT: Final[bool] = True


__all__ = [
    "FALLBACK_TO_TEXT",
    "IMAGE_JPEG_QUALITY",
    "IMAGE_LONG_SIDE",
    "IMAGE_MIN_LONG_SIDE",
    "MODEL_NAME",
    "OPENAI_BASE_URL",
    "REQUEST_TIMEOUT_SECONDS",
    "RETRY_INITIAL_DELAY",
    "RETRY_JITTER",
    "RETRY_MAX_ATTEMPTS",
    "RETRY_MAX_DELAY",
    "TOKEN_BUDGET_SAFETY",
    "probe_ctx_limit",
    "require_api_key",
    "resolve_ctx_limit",
]
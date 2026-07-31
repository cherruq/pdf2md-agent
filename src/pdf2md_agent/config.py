"""Project configuration loaded from environment variables at import time."""
from __future__ import annotations

import functools
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

from dotenv import load_dotenv

from pdf2md_agent.tuning import (  # noqa: F401 — re-exported in __all__ for backward compatibility
    DEFAULT_IMAGE_JPEG_QUALITY,
    DEFAULT_IMAGE_LONG_SIDE,
    DEFAULT_STITCH_MODE,
    FALLBACK_TO_TEXT,
    IMAGE_MIN_LONG_SIDE,
    TOKEN_BUDGET_SAFETY_DEFAULT,
)

if TYPE_CHECKING:
    from pdf2md_agent.cache import CacheLayout, CacheNoCacheFlags
    from pdf2md_agent.llm_retry import RetryConfig


log = logging.getLogger("pdf2md_agent.config")

load_dotenv()


# --- Env helpers ------------------------------------------------------------


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, default)
    return value.strip() if value else default


def _env_int_or_unlimited(name: str) -> int | None:
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


_DEFAULT_CTX_LIMIT: Final[int] = 128_000

_HARD_CODED_CTX_LIMITS: Final[dict[str, int]] = {
    "MiniMax-M3": 524_288,
    "MiniMax-Text-01": 1_000_000,
    "MiniMax-VL-01": 524_288,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "claude-3-5-sonnet-latest": 200_000,
    "claude-3-opus-latest": 200_000,
}


@functools.lru_cache(maxsize=1)
def resolve_ctx_limit() -> int:
    """Resolve the model's context-window token budget."""
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
            pass

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


TOKEN_BUDGET_SAFETY: Final[float] = TOKEN_BUDGET_SAFETY_DEFAULT
IMAGE_LONG_SIDE: Final[int] = DEFAULT_IMAGE_LONG_SIDE
IMAGE_JPEG_QUALITY: Final[int] = DEFAULT_IMAGE_JPEG_QUALITY

REQUEST_TIMEOUT_SECONDS: Final[float] = _env_positive_float(
    "PDF2MD_AGENT_REQUEST_TIMEOUT", 60.0
)


# --- LLM retry + fallback ---------------------------------------------------


RETRY_MAX_ATTEMPTS: Final[int | None] = _env_int_or_unlimited(
    "PDF2MD_AGENT_MAX_RETRIES"
)
RETRY_INITIAL_DELAY: Final[float] = 1.0
RETRY_MAX_DELAY: Final[float] = 900.0
RETRY_JITTER: Final[float] = 0.25


# --- Job Configuration Struct -----------------------------------------------


@dataclass(slots=True, frozen=True)
class ConversionConfig:
    """Immutable configuration struct for a single PDF-to-Markdown conversion job."""

    pdf: Path
    output: Path
    dpi: int
    layout: CacheLayout
    render_target: Path
    resolved_pages: list[int] | None
    no_cache: CacheNoCacheFlags
    retry_config: RetryConfig
    text_hint: bool
    image_long_side: int
    image_jpeg_quality: int
    ctx_limit: int
    request_timeout_seconds: float | None
    stitch_mode: str = DEFAULT_STITCH_MODE
    fallback_to_text: bool = FALLBACK_TO_TEXT
    started: float = field(default_factory=time.monotonic)


__all__ = [
    "ConversionConfig",
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
    "require_api_key",
    "resolve_ctx_limit",
]
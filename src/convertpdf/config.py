"""Project configuration loaded from the environment at import time."""
from __future__ import annotations

import os
from typing import Final

from dotenv import load_dotenv


load_dotenv()


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, default)
    return value.strip() if value else default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean (1/0/true/false/yes/no/on/off), got {raw!r}")


OPENAI_BASE_URL: Final[str] = _env("OPENAI_BASE_URL", "https://api.minimaxi.com/v1")
MODEL_NAME: Final[str] = _env("CONVERTPDF_MODEL", "MiniMax-M3")


# --- Token-budget / image-downscale knobs -----------------------------------
# MiniMax-M3 rejects payloads over ~2013 tokens; the 0.85 safety margin
# keeps us off the cliff edge while a paginate is in flight.

CTX_LIMIT: Final[int] = _env_int("CONVERTPDF_CTX_LIMIT", 2013)
TOKEN_BUDGET_SAFETY: Final[float] = _env_float("CONVERTPDF_TOKEN_BUDGET_SAFETY", 0.85)
IMAGE_LONG_SIDE: Final[int] = _env_int("CONVERTPDF_IMAGE_LONG_SIDE", 1536)
IMAGE_JPEG_QUALITY: Final[int] = _env_int("CONVERTPDF_IMAGE_JPEG_QUALITY", 85)
IMAGE_MIN_LONG_SIDE: Final[int] = _env_int("CONVERTPDF_IMAGE_MIN_LONG_SIDE", 768)
MAX_SUMMARY_CHARS: Final[int] = _env_int("CONVERTPDF_MAX_SUMMARY_CHARS", 800)


# --- LLM retry / fallback knobs ---------------------------------------------
# Defaults: 4 total attempts (1 initial + 3 retries) over ~7s of total wait
# before giving up on a page and falling back to text-layer markdown.

RETRY_MAX_ATTEMPTS: Final[int] = _env_int("CONVERTPDF_MAX_RETRIES", 4)
RETRY_INITIAL_DELAY: Final[float] = _env_float("CONVERTPDF_RETRY_INITIAL_DELAY", 1.0)
RETRY_BACKOFF: Final[float] = _env_float("CONVERTPDF_RETRY_BACKOFF", 2.0)
RETRY_MAX_DELAY: Final[float] = _env_float("CONVERTPDF_RETRY_MAX_DELAY", 30.0)
RETRY_JITTER: Final[float] = _env_float("CONVERTPDF_RETRY_JITTER", 0.25)
FALLBACK_TO_TEXT: Final[bool] = _env_bool("CONVERTPDF_FALLBACK_TO_TEXT", True)


def require_api_key() -> str:
    """Return the OpenAI API key from the environment, or raise with guidance."""
    value = _env("OPENAI_API_KEY")
    if not value:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return value
"""Project configuration loaded from the environment at import time."""
from __future__ import annotations

import os
from typing import Final

from dotenv import load_dotenv


load_dotenv()


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, default)
    return value.strip() if value else default


OPENAI_BASE_URL: Final[str] = _env("OPENAI_BASE_URL", "https://api.minimaxi.com/v1")
MODEL_NAME: Final[str] = _env("CONVERTPDF_MODEL", "MiniMax-M3")


def require_api_key() -> str:
    """Return the OpenAI API key from the environment, or raise with guidance."""
    value = _env("OPENAI_API_KEY")
    if not value:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return value
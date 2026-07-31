"""Internal tuning policy and static default constants for pipeline optimization."""

from __future__ import annotations

from typing import Final


# Default stitch mode for joining Markdown across pages ("heuristic" or "off")
DEFAULT_STITCH_MODE: Final[str] = "heuristic"

# Whether to fall back to native PDF text layer when AI extraction retries fail
FALLBACK_TO_TEXT: Final[bool] = True

# Token budget safety margin (percentage of context limit reserved for safety)
TOKEN_BUDGET_SAFETY_DEFAULT: Final[float] = 0.85

# Minimum long-side pixel limit allowed during iterative downscaling
IMAGE_MIN_LONG_SIDE: Final[int] = 768

# Default rendering DPI for PDF page rasterization
DEFAULT_DPI: Final[int] = 144

# Default image constraints
DEFAULT_IMAGE_LONG_SIDE: Final[int] = 1536
DEFAULT_IMAGE_JPEG_QUALITY: Final[int] = 85

# AI extraction coverage reflection parameters
REFLECTION_COVERAGE_THRESHOLD: Final[float] = 0.90
REFLECTION_MAX_ATTEMPTS: Final[int] = 2
PENALTY_PROMPT: Final[str] = (
    "\n\nCRITICAL WARNING: Your previous output missed significant portions "
    "of the native text. You MUST preserve ALL text. Please re-read the "
    "page carefully and transcribe completely."
)


__all__ = [
    "DEFAULT_DPI",
    "DEFAULT_IMAGE_JPEG_QUALITY",
    "DEFAULT_IMAGE_LONG_SIDE",
    "DEFAULT_STITCH_MODE",
    "FALLBACK_TO_TEXT",
    "IMAGE_MIN_LONG_SIDE",
    "PENALTY_PROMPT",
    "REFLECTION_COVERAGE_THRESHOLD",
    "REFLECTION_MAX_ATTEMPTS",
    "TOKEN_BUDGET_SAFETY_DEFAULT",
]

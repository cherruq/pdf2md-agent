"""Token-count estimator for prompt text and inlined page images.

Two pure functions, no I/O beyond ``Path.stat``:

* :func:`estimate_text_tokens` — coarse heuristic over CJK / non-CJK runs
  used to budget persona + per-page prompt cost before the LLM call.
* :func:`estimate_image_tokens` — converts a file size or byte buffer to
  its estimated base64 token cost. Pixels are *never* decoded; the
  estimator only reads ``Path.stat().st_size``.

All estimators are deliberately conservative — they never call any
external tokenizer (``tiktoken`` is forbidden by the project
guidelines) and they over-estimate so a budget-passing call is
guaranteed to fit in practice.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Final, Union

log = logging.getLogger("pdf2md_agent.token_estimator")

# 3.5 is empirically a safe upper bound observed in 400-response logs;
# the base64↔token mapping is opaque across providers, so over-estimate.
_IMAGE_BYTES_PER_TOKEN: Final[float] = 3.5

# Number of CJK / wide characters per token. Mixed CJK + Latin prose
# averages to roughly one token per ~1.5 chars; we use the conservative
# 1/3 ratio so pages heavy in Chinese text do not under-budget.
_CJK_CHARS_PER_TOKEN: Final[float] = 3.0

# Latin ratio is closer to 1 token per 4 chars; pairs of quotes/punctuation
# inflate this a bit but the heuristic is intentionally coarse.
_ASCII_CHARS_PER_TOKEN: Final[float] = 4.0

PathOrBytes = Union[str, Path, bytes, bytearray]


def estimate_text_tokens(s: str) -> int:
    """Estimate token cost of a text prompt using a mixed CJK/ASCII heuristic.

    Splits the input into CJK-runs (treated at 1 token per 3 chars) and ASCII
    runs (1 token per 4 chars), then sums both halves. The estimate is
    deliberately coarse — its purpose is budget *planning*, not exact billing.

    Args:
        s: The prompt text whose token cost we want to budget for.

    Returns:
        Estimated number of tokens as an ``int`` (always >= 0).
    """
    if not s:
        return 0

    cjk_chars = 0
    ascii_chars = 0
    for ch in s:
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x20000 <= code <= 0x2A6DF
            or 0xF900 <= code <= 0xFAFF
            or 0xFF00 <= code <= 0xFFEF
        ):
            cjk_chars += 1
        else:
            ascii_chars += 1

    cjk_tokens = math.ceil(cjk_chars / _CJK_CHARS_PER_TOKEN)
    ascii_tokens = math.ceil(ascii_chars / _ASCII_CHARS_PER_TOKEN)
    return cjk_tokens + ascii_tokens


def estimate_image_tokens(path_or_bytes: PathOrBytes, *, mime: str = "image/jpeg") -> int:
    """Estimate token cost of inlining an image as a base64 data URL.

    Only ``Path.stat().st_size`` is consulted — pixels are *not* decoded. The
    estimator pretends every byte ends up in a base64 string of length
    ``ceil(N/3) * 4`` and that each token covers ~3.5 base64 chars. This
    over-estimates compared to the model's actual rate but matches the
    behaviour reported by ``400 context window exceeds limit`` errors.

    Args:
        path_or_bytes: A local file ``Path``/``str`` or the raw ``bytes`` of
            an image. ``http(s)://`` URLs are not supported here — the
            caller should have downloaded them already.
        mime: Unused for the bytes-only estimator; accepted for API
            symmetry with future Pillow-aware estimators.

    Returns:
        Estimated number of tokens as an ``int``.
    """
    del mime
    if isinstance(path_or_bytes, (str, Path)):
        path = Path(path_or_bytes)
        if not path.is_file():
            log.debug("estimate_image_tokens: %s is not a file; assuming 0", path)
            return 0
        size = path.stat().st_size
    elif isinstance(path_or_bytes, (bytes, bytearray)):
        size = len(path_or_bytes)
    else:
        raise TypeError(
            f"estimate_image_tokens: unsupported type {type(path_or_bytes).__name__}"
        )

    b64_chars = ((size + 2) // 3) * 4
    return max(1, math.ceil(b64_chars / _IMAGE_BYTES_PER_TOKEN))


__all__ = [
    "PathOrBytes",
    "estimate_image_tokens",
    "estimate_text_tokens",
]
"""Estimate text + image token costs and plan per-page image downscale.

The vision API in use (``MiniMax-M3`` via ``api.minimaxi.com``) rejects payloads
whose total token count exceeds the configured context window. To keep each
call under the budget we need (a) a cheap text/image token estimator and
(b) a planner that picks the smallest downscale ``long_side`` for a page
PNG that still fits the budget.

All estimators are deliberately conservative. They never call any external
tokenizer (``tiktoken`` is forbidden by the project guidelines) and the
image estimator only reads ``Path.stat().st_size`` — it does not decode
pixel data. ``plan_for_image`` is the one place that opens the image, and
only to learn its original dimensions for the binary search.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Union

log = logging.getLogger("convertpdf.token_budget")

# Heuristic bytes-per-token ratio used by estimate_image_tokens. The model in
# use (~2k token context) allocates a fixed budget per image patch, but the
# exact mapping between base64-encoded image bytes and tokens is opaque. The
# value 3.5 is empirically a safe upper bound observed in 400-response logs.
_IMAGE_BYTES_PER_TOKEN: Final[float] = 3.5

# Number of CJK / wide characters per token. Mixed CJK + Latin prose averages
# to roughly one token per ~1.5 chars; we use the conservative 1/3 ratio so
# pages heavy in Chinese text do not under-budget.
_CJK_CHARS_PER_TOKEN: Final[float] = 3.0

# Latin ratio is closer to 1 token per 4 chars; pairs of quotes/punctuation
# inflate this a bit but the heuristic is intentionally coarse.
_ASCII_CHARS_PER_TOKEN: Final[float] = 4.0

PathOrBytes = Union[str, Path, bytes]


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
        # CJK Unified Ideographs + common extensions + fullwidth forms.
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
    del mime  # currently unused; kept for signature stability
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

    # base64 inflates by 4/3; round up to a full 4-char group.
    b64_chars = ((size + 2) // 3) * 4
    return max(1, math.ceil(b64_chars / _IMAGE_BYTES_PER_TOKEN))


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    """Result of budgeting one extract call.

    Attributes:
        total: persona + fixed_text + image tokens at the chosen
            ``needed_long_side``. Sized to fit within the safety margin.
        limit: ``ctx_limit * TOKEN_BUDGET_SAFETY`` (rounded down).
        fits: ``True`` if the chosen plan keeps the call under ``limit``.
            ``False`` means even the minimum allowed downscale plus the
            fixed-text budget would still exceed ``limit`` — the caller is
            expected to log a warning and proceed at the smallest size.
        needed_long_side: Long-side length (in pixels) that the page image
            should be downscaled to before inlining. Always set: if the
            image already fits, this still defaults to the requested
            ``target_long_side`` so runs stay predictable.
        reason: Short human-readable explanation of the decision (logging).
    """

    total: int
    limit: int
    fits: bool
    needed_long_side: int
    reason: str


def _b64_chars(size_bytes: int) -> int:
    """Return the length of a base64 string encoding ``size_bytes`` raw bytes."""
    return ((size_bytes + 2) // 3) * 4


def _tokens_for_size(size_bytes: int) -> int:
    """Convert a raw byte count to its estimated base64 token cost."""
    return max(1, math.ceil(_b64_chars(size_bytes) / _IMAGE_BYTES_PER_TOKEN))


def _est_size_at_long_side(
    orig_bytes: int,
    orig_long_side: int,
    target_long_side: int,
) -> int:
    """Crude pixel-area model: bytes scale with ``(target/orig_long_side)²``.

    JPEG files do not strictly obey this, but the bias is conservative in
    both directions — text-heavy pages compress to roughly the PNG-equivalent
    size, while photo pages compress a bit more. Good enough for picking a
    long_side; the LLM only sees the final bytes anyway.
    """
    if orig_long_side <= 0 or orig_bytes <= 0:
        return orig_bytes
    scale_sq = (target_long_side / orig_long_side) ** 2
    return max(1024, int(orig_bytes * scale_sq))


def _open_for_size(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` of ``path`` using Pillow.

    Falls back to (1, 1) if Pillow cannot open the file so the rest of the
    planner still produces a sane (if useless) result instead of raising
    into the pipeline.
    """
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - Pillow is a hard project dep
        log.warning("Pillow not importable in plan_for_image; using 1x1 fallback")
        return 1, 1
    try:
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception as exc:
        log.warning("plan_for_image: cannot open %s to read size (%s)", path, exc)
        return 1, 1


def plan_for_image(
    ctx_limit: int,
    *,
    persona_tokens: int,
    fixed_text_tokens: int,
    image_path: Path,
    target_long_side: int = 1536,
    min_long_side: int = 768,
    jpeg_quality: int = 85,
    safety: float = 0.85,
) -> BudgetDecision:
    """Plan a per-page image resize that keeps the extract call under budget.

    The decision is the **largest** ``long_side`` in
    ``[min_long_side, original_long_side]`` whose *estimated* image tokens
    still keep the call total under ``ctx_limit * safety`` (i.e. the least
    aggressive downscale that fits). If the original image already fits,
    ``needed_long_side`` is still reported as the configured
    ``target_long_side`` for runtime consistency.

    Args:
        ctx_limit: Hard context-window token limit reported by the model
            (e.g. ``2013``).
        persona_tokens: Estimated tokens for the agent persona + task
            system prompt (already pre-computed by the caller).
        fixed_text_tokens: Estimated tokens for the per-page variables:
            running summary, optional text-hint, the rendered task
            description scaffold.
        image_path: Local path to the page PNG.
        target_long_side: Desired long-side after downscaling when budget
            allows; this is the "happy path" size.
        min_long_side: Lower bound for the binary search — never recommend
            a resize smaller than this (OCR legibility floor).
        jpeg_quality: Currently unused in the estimator; reserved for a
            future bytes-per-pixel calibration pass.
        safety: Fraction of ``ctx_limit`` we are willing to spend; must be
            in ``(0.0, 1.0]``.

    Returns:
        A :class:`BudgetDecision` with ``fits`` reflecting whether the
        chosen long_side keeps the call under the safety limit.
    """
    del jpeg_quality  # reserved; see docstring
    if not (0.0 < safety <= 1.0):
        raise ValueError(f"safety must be in (0, 1], got {safety!r}")
    if target_long_side < min_long_side:
        raise ValueError(
            f"target_long_side={target_long_side} < min_long_side={min_long_side}"
        )

    limit = int(ctx_limit * safety)
    budget_for_image = max(0, limit - persona_tokens - fixed_text_tokens)
    current_tokens = estimate_image_tokens(image_path)
    original_bytes = (
        image_path.stat().st_size if image_path.is_file() else 0
    )

    if current_tokens <= budget_for_image:
        # Already fits — recommend the standard target size for consistency.
        total = persona_tokens + fixed_text_tokens + current_tokens
        return BudgetDecision(
            total=total,
            limit=limit,
            fits=True,
            needed_long_side=target_long_side,
            reason="image fits within budget at original size",
        )

    # Need to downscale. Find the LARGEST long_side whose estimated tokens
    # stay under budget_for_image, via integer binary search.
    orig_w, orig_h = _open_for_size(image_path)
    orig_long_side = max(orig_w, orig_h)

    if orig_long_side <= min_long_side:
        # Image is already smaller than the OCR legibility floor; nothing we
        # can do besides report that it doesn't fit.
        total = persona_tokens + fixed_text_tokens + current_tokens
        return BudgetDecision(
            total=total,
            limit=limit,
            fits=False,
            needed_long_side=target_long_side,
            reason=(
                f"image already smaller than min_long_side={min_long_side}; "
                "fixed-text budget is too tight"
            ),
        )

    upper = orig_long_side
    # Binary search for the *largest* long_side whose estimated tokens still
    # fit ``budget_for_image``. Tokens grow with long_side², so we move the
    # "low" bound up when mid fits (try bigger) and the "high" bound down
    # when it does not (try smaller). Tolerance of 4 px prevents jitter when
    # neighbouring sizes round to the same token estimate.
    best = min_long_side
    low, high = min_long_side, upper
    while high - low > 4:
        mid = (low + high) // 2
        est_bytes = _est_size_at_long_side(original_bytes, upper, mid)
        est_tokens = _tokens_for_size(est_bytes)
        if est_tokens <= budget_for_image:
            best = mid
            low = mid
        else:
            high = mid

    if best < upper:
        est_bytes = _est_size_at_long_side(original_bytes, upper, best)
        est_tokens = _tokens_for_size(est_bytes)
        total = persona_tokens + fixed_text_tokens + est_tokens
        return BudgetDecision(
            total=total,
            limit=limit,
            fits=total <= limit,
            needed_long_side=best,
            reason=(
                f"downscaled from {upper}px to {best}px to fit "
                f"budget={budget_for_image} image tokens"
            ),
        )

    # Binary search exhausted without finding a smaller size that fits.
    est_bytes = _est_size_at_long_side(original_bytes, upper, target_long_side)
    est_tokens = _tokens_for_size(est_bytes)
    total = persona_tokens + fixed_text_tokens + est_tokens
    return BudgetDecision(
        total=total,
        limit=limit,
        fits=total <= limit,
        needed_long_side=target_long_side,
        reason="binary search exhausted budget headroom; using target_long_side",
    )


__all__ = [
    "BudgetDecision",
    "estimate_image_tokens",
    "estimate_text_tokens",
    "plan_for_image",
]

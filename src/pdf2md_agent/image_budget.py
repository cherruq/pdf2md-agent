"""Per-page image budget planner.

The vision API rejects payloads whose total token count exceeds the
configured context window. :func:`plan_for_image` picks the smallest
downscale ``long_side`` for a page PNG that still keeps the call under
``ctx_limit * safety`` — the largest size that fits, so OCR fidelity is
preserved wherever possible.

Implementation lives in two layers:

* :class:`BudgetDecision` — the typed result, frozen dataclass.
* :func:`plan_for_image` — the entry point. Internally uses
  :func:`_open_for_size` (with a bytes-derived fallback when Pillow
  cannot open the file) and a binary search over ``long_side``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

log = logging.getLogger("pdf2md_agent.image_budget")

# Shared with :mod:`pdf2md_agent.token_estimator`. Imported lazily so the
# image-budget planner can be tested without the estimator pulling in
# extra dependencies.
_BYTES_PER_TOKEN: Final[float] = 3.5
"""Base64 chars per token — mirrored from ``token_estimator``; the planner
uses it for size-based estimates without depending on the estimator
public API."""


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


# --- Size estimation primitives --------------------------------------------


def _b64_chars(size_bytes: int) -> int:
    """Return the length of a base64 string encoding ``size_bytes`` raw bytes."""
    return ((size_bytes + 2) // 3) * 4


def _tokens_for_size(size_bytes: int) -> int:
    """Convert a raw byte count to its estimated base64 token cost."""
    return max(1, math.ceil(_b64_chars(size_bytes) / _BYTES_PER_TOKEN))


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


def _bytes_to_fallback_size(num_bytes: int) -> tuple[int, int]:
    """Estimate ``(long_side, long_side)`` from a JPEG byte count.

    Heuristic: a quality-85 JPEG compresses to roughly 0.25 bytes/pixel
    (≈ 4 pixels/byte), so a square image has ``long_side ≈ sqrt(bytes * 4)``
    ≈ ``2 * sqrt(bytes)``. Floored at 256 px so a 0-byte / tiny file still
    produces a sane size rather than 0, which the binary search would
    reject with ``orig_long_side <= min_long_side``.
    """
    if num_bytes <= 0:
        return 256, 256
    side = max(256, int(2 * math.sqrt(num_bytes)))
    return side, side


def _open_for_size(path: Path, *, fallback_bytes: int) -> tuple[int, int]:
    """Return ``(width, height)`` of ``path`` using Pillow.

    Falls back to a byte-derived estimate when Pillow cannot open the
    file. Without this fallback the planner would treat the corrupt
    image as 1×1, conclude it already fits the budget, and inline the
    raw oversized blob to the LLM (D6-007). ``fallback_bytes`` is the
    ``Path.stat().st_size`` cached by the caller, so the estimate
    reflects how much data the corrupt blob actually contains.
    """
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - Pillow is a hard project dep
        log.warning(
            "Pillow not importable in plan_for_image; using bytes-based fallback (%d bytes) for %s",
            fallback_bytes,
            path,
        )
        return _bytes_to_fallback_size(fallback_bytes)
    try:
        with Image.open(path) as img:
            return img.size
    except Exception as exc:
        log.warning(
            "plan_for_image: cannot open %s to read size (%s); using "
            "bytes-based fallback (%d bytes) to keep planner sizing sane",
            path,
            exc,
            fallback_bytes,
        )
        return _bytes_to_fallback_size(fallback_bytes)


# --- Public entry point -----------------------------------------------------


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
            (see :func:`pdf2md_agent.config.resolve_ctx_limit` for how
            this is resolved).
        persona_tokens: Estimated tokens for the agent persona + task
            system prompt (already pre-computed by the caller).
        fixed_text_tokens: Estimated tokens for the per-page variables:
            optional text-hint and the rendered task description scaffold.
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
    if target_long_side < 1 or min_long_side < 1:
        raise ValueError(
            f"target_long_side and min_long_side must be >= 1, got target={target_long_side}, min={min_long_side}"
        )
    if target_long_side < min_long_side:
        raise ValueError(f"target_long_side={target_long_side} < min_long_side={min_long_side}")

    limit = int(ctx_limit * safety)
    budget_for_image = max(0, limit - persona_tokens - fixed_text_tokens)
    try:
        size = image_path.stat().st_size
    except OSError:
        size = 0
    original_bytes = size
    current_tokens = _tokens_for_size(size) if size > 0 else 0

    if current_tokens <= budget_for_image:
        total = persona_tokens + fixed_text_tokens + current_tokens
        return BudgetDecision(
            total=total,
            limit=limit,
            fits=True,
            needed_long_side=target_long_side,
            reason="image fits within budget at original size",
        )

    orig_w, orig_h = _open_for_size(image_path, fallback_bytes=original_bytes)
    orig_long_side = max(orig_w, orig_h)

    if orig_long_side <= min_long_side:
        total = persona_tokens + fixed_text_tokens + current_tokens
        return BudgetDecision(
            total=total,
            limit=limit,
            fits=False,
            needed_long_side=orig_long_side,
            reason=(f"image already smaller than min_long_side={min_long_side}; fixed-text budget is too tight"),
        )

    upper = orig_long_side
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
            reason=(f"downscaled from {upper}px to {best}px to fit budget={budget_for_image} image tokens"),
        )

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


__all__ = ["BudgetDecision", "plan_for_image"]

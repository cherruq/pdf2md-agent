"""Per-page image preparation: token-budget planning, downscaling, tiling.

Each call to the vision model inlines the page as a base64 data URL. The
model's context window is finite, so for any non-trivial page the runner
needs to (a) estimate the cost of the persona + per-page prompt + image,
(b) if that exceeds the budget, downscale or split the page into tiles.

This module isolates that preparation from the rest of the pipeline:

* :func:`prepare_page_image` — the entry point the runner calls once per
  page. Returns a :class:`PreparedPage` describing which image (or tiles)
  to attach.
* :func:`_resize_page_png` — LANCZOS downscale to a JPEG copy.
* :func:`_make_tiles` — split an oversized page into two vertical tiles
  with a small overlap; used when even the smallest allowed downscale
  won't fit the budget.

The LANCZOS + JPEG re-encode here mirrors what
:func:`pdf2md_agent.crew.multimodal_patch._encode_local_image` does in
memory, so the on-disk resized cache file looks identical to what the
patch would produce inline.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from typing import TYPE_CHECKING
from pdf2md_agent.cache import CacheLayout
from pdf2md_agent.pdf_renderer import RenderedPage
from pdf2md_agent.image_budget import plan_for_image
from pdf2md_agent.token_estimator import (
    estimate_image_tokens,
    estimate_text_tokens,
)
from pdf2md_agent.config import resolve_ctx_limit
from pdf2md_agent.crew.agents import EXTRACTOR_BACKSTORY
from pdf2md_agent.crew.tasks import build_extract_description
from pdf2md_agent.tuning import IMAGE_MIN_LONG_SIDE, TOKEN_BUDGET_SAFETY_DEFAULT

if TYPE_CHECKING:
    from pdf2md_agent.config import ConversionConfig

log = logging.getLogger("pdf2md_agent.runner")


@dataclass(frozen=True, slots=True)
class PreparedPage:
    """Which image(s) to attach to the extractor for one page."""

    attach_image_path: Path
    """Path the task should reference. Equal to ``page.image_path`` for
    pages that already fit the budget at their original size."""
    is_tiled: bool
    """True if the page was split into ``tile_paths`` (extreme budget case)."""
    tile_paths: list[Path]
    """Half-overlap JPEG tiles; empty when ``is_tiled`` is False."""
    page_started: float
    """``time.monotonic()`` snapshot taken right before the LLM call —
    used by the caller to log per-page elapsed time."""


def _resize_page_png(
    src: Path, dst: Path, *, target_long_side: int, jpeg_quality: int
) -> None:
    """Render ``src`` to ``dst`` as a downscaled JPEG.

    Uses the same LANCZOS resampler as
    :func:`pdf2md_agent.crew.multimodal_patch._encode_local_image` so the
    pre-resized cache file looks identical to what the in-memory patch
    would produce inline.
    """
    from PIL import Image

    with Image.open(src) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((target_long_side, target_long_side), Image.LANCZOS)
        img.save(dst, "JPEG", quality=jpeg_quality, optimize=True)


def _resized_cache_path(layout: CacheLayout, page_number: int) -> Path:
    """Path for the downscaled JPEG copy of ``page_number``."""
    return layout.pages_dir / f"page_{page_number:04d}_resized.jpg"


def _make_tiles(
    page: RenderedPage, pages_dir: Path, *, jpeg_quality: int
) -> tuple[Path, Path]:
    """Split ``page.image_path`` into two vertically-stacked JPEG tiles.

    The tiles overlap by 10% of the page height so any text near the
    boundary still appears in one of the two halves. Returns
    ``(tile1_path, tile2_path)``; both are written under ``pages_dir``.
    Cached: if the tile files already exist on disk they are reused
    without re-cropping.
    """
    from PIL import Image

    tile1_path = pages_dir / f"page_{page.page_number:04d}_tile1.jpg"
    tile2_path = pages_dir / f"page_{page.page_number:04d}_tile2.jpg"

    if tile1_path.is_file() and tile2_path.is_file():
        return tile1_path, tile2_path

    pages_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(page.image_path) as img:
        width, height = img.size
        overlap = int(height * 0.1)
        mid = height // 2

        top_box = (0, 0, width, mid + overlap)
        bottom_box = (0, mid - overlap, width, height)

        img.crop(top_box).convert("RGB").save(
            tile1_path, "JPEG", quality=jpeg_quality
        )
        img.crop(bottom_box).convert("RGB").save(
            tile2_path, "JPEG", quality=jpeg_quality
        )

    return tile1_path, tile2_path


def prepare_page_image(
    *,
    page: RenderedPage,
    text_hint_str: str,
    config: ConversionConfig,
    idx: int,
    total: int,
) -> PreparedPage:
    """Plan and produce the image(s) the extractor should attach.

    The runner passes the exact same strings (`text_hint`, etc.)
    that will eventually build the extract task. If the total text + image
    tokens exceed `ctx_limit` * `token_budget_safety`, the image is iteratively
    downscaled (binary search) until it fits. If it cannot fit even at
    `image_min_long_side`, the layout falls back to tile splitting (!32).
    """
    layout = config.layout
    image_long_side = config.image_long_side
    image_jpeg_quality = config.image_jpeg_quality
    ctx_limit = config.ctx_limit if config.ctx_limit > 0 else resolve_ctx_limit()

    persona_tokens = estimate_text_tokens(EXTRACTOR_BACKSTORY)
    description_for_budget = build_extract_description(
        page.image_path, text_hint_str
    )
    fixed_text_tokens = estimate_text_tokens(description_for_budget)
    decision = plan_for_image(
        ctx_limit=ctx_limit,
        persona_tokens=persona_tokens,
        fixed_text_tokens=fixed_text_tokens,
        image_path=page.image_path,
        target_long_side=image_long_side,
        min_long_side=IMAGE_MIN_LONG_SIDE,
        jpeg_quality=image_jpeg_quality,
        safety=TOKEN_BUDGET_SAFETY_DEFAULT,
    )
    current_img_tokens = estimate_image_tokens(page.image_path)
    log.info(
        "  [%d/%d] page %d: tokens est. total=%d (text=%d, img=%d), "
        "target_long_side=%d, reason=%s",
        idx,
        total,
        page.page_number,
        decision.total,
        persona_tokens + fixed_text_tokens,
        current_img_tokens,
        decision.needed_long_side,
        decision.reason,
    )

    pages_dir = layout.pages_dir

    if not decision.fits:
        # Even the smallest allowed downscale won't fit — split into
        # two tiles so the model sees the page in two passes.
        log.warning(
            "  [%d/%d] page %d: Extreme downscaling needed, "
            "splitting into tiles.",
            idx, total, page.page_number,
        )
        tile1_path, tile2_path = _make_tiles(
            page, pages_dir, jpeg_quality=image_jpeg_quality,
        )
        log.info(
            "  [%d/%d] page %d: extract + format starting",
            idx, total, page.page_number,
        )
        return PreparedPage(
            attach_image_path=page.image_path,
            is_tiled=True,
            tile_paths=[tile1_path, tile2_path],
            page_started=time.monotonic(),
        )

    needs_resize = decision.needed_long_side < image_long_side
    if needs_resize:
        resized_path = _resized_cache_path(layout, page.page_number)
        if not resized_path.is_file():
            pages_dir.mkdir(parents=True, exist_ok=True)
            _resize_page_png(
                page.image_path,
                resized_path,
                target_long_side=decision.needed_long_side,
                jpeg_quality=image_jpeg_quality,
            )
        log.info(
            "  [%d/%d] page %d: extract + format starting",
            idx, total, page.page_number,
        )
        return PreparedPage(
            attach_image_path=resized_path,
            is_tiled=False,
            tile_paths=[],
            page_started=time.monotonic(),
        )

    log.info(
        "  [%d/%d] page %d: extract + format starting",
        idx, total, page.page_number,
    )
    return PreparedPage(
        attach_image_path=page.image_path,
        is_tiled=False,
        tile_paths=[],
        page_started=time.monotonic(),
    )


__all__ = [
    "PreparedPage",
    "prepare_page_image",
]
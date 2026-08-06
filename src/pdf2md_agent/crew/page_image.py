"""逐页图像准备：token 预算规划、降采样（降分辨率）、切片。

每次调用视觉模型时都会将页面作为 base64 data URL 内联。由于模型的
上下文窗口是有限的，因此对于任何非简单的页面，runner 都需要
(a) 估算人设 + 逐页提示词 + 图像的成本，
(b) 如果超出预算，则对页面进行降采样或分割成多个切片。

此模块将这些准备工作与流水线的其余部分隔离开来：

* :func:`prepare_page_image` — runner 每页调用一次的入口点。
  返回一个 :class:`PreparedPage`，描述要附加哪个图像（或切片）。
* :func:`_resize_page_png` — 使用 LANCZOS 算法降采样为 JPEG 副本。
* :func:`_make_tiles` — 将过大的页面分割成两个带少量重叠的
  垂直切片；当允许的最小降采样仍然无法满足预算时使用。

这里的 LANCZOS + JPEG 重新编码与
:func:`pdf2md_agent.crew.multimodal_patch._encode_local_image` 在内存中的操作一致，
因此磁盘上经过调整大小的缓存文件看起来与内联修补程序生成的结果完全相同。
"""

from __future__ import annotations

import logging
from PIL import Image
import time
from dataclasses import dataclass
from pathlib import Path

from typing import TYPE_CHECKING
from pdf2md_agent.cache import CacheLayout
from pdf2md_agent.pdf_renderer import RenderedPage
from pdf2md_agent.crew.types import PageRunContext, PreparedPage
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


def _resize_page_png(src: Path, dst: Path, *, target_long_side: int, jpeg_quality: int) -> None:
    """将 ``src`` 渲染到 ``dst`` 作为降采样后的 JPEG。

    使用与 :func:`pdf2md_agent.crew.multimodal_patch._encode_local_image` 相同的
    LANCZOS 重采样器，因此预先调整大小后的缓存文件看起来与内存中补丁
    内联生成的结果完全相同。
    """
    with Image.open(src) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((target_long_side, target_long_side), Image.LANCZOS)
        img.save(dst, "JPEG", quality=jpeg_quality, optimize=True)


def _resized_cache_path(layout: CacheLayout, page_number: int) -> Path:
    """``page_number`` 的降采样 JPEG 副本路径。"""
    return layout.pages_dir / f"page_{page_number:04d}_resized.jpg"


def _make_tiles(page: RenderedPage, pages_dir: Path, *, jpeg_quality: int) -> tuple[Path, Path]:
    """将 ``page.image_path`` 分割成两个垂直堆叠的 JPEG 切片。

    切片在页面高度上重叠 10%，以便边界附近的任何文本
    仍会出现在这两半中的一个里面。返回
    ``(tile1_path, tile2_path)``；两者都写入在 ``pages_dir`` 下。
    已缓存：如果切片文件已经存在于磁盘上，它们将被重用，
    而不会重新裁剪。
    """
    tile1_path = pages_dir / f"page_{page.ctx.page_number:04d}_tile1.jpg"
    tile2_path = pages_dir / f"page_{page.ctx.page_number:04d}_tile2.jpg"

    if tile1_path.is_file() and tile2_path.is_file():
        return tile1_path, tile2_path

    pages_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(page.image_path) as img:
        width, height = img.size
        overlap = int(height * 0.1)
        mid = height // 2

        top_box = (0, 0, width, mid + overlap)
        bottom_box = (0, mid - overlap, width, height)

        img.crop(top_box).convert("RGB").save(tile1_path, "JPEG", quality=jpeg_quality)
        img.crop(bottom_box).convert("RGB").save(tile2_path, "JPEG", quality=jpeg_quality)

    return tile1_path, tile2_path


def prepare_page_image(
    *,
    page: RenderedPage,
    text_hint_str: str,
    config: ConversionConfig,
) -> PreparedPage:
    """规划并生成提取器应当附加的图像。

    runner 传递最终将构建提取任务的完全相同的字符串（`text_hint` 等）。
    如果总的文本 + 图像 tokens 超过了 `ctx_limit` * `token_budget_safety`，
    图像会被迭代降采样（二分查找）直到满足要求。如果即使在 `image_min_long_side`
    下也无法容纳，则布局将回退到切片分割（!32）。
    """
    layout = config.layout
    image_long_side = config.image_long_side
    image_jpeg_quality = config.image_jpeg_quality
    ctx_limit = config.ctx_limit if config.ctx_limit > 0 else resolve_ctx_limit()

    persona_tokens = estimate_text_tokens(EXTRACTOR_BACKSTORY)
    description_for_budget = build_extract_description(page.image_path, text_hint_str)
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
        "  [%d/%d] page %d: tokens est. total=%d (text=%d, img=%d), target_long_side=%d, reason=%s",
        page.ctx.idx,
        page.ctx.total,
        page.ctx.page_number,
        decision.total,
        persona_tokens + fixed_text_tokens,
        current_img_tokens,
        decision.needed_long_side,
        decision.reason,
    )

    is_tiled = False
    tile_paths: list[Path] = []
    attach_path: Path

    if not decision.fits:
        log.warning(
            "  [%d/%d] page %d: Extreme downscaling needed, splitting into tiles.",
            page.ctx.idx,
            page.ctx.total,
            page.ctx.page_number,
        )
        tile1, tile2 = _make_tiles(
            page,
            layout.pages_dir,
            jpeg_quality=image_jpeg_quality,
        )
        is_tiled = True
        tile_paths = [tile1, tile2]
        attach_path = page.image_path
    elif decision.needed_long_side < image_long_side:
        downscaled_path = _resized_cache_path(layout, page.ctx.page_number)
        if not downscaled_path.is_file():
            layout.pages_dir.mkdir(parents=True, exist_ok=True)
            _resize_page_png(
                page.image_path,
                downscaled_path,
                target_long_side=decision.needed_long_side,
                jpeg_quality=image_jpeg_quality,
            )
        attach_path = downscaled_path
    else:
        attach_path = page.image_path

    log.info(
        "  [%d/%d] page %d: extract + format starting",
        page.ctx.idx,
        page.ctx.total,
        page.ctx.page_number,
    )
    return PreparedPage(
        page=page,
        ctx=page.ctx,
        text_hint_str=text_hint_str,
        attach_image_path=attach_path,
        is_tiled=is_tiled,
        tile_paths=tile_paths,
    )


__all__ = [
    "PreparedPage",
    "prepare_page_image",
]

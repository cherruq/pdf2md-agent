"""逐页 CrewAI 流水线编排器。"""

from __future__ import annotations

import logging
import time
import time
from dataclasses import replace
from crewai import LLM

from pdf2md_agent.cache import (
    is_page_complete,
)
from pdf2md_agent.config import (
    MODEL_NAME,
    ConversionConfig,
)
from pdf2md_agent.crew.extraction import run_extraction_loop
from pdf2md_agent.crew.multimodal_patch import patch_add_image_tool
from pdf2md_agent.crew.page_image import prepare_page_image
from pdf2md_agent.crew.types import PageResult, PageRunContext
from pdf2md_agent.pdf_renderer import RenderedPage, read_page_text

log = logging.getLogger("pdf2md_agent.runner")


def _process_single_page(
    *,
    page: RenderedPage,
    config: ConversionConfig,
    llm: LLM,
) -> tuple[PageResult, bool]:
    """通过短路或完整的提取循环处理单个 PDF 页面。"""
    ctx = replace(page.ctx, page_started=time.monotonic())
    page = replace(page, ctx=ctx)

    artifacts = config.layout.artifacts_for(ctx.page_number)

    # 短路：format.md 已缓存 → 信任。
    if not config.no_cache.format and is_page_complete(config.layout, ctx.page_number):
        cached_md = artifacts.format_markdown.read_text(encoding="utf-8").strip()
        log.info(
            "  [%d/%d] page %d: cached, skipping",
            ctx.idx,
            ctx.total,
            ctx.page_number,
        )
        return PageResult(ctx.page_number, cached_md), False

    # 完整的流水线路径。
    text_hint_str = ""
    if config.text_hint:
        text_hint_str = page.text if getattr(page, "text", "") else read_page_text(artifacts.page_text)
    prepared = prepare_page_image(
        page=page,
        text_hint_str=text_hint_str,
        config=config,
    )

    outcome = run_extraction_loop(
        prepared=prepared,
        config=config,
        llm=llm,
    )

    if outcome.fell_back:
        return PageResult(ctx.page_number, outcome.format_md), True

    artifacts.format_markdown.write_text(outcome.format_md, encoding="utf-8")

    elapsed = time.monotonic() - ctx.page_started
    log.info(
        "  [%d/%d] page %d: done in %.1fs (%s chars)",
        ctx.idx,
        ctx.total,
        ctx.page_number,
        elapsed,
        f"{len(outcome.format_md):,}",
    )
    return PageResult(ctx.page_number, outcome.format_md), False


def run_pipeline(
    *,
    pages: list[RenderedPage],
    config: ConversionConfig,
    llm: LLM,
) -> list[PageResult]:
    """在 ``pages`` 上运行逐页的 CrewAI 流水线并返回页面结果。"""
    patch_add_image_tool(
        target_long_side=config.image_long_side,
        jpeg_quality=config.image_jpeg_quality,
    )

    results: list[PageResult] = []
    pipeline_started = time.monotonic()
    total = len(pages)
    fallback_pages: list[int] = []
    log.info(
        "pipeline started: pages=%d, dpi=%d, model=%s, no_cache=%s",
        total,
        config.dpi,
        MODEL_NAME,
        config.no_cache.as_dict(),
    )

    for page in pages:
        page_res, fell_back = _process_single_page(
            page=page,
            config=config,
            llm=llm,
        )
        results.append(page_res)
        if fell_back:
            fallback_pages.append(page.ctx.page_number)

    total_elapsed = time.monotonic() - pipeline_started
    log.info(
        "pipeline complete: %d page(s) in %.1fs (%.1fs avg)",
        total,
        total_elapsed,
        total_elapsed / max(total, 1),
    )
    if fallback_pages:
        log.info(
            "run complete: %d pages, %d used fallback (text layer): %s",
            total,
            len(fallback_pages),
            fallback_pages,
        )
    return results


# 用于统一转换流水线的第 2 步入口点别名
step2_extract_pages = run_pipeline

__all__ = [
    "run_pipeline",
    "step2_extract_pages",
]

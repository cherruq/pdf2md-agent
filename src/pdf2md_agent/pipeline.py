"""pdf2md-agent 转换的流水线(Pipeline)编排与配置。"""

from __future__ import annotations

import logging
import sys
import time

from pdf2md_agent.cache import (
    atomic_write_text,
    check_meta_matches,
    read_meta,
    write_meta,
)
from pdf2md_agent.config import (
    TOKEN_BUDGET_SAFETY,
    ConversionConfig,
)
from pdf2md_agent.crew.orchestrator import run_extraction_phase
from pdf2md_agent.crew.types import RenderedPage
from pdf2md_agent.pdf_renderer import render_pages
from pdf2md_agent.post_stream import StitchMode, stitch_pages
from pdf2md_agent.vision import make_vision_llm

log = logging.getLogger("pdf2md_agent.pipeline")


def run_unified_conversion(config: ConversionConfig) -> int:
    """执行统一的 3 步 PDF 到 Markdown 转换流水线。"""
    log.info("converting %s", config.pdf)
    log.info("  output:          %s", config.output)
    log.info("  cache:           %s", config.layout.root)
    log.info("  dpi:             %d", config.dpi)
    log.info("  pages:           %s", "all" if config.resolved_pages is None else config.resolved_pages)
    log.info("  no-cache:        %s", config.no_cache.as_dict())
    log.info("  text-hint:       %s", "on" if config.text_hint else "off")
    log.info("  workers:         %d", config.max_workers)

    # --- Fail-fast 校验大模型配置 ---
    llm = make_vision_llm()

    existing_meta = read_meta(config.layout.meta_path)
    if existing_meta is not None and not config.no_cache.all():
        reasons = check_meta_matches(
            existing_meta,
            pdf=str(config.pdf.resolve()),
        )
        if reasons:
            for r in reasons:
                print(f"error: cache invalid: {r}", file=sys.stderr)
            print(
                "error: meta.json fingerprint drift detected. "
                "re-run with --no-cache-all or wipe "
                f"{config.layout.root} to rebuild the cache.",
                file=sys.stderr,
            )
            return 1
    write_meta(
        config.layout.meta_path,
        pdf=config.pdf,
    )

    # --- 步骤 1: 静态渲染与实时文本缓存同步 ---
    log.info("Step 1: rendering PDF to PNGs at %d dpi%s...", config.dpi, " (subset)" if config.resolved_pages else "")
    pages: list[RenderedPage] = render_pages(config)
    log.info("Step 1 done: rendered %d page(s) to %s", len(pages), config.render_target)

    # --- 步骤 2: 并行的逐页 AI 提取循环 ---
    log.info("Step 2: running per-page extraction and formatting pipeline")
    log.info(
        "  retry:           max_attempts=%s, initial_delay=%.1fs, fibonacci, max_delay=%.1fs, jitter=±%.0f%%",
        config.retry_config.max_attempts if config.retry_config.max_attempts is not None else "\u221e",
        config.retry_config.initial_delay,
        config.retry_config.max_delay,
        config.retry_config.jitter * 100,
    )
    log.info(
        "  budget:          ctx_limit=%d, safety=%.0f%%, image_long_side=%dpx, image_q=%d",
        config.ctx_limit,
        TOKEN_BUDGET_SAFETY * 100,
        config.image_long_side,
        config.image_jpeg_quality,
    )
    results = run_extraction_phase(
        pages=pages,
        config=config,
        llm=llm,
    )

    # --- 步骤 3: 全局后处理、清理与跨页拼接(Stitching) ---
    stitch_mode = StitchMode(config.stitch_mode)
    markdown = stitch_pages(results, llm=llm, mode=stitch_mode)
    log.info("Step 3: stitch (%s) done", stitch_mode.value)
    atomic_write_text(config.output, markdown)
    elapsed = time.monotonic() - config.started
    log.info(
        "wrote %s — %d page(s), %s chars in %.1fs",
        config.output,
        len(results),
        f"{len(markdown):,}",
        elapsed,
    )
    return 0


__all__ = [
    "atomic_write_text",
    "check_meta_matches",
    "make_vision_llm",
    "read_meta",
    "run_extraction_phase",
    "run_unified_conversion",
    "stitch_pages",
    "write_meta",
]

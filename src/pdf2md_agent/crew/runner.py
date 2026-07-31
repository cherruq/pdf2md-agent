"""Per-page CrewAI pipeline orchestrator."""

from __future__ import annotations

import logging
import time
from crewai import LLM

from pdf2md_agent.cache import (
    is_page_complete,
)
from pdf2md_agent.config import (
    MODEL_NAME,
    ConversionConfig,
)
from pdf2md_agent.crew.agents import (  # noqa: F401  re-exports kept for backward-compat with test surface
    EXTRACTOR_BACKSTORY,
    make_extractor,
)
from pdf2md_agent.crew.fallback import (  # noqa: F401  re-exports kept for backward-compat with test surface
    _record_text_layer_fallback,
    _text_layer_fallback,
)
from pdf2md_agent.crew.multimodal_patch import patch_add_image_tool
from pdf2md_agent.crew.output import _output, _strip_think  # noqa: F401  re-exports kept for backward-compat with test surface
from pdf2md_agent.crew.page_image import (  # noqa: F401  re-exports kept for backward-compat with test surface
    _resize_page_png,
    prepare_page_image,
)
from pdf2md_agent.crew.results import PageResult
from pdf2md_agent.crew.tasks import (  # noqa: F401  re-exports kept for backward-compat with test surface
    make_extract_task,
)
from crewai import Crew, Process  # noqa: F401  re-exported so tests can patch `pdf2md_agent.crew.runner.Crew` without `create=True`
from pdf2md_agent.pdf_renderer import RenderedPage, read_page_text, render_pdf  # noqa: F401  re-exported so tests can patch `pdf2md_agent.crew.runner.RenderedPage` / `.render_pdf` without `create=True`
from pdf2md_agent.vision import make_vision_llm  # noqa: F401  re-exported so tests can patch `pdf2md_agent.crew.runner.make_vision_llm` without `create=True`

log = logging.getLogger("pdf2md_agent.runner")


def _process_single_page(
    *,
    page: RenderedPage,
    config: ConversionConfig,
    llm: LLM,
    idx: int,
    total: int,
) -> tuple[PageResult, bool]:
    """Process a single PDF page through short-circuits or the full extraction loop."""
    # Lazy import to keep the runner ↔ extraction import order acyclic.
    from pdf2md_agent.crew.extraction import run_extraction_loop

    artifacts = config.layout.artifacts_for(page)

    # Short-circuit: format.md cached → trust.
    if not config.no_cache.format and is_page_complete(config.layout, page.page_number):
        cached_md = artifacts.format_markdown.read_text(encoding="utf-8").strip()
        log.info(
            "  [%d/%d] page %d: cached, skipping",
            idx,
            total,
            page.page_number,
        )
        return PageResult(page.page_number, cached_md), False

    # Full pipeline path.
    text_hint_str = ""
    if config.text_hint:
        text_hint_str = page.text if getattr(page, "text", "") else read_page_text(artifacts.page_text)
    prepared = prepare_page_image(
        page=page,
        text_hint_str=text_hint_str,
        config=config,
        idx=idx,
        total=total,
    )

    outcome = run_extraction_loop(
        page=page,
        prepared=prepared,
        config=config,
        llm=llm,
        text_hint_str=text_hint_str,
        idx=idx,
        total=total,
    )

    if outcome.fell_back:
        assert outcome.page_result is not None
        return outcome.page_result, True

    artifacts.format_markdown.write_text(outcome.format_md, encoding="utf-8")

    elapsed = time.monotonic() - prepared.page_started
    log.info(
        "  [%d/%d] page %d: done in %.1fs (%s chars)",
        idx,
        total,
        page.page_number,
        elapsed,
        f"{len(outcome.format_md):,}",
    )
    return PageResult(page.page_number, outcome.format_md), False


def run_pipeline(
    *,
    pages: list[RenderedPage],
    config: ConversionConfig,
    llm: LLM,
) -> list[PageResult]:
    """Run the per-page CrewAI pipeline across ``pages`` and return page results."""
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

    for idx, page in enumerate(pages, start=1):
        page_res, fell_back = _process_single_page(
            page=page,
            config=config,
            llm=llm,
            idx=idx,
            total=total,
        )
        results.append(page_res)
        if fell_back:
            fallback_pages.append(page.page_number)

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


# Step 2 entry point alias for the unified conversion pipeline
step2_extract_pages = run_pipeline

__all__ = [
    "Crew",  # re-exported from crewai (test patch surface)
    "RenderedPage",  # re-exported from pdf_renderer
    "PageResult",  # re-exported from pdf2md_agent.crew.results
    "_output",  # re-exported from pdf2md_agent.crew.output
    "_record_text_layer_fallback",  # re-exported from pdf2md_agent.crew.fallback
    "_resize_page_png",  # re-exported from pdf2md_agent.crew.page_image
    "_strip_think",  # re-exported from pdf2md_agent.crew.output
    "_text_layer_fallback",  # re-exported from pdf2md_agent.crew.fallback
    "make_extractor",  # re-exported from pdf2md_agent.crew.agents
    "make_extract_task",  # re-exported from pdf2md_agent.crew.tasks
    "make_vision_llm",  # re-exported from vision
    "render_pdf",  # re-exported from pdf2md_agent.pdf_renderer
    "run_pipeline",
    "step2_extract_pages",
]

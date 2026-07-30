"""Per-page CrewAI pipeline orchestrator."""
from __future__ import annotations

import logging
import time
from typing import Any

from crewai import LLM
from pydantic import ValidationError

from pdf2md_agent.cache import (
    FALLBACK_SENTINEL as _FALLBACK_SENTINEL,  # noqa: F401  re-exported under legacy underscore-prefixed name for test back-compat
    CacheLayout,
    CacheNoCacheFlags,
    has_cached_extract,
    is_page_complete,
)
from pdf2md_agent.config import (
    IMAGE_JPEG_QUALITY,
    IMAGE_LONG_SIDE,
    IMAGE_MIN_LONG_SIDE,
    MODEL_NAME,
    TOKEN_BUDGET_SAFETY,
    ConversionConfig,
    resolve_ctx_limit,
)
from pdf2md_agent.crew.agents import (  # noqa: F401  re-exports kept for backward-compat with test surface
    EXTRACTOR_BACKSTORY,
    make_extractor,
    make_formatter,
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
    make_format_task,
    make_format_task_from_extract_file,
)
from crewai import Crew, Process  # noqa: F401  re-exported so tests can patch `pdf2md_agent.crew.runner.Crew` without `create=True`
from pdf2md_agent.llm_retry import (
    RetryConfig,
    _safe_exc_summary,
    call_with_retry,
    is_transient,
)
from pdf2md_agent.pdf_renderer import PageImage, render_pdf  # noqa: F401  re-exported so tests can patch `pdf2md_agent.crew.runner.PageImage` / `.render_pdf` without `create=True`
from pdf2md_agent.vision import make_vision_llm  # noqa: F401  re-exported so tests can patch `pdf2md_agent.crew.runner.make_vision_llm` without `create=True`

log = logging.getLogger("pdf2md_agent.runner")


def _run_format_only(
    *,
    page_number: int,
    artifacts: Any,
    llm: LLM,
    config: ConversionConfig,
    fallback_to_text: bool = True,
    **_kwargs: object,
) -> tuple[str, bool]:
    """Run formatter without the extractor (used by ``--no-cache-extract`` short-circuit)."""
    formatter = make_formatter(llm)
    format_t = make_format_task_from_extract_file(formatter, artifacts.extract_text)
    tasks = [format_t]
    agents_list = [formatter]

    crew = Crew(
        agents=agents_list,
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )

    format_md: str
    did_fallback = False
    try:
        call_with_retry(
            crew.kickoff,
            config=config.retry_config,
            label=f"no-cache-extract page {page_number}",
            timeout_seconds=config.request_timeout_seconds,
        )
        format_md = _output(format_t)
    except ValidationError:
        if not fallback_to_text:
            raise
        log.warning(
            "  page %d: no-cache-extract produced malformed output; "
            "writing extract.txt as-is",
            page_number,
        )
        format_md = artifacts.extract_text.read_text(encoding="utf-8")
        did_fallback = True
    except BaseException as exc:  # noqa: BLE001 — see runner/AGENTS.md
        if not fallback_to_text or not is_transient(exc):
            raise
        log.warning(
            "  page %d: no-cache-extract failed after retries (%s); "
            "writing extract.txt as-is",
            page_number,
            _safe_exc_summary(exc),
        )
        format_md = artifacts.extract_text.read_text(encoding="utf-8")
        did_fallback = True

    artifacts.format_markdown.write_text(format_md, encoding="utf-8")
    return format_md, did_fallback


def _process_single_page(
    *,
    page: PageImage,
    idx: int,
    total: int,
    config: ConversionConfig,
    llm: LLM,
    fallback_to_text: bool = True,
    phases: str,
    extractor_persona_text: str,
) -> tuple[PageResult, bool]:
    """Process a single PDF page through short-circuits or the full extraction loop."""
    # Lazy import to keep the runner ↔ extraction import order acyclic.
    from pdf2md_agent.crew.extraction import run_extraction_loop

    artifacts = config.layout.artifacts_for(page)

    # Short-circuit 1: format.md + extract.txt both cached → trust.
    if not config.no_cache.format and is_page_complete(config.layout, page.page_number):
        cached_md = artifacts.format_markdown.read_text(encoding="utf-8").strip()
        log.info(
            "  [%d/%d] page %d: cached, skipping",
            idx, total, page.page_number,
        )
        return PageResult(page.page_number, cached_md), False

    # Short-circuit 2: --no-cache-extract with cached extract → re-format only.
    if (
        config.no_cache.extract
        and not config.no_cache.format
        and has_cached_extract(config.layout, page.page_number)
    ):
        page_started = time.monotonic()
        log.info(
            "  [%d/%d] page %d: no-cache-extract (cached extract, no image)",
            idx, total, page.page_number,
        )
        fmt_out, did_fallback = _run_format_only(
            page_number=page.page_number,
            artifacts=artifacts,
            llm=llm,
            config=config,
            fallback_to_text=fallback_to_text,
        )
        elapsed = time.monotonic() - page_started
        log.info(
            "  [%d/%d] page %d: no-cache-extract done in %.1fs%s",
            idx, total, page.page_number, elapsed,
            " (fallback)" if did_fallback else "",
        )
        return PageResult(page.page_number, fmt_out), False
    if config.no_cache.extract and not has_cached_extract(config.layout, page.page_number):
        log.warning(
            "  [%d/%d] page %d: extract.txt missing, "
            "falling back to full extract+format",
            idx, total, page.page_number,
        )

    # Full pipeline path.
    text_hint_str = (
        artifacts.page_text.read_text(encoding="utf-8") if config.text_hint else ""
    )
    ctx_limit = config.ctx_limit if config.ctx_limit > 0 else resolve_ctx_limit()
    prepared = prepare_page_image(
        page=page,
        layout=config.layout,
        text_hint_str=text_hint_str,
        ctx_limit=ctx_limit,
        image_long_side=config.image_long_side,
        image_min_long_side=IMAGE_MIN_LONG_SIDE,
        image_jpeg_quality=config.image_jpeg_quality,
        token_budget_safety=TOKEN_BUDGET_SAFETY,
        idx=idx, total=total,
        extractor_persona_text=extractor_persona_text,
        phases=phases,
    )

    outcome = run_extraction_loop(
        config=config,
        page=page,
        idx=idx,
        total=total,
        prepared=prepared,
        text_hint_str=text_hint_str,
        llm=llm,
        fallback_to_text=fallback_to_text,
    )

    if outcome.fell_back:
        assert outcome.page_result is not None
        return outcome.page_result, True

    artifacts.extract_text.write_text(outcome.extract_text, encoding="utf-8")
    artifacts.format_markdown.write_text(outcome.format_md, encoding="utf-8")

    elapsed = time.monotonic() - prepared.page_started
    log.info(
        "  [%d/%d] page %d: done in %.1fs (%s chars)",
        idx, total, page.page_number, elapsed, f"{len(outcome.format_md):,}",
    )
    return PageResult(page.page_number, outcome.format_md), False


def run_pipeline(
    *,
    pages: list[PageImage],
    config: ConversionConfig,
    llm: LLM,
    fallback_to_text: bool = True,
    **_kwargs: object,
) -> list[PageResult]:
    """Run the per-page CrewAI pipeline across ``pages`` and return page results."""
    ctx_limit = config.ctx_limit if config.ctx_limit > 0 else resolve_ctx_limit()

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
        total, config.dpi, MODEL_NAME, config.no_cache.as_dict(),
    )
    phases = "extract + format"
    extractor_persona_text = EXTRACTOR_BACKSTORY

    for idx, page in enumerate(pages, start=1):
        page_res, fell_back = _process_single_page(
            page=page,
            idx=idx,
            total=total,
            config=config,
            llm=llm,
            fallback_to_text=fallback_to_text,
            phases=phases,
            extractor_persona_text=extractor_persona_text,
        )
        results.append(page_res)
        if fell_back:
            fallback_pages.append(page.page_number)

    total_elapsed = time.monotonic() - pipeline_started
    log.info(
        "pipeline complete: %d page(s) in %.1fs (%.1fs avg)",
        total, total_elapsed, total_elapsed / max(total, 1),
    )
    if fallback_pages:
        log.info(
            "run complete: %d pages, %d used fallback (text layer): %s",
            total, len(fallback_pages), fallback_pages,
        )
    return results


# Step 2 entry point alias for the unified conversion pipeline
step2_extract_pages = run_pipeline

__all__ = [
    "Crew",  # re-exported from crewai (test patch surface)
    "PageImage",  # re-exported from pdf_renderer
    "PageResult",  # re-exported from pdf2md_agent.crew.results
    "_FALLBACK_SENTINEL",  # legacy alias: tests import this name from runner
    "_output",  # re-exported from pdf2md_agent.crew.output
    "_record_text_layer_fallback",  # re-exported from pdf2md_agent.crew.fallback
    "_resize_page_png",  # re-exported from pdf2md_agent.crew.page_image
    "_strip_think",  # re-exported from pdf2md_agent.crew.output
    "_text_layer_fallback",  # re-exported from pdf2md_agent.crew.fallback
    "make_extractor",  # re-exported from pdf2md_agent.crew.agents
    "make_extract_task",  # re-exported from pdf2md_agent.crew.tasks
    "make_format_task",  # re-exported from pdf2md_agent.crew.tasks
    "make_format_task_from_extract_file",  # re-exported from pdf2md_agent.crew.tasks
    "make_formatter",  # re-exported from pdf2md_agent.crew.agents
    "make_vision_llm",  # re-exported from vision
    "render_pdf",  # re-exported from pdf2md_agent.pdf_renderer
    "run_pipeline",
    "step2_extract_pages",
]
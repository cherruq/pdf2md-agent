"""Per-page CrewAI runner: maintains running summary state, writes cache.

Budgets every extract call against :func:`plan_for_image`: when a raw page
PNG would blow the model context window (e.g. a 184 KB 144 DPI PNG
encoding to ~71 k base64 tokens), the page is downscaled to a JPEG copy
under ``layout.pages_dir`` before the agent ever sees it.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from crewai import Crew, LLM, Process
from pydantic import ValidationError

from pdf2md_agent.cache import (
    CacheLayout,
    CacheNoCacheFlags,
    PageArtifacts,
    has_cached_extract,
    is_page_complete,
    read_summary,
    write_summary,
)
from pdf2md_agent.config import (
    IMAGE_JPEG_QUALITY,
    IMAGE_LONG_SIDE,
    IMAGE_MIN_LONG_SIDE,
    MAX_SUMMARY_CHARS,
    MODEL_NAME,
    resolve_ctx_limit,
    TOKEN_BUDGET_SAFETY,
)
from pdf2md_agent.crew.agents import (
    EXTRACTOR_BACKSTORY,
    PERSONA_VERSION,
    make_extractor,
    make_formatter,
    make_summarizer,
)
from pdf2md_agent.crew.multimodal_patch import patch_add_image_tool
from pdf2md_agent.crew.tasks import (
    _truncate_summary,
    build_extract_description,
    make_extract_task,
    make_format_task,
    make_format_task_from_extract_file,
    make_summarize_task,
)
from pdf2md_agent.llm_retry import RetryConfig, call_with_retry, is_transient, _safe_exc_summary
from pdf2md_agent.pdf_renderer import PageImage, render_pdf  # noqa: F401  re-exported so tests can patch `pdf2md_agent.crew.runner.render_pdf` without `create=True`
from pdf2md_agent.token_budget import (
    estimate_image_tokens,
    estimate_text_tokens,
    plan_for_image,
)
from pdf2md_agent.vision import make_vision_llm  # noqa: F401  re-exported so tests can patch `pdf2md_agent.crew.runner.make_vision_llm` without `create=True`

log = logging.getLogger("pdf2md_agent.runner")

_THINK_OPEN = chr(60) + "think" + chr(62)
_THINK_CLOSE = chr(60) + "/think" + chr(62)
_THINK_BLOCK_RE = re.compile(_THINK_OPEN + r".*?" + _THINK_CLOSE, re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove inline model reasoning blocks from output.

    Some models wrap their scratchpad in reasoning tags; the configured
    MiniMax-M3 endpoint sometimes leaves them in the response. Strip them
    defensively before downstream consumers see them.
    """
    return _THINK_BLOCK_RE.sub("", text).strip()


def _output(output_text: object) -> str:
    """Extract clean text from a CrewAI task's output."""
    out = getattr(output_text, "output", None)
    if out is None:
        return ""
    raw = getattr(out, "raw", None)
    text = raw if isinstance(raw, str) else str(out)
    return _strip_think(text)


def _text_layer_fallback(artifacts: PageArtifacts) -> str:
    """Build a best-effort markdown page from the PDF's native text layer.

    Used when the vision model is unreachable after all retries. The page's
    PNG is dropped from the output (we can't describe it) and the text is
    emitted verbatim in a fenced block so reviewers can spot drift.
    """
    text = artifacts.page_text.read_text(encoding="utf-8").strip()
    if not text:
        return (
            "*(vision model unavailable and PDF text layer is empty for this "
            "page — no content recovered)*"
        )
    return (
        "*(vision model unavailable — falling back to PDF text layer; "
        "tables, figures, and layout are NOT preserved)*\n\n"
        "```\n"
        f"{text}\n"
        "```\n"
    )


def _resize_page_png(src: Path, dst: Path, *, target_long_side: int, jpeg_quality: int) -> None:
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


def _record_text_layer_fallback(
    *,
    idx: int,
    total: int,
    page_number: int,
    page_started: float,
    artifacts: PageArtifacts,
    summary: str,
    completion_label: str,
) -> PageResult:
    format_md = _text_layer_fallback(artifacts)
    artifacts.extract_text.write_text(
        _FALLBACK_SENTINEL.format(page=page_number),
        encoding="utf-8",
    )
    artifacts.format_markdown.write_text(format_md, encoding="utf-8")
    elapsed = time.monotonic() - page_started
    log.info(
        "  [%d/%d] page %d: done in %.1fs (%s, %s chars)",
        idx,
        total,
        page_number,
        elapsed,
        completion_label,
        f"{len(format_md):,}",
    )
    return PageResult(page_number, format_md, summary)


_FALLBACK_SENTINEL: str = (
    "(vision model unavailable for page {page}; text-layer fallback emitted; "
    "treat as sentinel — no extractor payload available)\n"
)


@dataclass(frozen=True, slots=True)
class PageResult:
    """One page's final markdown + the running summary after this page."""

    page_number: int
    markdown: str
    summary: str


def run_pipeline(
    *,
    pages: list[PageImage],
    layout: CacheLayout,
    with_summary: bool,
    no_cache: CacheNoCacheFlags,
    text_hint: bool,
    llm: LLM,
    retry_config: RetryConfig | None = None,
    fallback_to_text: bool = True,
    ctx_limit: int = 0,
    image_long_side: int = IMAGE_LONG_SIDE,
    image_min_long_side: int = IMAGE_MIN_LONG_SIDE,
    image_jpeg_quality: int = IMAGE_JPEG_QUALITY,
    max_summary_chars: int = MAX_SUMMARY_CHARS,
    token_budget_safety: float = TOKEN_BUDGET_SAFETY,
    dpi: int = 144,
    request_timeout_seconds: float | None = None,
    assets_dir: Path | None = None,
) -> list[PageResult]:
    """Run the per-page CrewAI pipeline across ``pages`` and return page results.

    ``text_hint`` controls whether the native PDF text layer is fed to the
    extractor agent as a per-page hint. Disabled → pass empty string.

    ``retry_config`` controls transient-error retry around each page's
    ``crew.kickoff()`` call. On exhaustion, if ``fallback_to_text`` is True,
    the page is rendered as a fenced text-layer markdown stub so the rest of
    the pipeline keeps moving; otherwise the exception propagates.

    ``no_cache`` is the per-resource opt-out switch (see
    :class:`pdf2md_agent.cache.CacheNoCacheFlags`). Every flag defaults to
    ``False`` (trust cache). Setting ``no_cache.format`` short-circuits the
    entire per-page pipeline when ``format.md`` exists. Setting
    ``no_cache.extract`` (but not ``no_cache.format``) re-runs only the
    formatter (+ summarizer) using the cached ``extract.txt``; pages whose
    extract is missing fall through to the full pipeline.

    The remaining keyword arguments are the token-budget knobs (see
    :mod:`pdf2md_agent.config`). All have sensible defaults.

    ``ctx_limit`` of 0 means "resolve at runtime" — the actual value comes
    from :func:`pdf2md_agent.config.resolve_ctx_limit`, which consults the
    env var, the ``/v1/models`` probe, and finally a hardcoded default.
    """
    if ctx_limit <= 0:
        ctx_limit = resolve_ctx_limit()

    patch_add_image_tool(
        target_long_side=image_long_side,
        jpeg_quality=image_jpeg_quality,
    )

    summary = ""
    if not no_cache.summary:
        summary = read_summary(layout.summary_path)
    results: list[PageResult] = []
    pipeline_started = time.monotonic()
    total = len(pages)
    fallback_pages: list[int] = []
    log.info(
        "pipeline started: pages=%d, dpi=%d, model=%s, persona=%s, "
        "with_summary=%s, no_cache=%s",
        total,
        dpi,
        MODEL_NAME,
        PERSONA_VERSION,
        with_summary,
        no_cache.as_dict(),
    )
    phases = "extract + format + summarize" if with_summary else "extract + format"
    pages_dir = layout.pages_dir
    summary_path = layout.summary_path

    extractor_persona_text = EXTRACTOR_BACKSTORY

    for idx, page in enumerate(pages, start=1):
        artifacts = layout.artifacts_for(page)

        if not no_cache.format and is_page_complete(layout, page.page_number):
            cached_md = artifacts.format_markdown.read_text(encoding="utf-8").strip()
            log.info("  [%d/%d] page %d: cached, skipping", idx, total, page.page_number)
            results.append(PageResult(page.page_number, cached_md, summary))
            continue

        if (
            no_cache.extract
            and not no_cache.format
            and has_cached_extract(layout, page.page_number)
        ):
            page_started = time.monotonic()
            log.info(
                "  [%d/%d] page %d: no-cache-extract (cached extract, no image)",
                idx,
                total,
                page.page_number,
            )
            fmt_out, summary, did_fallback = _run_format_summarize_only(
                page_number=page.page_number,
                artifacts=artifacts,
                summary_in=summary,
                summary_path=summary_path,
                with_summary=with_summary,
                llm=llm,
                retry_config=retry_config,
                fallback_to_text=fallback_to_text,
                max_summary_chars=max_summary_chars,
            )
            elapsed = time.monotonic() - page_started
            log.info(
                "  [%d/%d] page %d: no-cache-extract done in %.1fs%s",
                idx,
                total,
                page.page_number,
                elapsed,
                " (fallback)" if did_fallback else "",
            )
            artifacts.format_markdown.write_text(fmt_out, encoding="utf-8")
            results.append(PageResult(page.page_number, fmt_out, summary))
            continue
        if no_cache.extract and not has_cached_extract(layout, page.page_number):
            log.warning(
                "  [%d/%d] page %d: extract.txt missing, "
                "falling back to full extract+format",
                idx,
                total,
                page.page_number,
            )

        extractor = make_extractor(llm)
        formatter = make_formatter(llm)
        summarizer = None
        if with_summary:
            summarizer = make_summarizer(llm)

        text_hint_str = (
            artifacts.page_text.read_text(encoding="utf-8") if text_hint else ""
        )

        persona_tokens = estimate_text_tokens(extractor_persona_text)
        description_for_budget = build_extract_description(
            page.image_path, text_hint_str, summary,
            max_summary_chars=max_summary_chars,
        )
        fixed_text_tokens = estimate_text_tokens(description_for_budget)
        decision = plan_for_image(
            ctx_limit=ctx_limit,
            persona_tokens=persona_tokens,
            fixed_text_tokens=fixed_text_tokens,
            image_path=page.image_path,
            target_long_side=image_long_side,
            min_long_side=image_min_long_side,
            jpeg_quality=image_jpeg_quality,
            safety=token_budget_safety,
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

        attach_image_path: Path = page.image_path
        resized_path = _resized_cache_path(layout, page.page_number)
        needs_resize = (
            not decision.fits
            or decision.needed_long_side < image_long_side
        )
        is_tiled = False
        tile_paths: list[Path] = []

        if not decision.fits:
            # Extreme downscaling detected, trigger tiling fallback
            is_tiled = True
            log.warning("  [%d/%d] page %d: Extreme downscaling needed, splitting into tiles.", idx, total, page.page_number)
            pages_dir.mkdir(parents=True, exist_ok=True)
            tile1_path = pages_dir / f"page_{page.page_number:04d}_tile1.jpg"
            tile2_path = pages_dir / f"page_{page.page_number:04d}_tile2.jpg"

            if not tile1_path.is_file() or not tile2_path.is_file():
                from PIL import Image
                with Image.open(page.image_path) as img:
                    width, height = img.size
                    overlap = int(height * 0.1) # 10% overlap
                    mid = height // 2

                    top_box = (0, 0, width, mid + overlap)
                    bottom_box = (0, mid - overlap, width, height)

                    img.crop(top_box).convert("RGB").save(tile1_path, "JPEG", quality=image_jpeg_quality)
                    img.crop(bottom_box).convert("RGB").save(tile2_path, "JPEG", quality=image_jpeg_quality)

            tile_paths = [tile1_path, tile2_path]
            attach_image_path = page.image_path # Still attached for metadata, but tasks ignore it

        elif needs_resize:
            if not resized_path.is_file():
                pages_dir.mkdir(parents=True, exist_ok=True)
                _resize_page_png(
                    page.image_path,
                    resized_path,
                    target_long_side=decision.needed_long_side,
                    jpeg_quality=image_jpeg_quality,
                )
            attach_image_path = resized_path

        log.info("  [%d/%d] page %d: %s starting", idx, total, page.page_number, phases)
        page_started = time.monotonic()

        # Deterministic coverage reflection loop
        reflection_attempts = 0
        max_reflections = 2
        coverage_threshold = 0.85

        # We only consider non-whitespace/non-punctuation characters for coverage
        import re
        import difflib
        def _clean_for_coverage(text: str) -> str:
            return re.sub(r'\s+', '', text)

        native_clean = _clean_for_coverage(text_hint_str)
        needs_coverage_check = len(native_clean) > 20

        penalty_prompt = ""

        # Get list of native images available for this page
        available_images = []
        if assets_dir:
            for img_file in assets_dir.glob(f"page_{page.page_number:04d}_img_*"):
                if img_file.is_file():
                    available_images.append(img_file.name)

        while True:
            # Recreate tasks per iteration to avoid CrewAI state leaking
            extract_t = make_extract_task(
                extractor,
                attach_image_path,
                text_hint=text_hint_str + penalty_prompt,
                previous_summary=summary,
                max_summary_chars=max_summary_chars,
                available_images=available_images,
                is_tiled=is_tiled,
                tile_paths=tile_paths,
            )
            format_t = make_format_task(formatter, extract_t)
            tasks = [extract_t, format_t]
            agents_list = [extractor, formatter]
            if summarizer is not None:
                summarize_t = make_summarize_task(
                    summarizer, format_t, summary, max_chars=max_summary_chars
                )
                tasks.append(summarize_t)
                agents_list.append(summarizer)
            else:
                summarize_t = None

            crew = Crew(
                agents=agents_list,
                tasks=tasks,
                process=Process.sequential,
                verbose=False,
            )

            try:
                call_with_retry(
                    crew.kickoff,
                    config=retry_config or RetryConfig(),
                    label=f"page {page.page_number}" + (f" (reflection {reflection_attempts})" if reflection_attempts > 0 else ""),
                    timeout_seconds=request_timeout_seconds,
                )
            except ValidationError as exc:
                if not fallback_to_text:
                    raise
                log.warning(
                    "  [%d/%d] page %d: model returned malformed response "
                    "(%s, %d validation error(s)); falling back to text layer",
                    idx,
                    total,
                    page.page_number,
                    type(exc).__name__,
                    len(exc.errors()),
                )
                results.append(_record_text_layer_fallback(
                    idx=idx,
                    total=total,
                    page_number=page.page_number,
                    page_started=page_started,
                    artifacts=artifacts,
                    summary=summary,
                    completion_label="validation-fallback",
                ))
                fallback_pages.append(page.page_number)
                break
            except BaseException as exc:
                if not fallback_to_text or not is_transient(exc):
                    raise
                log.warning(
                    "  [%d/%d] page %d: vision pipeline failed after retries (%s); "
                    "falling back to text layer",
                    idx,
                    total,
                    page.page_number,
                    _safe_exc_summary(exc),
                )
                results.append(_record_text_layer_fallback(
                    idx=idx,
                    total=total,
                    page_number=page.page_number,
                    page_started=page_started,
                    artifacts=artifacts,
                    summary=summary,
                    completion_label="fallback",
                ))
                fallback_pages.append(page.page_number)
                break

            extract_text = _output(extract_t)
            format_md = _output(format_t)

            if needs_coverage_check and reflection_attempts < max_reflections:
                md_clean = _clean_for_coverage(format_md)
                # Use SequenceMatcher to find the longest contiguous matching subsequences
                sm = difflib.SequenceMatcher(None, native_clean, md_clean)
                # Calculate coverage as the ratio of matching characters relative to the native text length
                # Calculate matches
                match_blocks = sm.get_matching_blocks()
                hit_count = sum(block.size for block in match_blocks)
                coverage = hit_count / len(native_clean) if len(native_clean) > 0 else 1.0

                if coverage < coverage_threshold:
                    reflection_attempts += 1
                    log.warning(
                        "  [%d/%d] page %d: poor text coverage (%.2f < %.2f); triggering reflection %d",
                        idx, total, page.page_number, coverage, coverage_threshold, reflection_attempts
                    )
                    penalty_prompt = (
                        "\n\nCRITICAL WARNING: Your previous output missed significant portions of the native text. "
                        "You MUST preserve ALL text. Please re-read the page carefully and transcribe completely."
                    )
                    continue

            # Passed coverage or out of reflections
            break

        if page.page_number in fallback_pages:
            continue
        artifacts.extract_text.write_text(extract_text, encoding="utf-8")
        artifacts.format_markdown.write_text(format_md, encoding="utf-8")

        if summarize_t is not None and not no_cache.summary and with_summary:
            summary = _output(summarize_t)
            if len(summary) > max_summary_chars:
                summary = _truncate_summary(summary, max_summary_chars)
            write_summary(summary_path, summary)

        elapsed = time.monotonic() - page_started
        log.info(
            "  [%d/%d] page %d: done in %.1fs (%s chars)",
            idx,
            total,
            page.page_number,
            elapsed,
            f"{len(format_md):,}",
        )
        results.append(PageResult(page.page_number, format_md, summary))

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


def _run_format_summarize_only(
    *,
    page_number: int,
    artifacts: PageArtifacts,
    summary_in: str,
    summary_path: Path,
    with_summary: bool,
    llm: LLM,
    retry_config: RetryConfig,
    fallback_to_text: bool,
    max_summary_chars: int,
    request_timeout_seconds: float | None = None,
) -> tuple[str, str, bool]:
    """Run formatter + (optional) summarizer without the extractor.

    Used by the ``--no-cache-extract`` short-circuit: when the runner
    trusts the cached ``extract.txt`` but needs a fresh formatter pass
    (e.g. a resume-after-failure retry). The format task's description
    inlines the on-disk extract.txt content as a fenced block, matching
    the text-hint seam.

    On retry exhaustion with ``fallback_to_text=True``, the cached
    ``extract.txt`` is written through unchanged as the new ``format.md``
    (the natural analogue of "fallback to text layer" for a page that
    never ran the extractor). With ``fallback_to_text=False`` the exception
    propagates.

    Returns ``(format_md, summary_out, did_fallback)``.
    """
    formatter = make_formatter(llm)
    format_t = make_format_task_from_extract_file(formatter, artifacts.extract_text)

    if not with_summary:
        tasks = [format_t]
        agents_list = [formatter]
        summarize_t = None
    else:
        summarizer = make_summarizer(llm)
        summarize_t = make_summarize_task(
            summarizer, format_t, summary_in, max_chars=max_summary_chars
        )
        tasks = [format_t, summarize_t]
        agents_list = [formatter, summarizer]

    crew = Crew(
        agents=agents_list,
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )

    format_md: str
    summary_out: str
    did_fallback = False
    try:
        call_with_retry(
            crew.kickoff,
            config=retry_config,
            label=f"no-cache-extract page {page_number}",
            timeout_seconds=request_timeout_seconds,
        )
        format_md = _output(format_t)
        summary_out = _output(summarize_t) if summarize_t is not None else summary_in
    except ValidationError:
        if not fallback_to_text:
            raise
        log.warning(
            "  page %d: no-cache-extract produced malformed output; writing extract.txt as-is",
            page_number,
        )
        format_md = artifacts.extract_text.read_text(encoding="utf-8")
        summary_out = summary_in
        did_fallback = True
    except BaseException as exc:
        if not fallback_to_text or not is_transient(exc):
            raise
        log.warning(
            "  page %d: no-cache-extract failed after retries (%s); writing extract.txt as-is",
            page_number,
            _safe_exc_summary(exc),
        )
        format_md = artifacts.extract_text.read_text(encoding="utf-8")
        summary_out = summary_in
        did_fallback = True

    artifacts.format_markdown.write_text(format_md, encoding="utf-8")

    if summarize_t is not None and not did_fallback:
        if len(summary_out) > max_summary_chars:
            summary_out = _truncate_summary(summary_out, max_summary_chars)
        write_summary(summary_path, summary_out)

    return format_md, summary_out, did_fallback


__all__ = [
    "PageImage",  # re-exported from pdf2md_agent.pdf_renderer
    "PageResult",
    "make_vision_llm",  # re-exported from pdf2md_agent.vision
    "render_pdf",  # re-exported from pdf2md_agent.pdf_renderer
    "run_pipeline",
]

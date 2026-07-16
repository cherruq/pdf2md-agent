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

from convertpdf.cache import (
    CacheLayout,
    is_page_complete,
    read_summary,
    write_summary,
)
from convertpdf.config import (
    CTX_LIMIT,
    IMAGE_JPEG_QUALITY,
    IMAGE_LONG_SIDE,
    IMAGE_MIN_LONG_SIDE,
    MAX_SUMMARY_CHARS,
    TOKEN_BUDGET_SAFETY,
)
from convertpdf.crew.agents import EXTRACTOR_PERSONA, make_extractor, make_formatter, make_summarizer
from convertpdf.crew.multimodal_patch import patch_add_image_tool
from convertpdf.crew.tasks import (
    _truncate_summary,
    build_extract_description,
    make_extract_task,
    make_format_task,
    make_summarize_task,
)
from convertpdf.llm_retry import RetryConfig, call_with_retry, is_transient
from convertpdf.pdf_renderer import PageImage
from convertpdf.token_budget import (
    estimate_image_tokens,
    estimate_text_tokens,
    plan_for_image,
)

log = logging.getLogger("convertpdf.runner")

_THINK_OPEN = chr(60) + "think" + chr(62)
_THINK_CLOSE = chr(60) + "/think" + chr(62)
_THINK_RE = re.compile(_THINK_OPEN + r".*?" + _THINK_CLOSE, re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove inline model reasoning blocks from output.

    Some models wrap their scratchpad in reasoning tags; the configured
    MiniMax-M3 endpoint sometimes leaves them in the response. Strip them
    defensively before downstream consumers see them.
    """
    return _THINK_RE.sub("", text).strip()


def _output(task) -> str:
    """Extract clean text from a CrewAI task's output."""
    out = getattr(task, "output", None)
    if out is None:
        return ""
    raw = getattr(out, "raw", None)
    text = raw if isinstance(raw, str) else str(out)
    return _strip_think(text)


def _text_layer_fallback(artifacts) -> str:
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
    :func:`convertpdf.crew.multimodal_patch._encode_local_image` so the
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
    resume: bool,
    text_hint: bool,
    llm: LLM,
    retry_config: RetryConfig | None = None,
    fallback_to_text: bool = True,
    ctx_limit: int = CTX_LIMIT,
    image_long_side: int = IMAGE_LONG_SIDE,
    image_min_long_side: int = IMAGE_MIN_LONG_SIDE,
    image_jpeg_quality: int = IMAGE_JPEG_QUALITY,
    max_summary_chars: int = MAX_SUMMARY_CHARS,
    token_budget_safety: float = TOKEN_BUDGET_SAFETY,
) -> list[PageResult]:
    """Run the per-page CrewAI pipeline across ``pages`` and return page results.

    ``text_hint`` controls whether the native PDF text layer is fed to the
    extractor agent as a per-page hint. Disabled → pass empty string.

    ``retry_config`` controls transient-error retry around each page's
    ``crew.kickoff()`` call. On exhaustion, if ``fallback_to_text`` is True,
    the page is rendered as a fenced text-layer markdown stub so the rest of
    the pipeline keeps moving; otherwise the exception propagates.

    The remaining keyword arguments are the token-budget knobs (see
    :mod:`convertpdf.config`). All have sensible defaults.
    """
    extractor = make_extractor(llm)
    formatter = make_formatter(llm)
    summarizer = make_summarizer(llm) if with_summary else None

    patch_add_image_tool(
        target_long_side=image_long_side,
        jpeg_quality=image_jpeg_quality,
    )

    summary = read_summary(layout.summary_path)
    results: list[PageResult] = []
    pipeline_started = time.monotonic()
    total = len(pages)
    phases = "extract + format + summarize" if with_summary else "extract + format"

    extractor_persona_text = EXTRACTOR_PERSONA

    for idx, page in enumerate(pages, start=1):
        artifacts = layout.artifacts_for(page)

        if resume and is_page_complete(layout, page.page_number):
            cached_md = artifacts.format_markdown.read_text(encoding="utf-8").strip()
            if with_summary:
                summary = read_summary(layout.summary_path)
            log.info("  [%d/%d] page %d: cached, skipping", idx, total, page.page_number)
            results.append(PageResult(page.page_number, cached_md, summary))
            continue

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
        if needs_resize:
            if not resized_path.is_file():
                layout.pages_dir.mkdir(parents=True, exist_ok=True)
                _resize_page_png(
                    page.image_path,
                    resized_path,
                    target_long_side=decision.needed_long_side,
                    jpeg_quality=image_jpeg_quality,
                )
            attach_image_path = resized_path

        log.info("  [%d/%d] page %d: %s starting", idx, total, page.page_number, phases)
        page_started = time.monotonic()

        extract_t = make_extract_task(
            extractor,
            attach_image_path,
            text_hint=text_hint_str,
            previous_summary=summary,
            max_summary_chars=max_summary_chars,
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
                label=f"page {page.page_number}",
            )
        except ValidationError as exc:
            if not fallback_to_text:
                raise
            log.error(
                "  [%d/%d] page %d: model returned malformed response (%s); "
                "falling back to text layer",
                idx,
                total,
                page.page_number,
                type(exc).__name__,
            )
            extract_text = ""
            format_md = _text_layer_fallback(artifacts)
            artifacts.extract_text.write_text(extract_text, encoding="utf-8")
            artifacts.format_markdown.write_text(format_md, encoding="utf-8")
            elapsed = time.monotonic() - page_started
            log.info(
                "  [%d/%d] page %d: done in %.1fs (validation-fallback, %s chars)",
                idx,
                total,
                page.page_number,
                elapsed,
                f"{len(format_md):,}",
            )
            results.append(PageResult(page.page_number, format_md, summary))
            continue
        except BaseException as exc:
            if not fallback_to_text or not is_transient(exc):
                raise
            log.error(
                "  [%d/%d] page %d: vision pipeline failed after retries; "
                "falling back to text layer",
                idx,
                total,
                page.page_number,
            )
            extract_text = ""
            format_md = _text_layer_fallback(artifacts)
            artifacts.extract_text.write_text(extract_text, encoding="utf-8")
            artifacts.format_markdown.write_text(format_md, encoding="utf-8")
            elapsed = time.monotonic() - page_started
            log.info(
                "  [%d/%d] page %d: done in %.1fs (fallback, %s chars)",
                idx,
                total,
                page.page_number,
                elapsed,
                f"{len(format_md):,}",
            )
            results.append(PageResult(page.page_number, format_md, summary))
            continue

        extract_text = _output(extract_t)
        format_md = _output(format_t)
        artifacts.extract_text.write_text(extract_text, encoding="utf-8")
        artifacts.format_markdown.write_text(format_md, encoding="utf-8")

        if summarize_t is not None:
            summary = _output(summarize_t)
            if len(summary) > max_summary_chars:
                summary = _truncate_summary(summary, max_summary_chars)
            write_summary(layout.summary_path, summary)

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
    return results

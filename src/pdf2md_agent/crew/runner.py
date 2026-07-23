"""Per-page pipeline runner: explicit data flow, no hidden state.

Each page goes through three independent LLM calls — extractor, formatter,
summarizer — each wrapped in its own ``call_with_retry`` so transient
failures on one call do not poison the next. Call arguments are the entire
visible state: what the planner budgets is exactly what goes over the wire.

Why explicit calls instead of a CrewAI ``Crew.kickoff``? CrewAI's
``_invoke_loop_native_tools`` is a think→action→observe→answer loop; the
``AddImageTool`` result (base64 JPEG) is re-inlined into every subsequent
turn, which made the planner's token estimate diverge from the actual
sent payload. Going direct via :mod:`pdf2md_agent.raw_pipeline` keeps each
call to exactly one HTTP request with exactly one image inline (or zero,
for text-only calls).

Budgets every extract call against :func:`plan_for_image`: when a raw page
PNG would blow the model context window (e.g. a 184 KB 144 DPI PNG encoding
to ~71 k base64 tokens), the page is downscaled to a JPEG copy under
``layout.pages_dir`` before the extractor sees it.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from pdf2md_agent import raw_pipeline
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
from pdf2md_agent.llm_retry import (
    RetryConfig,
    _safe_exc_summary,
    call_with_retry,
    is_transient,
)
from pdf2md_agent.pdf_renderer import (  # noqa: F401  re-exported so tests can patch `pdf2md_agent.crew.runner.render_pdf` without `create=True`
    PageImage,
    render_pdf,
)
from pdf2md_agent.raw_pipeline import (
    PERSONA_VERSION,
    call_extractor,
    call_formatter,
    call_summarizer,
)
from pdf2md_agent.token_budget import (
    estimate_image_tokens,
    estimate_text_tokens,
    plan_for_image,
)

log = logging.getLogger("pdf2md_agent.runner")


# Re-export the client factory so tests can patch
# ``pdf2md_agent.crew.runner._make_client`` without ``create=True``.
_make_client = raw_pipeline._make_client  # noqa: F401


_FALLBACK_SENTINEL: str = (
    "(vision model unavailable for page {page}; text-layer fallback emitted; "
    "treat as sentinel — no extractor payload available)\n"
)


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
    :func:`pdf2md_agent.raw_pipeline._encode_local_image` so the
    pre-resized cache file is byte-identical to what the inline call would
    produce at extract time.
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
) -> list[PageResult]:
    """Run the per-page pipeline across ``pages`` and return page results.

    ``text_hint`` controls whether the native PDF text layer is fed to the
    extractor call as a per-page hint. Disabled → pass empty string.

    ``retry_config`` controls transient-error retry around each individual
    LLM call (extract / format / summarize). On exhaustion, if
    ``fallback_to_text`` is True, the page is rendered as a fenced text-layer
    markdown stub so the rest of the pipeline keeps moving; otherwise the
    exception propagates.

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
    retry_config = retry_config or RetryConfig()
    client = _make_client()

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
                client=client,
                page_number=page.page_number,
                artifacts=artifacts,
                summary_in=summary,
                summary_path=summary_path,
                with_summary=with_summary,
                retry_config=retry_config,
                fallback_to_text=fallback_to_text,
                max_summary_chars=max_summary_chars,
                request_timeout_seconds=request_timeout_seconds,
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

        text_hint_str = (
            artifacts.page_text.read_text(encoding="utf-8") if text_hint else ""
        )

        persona_tokens = estimate_text_tokens(raw_pipeline.EXTRACTOR_PERSONA)
        user_text_for_budget = raw_pipeline.build_extractor_user_text(
            text_hint=text_hint_str,
            previous_summary=summary,
            max_summary_chars=max_summary_chars,
        )
        fixed_text_tokens = estimate_text_tokens(user_text_for_budget)
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

        try:
            extract_text = call_with_retry(
                lambda: _do_extract(
                    client=client,
                    image_path=attach_image_path,
                    text_hint=text_hint_str,
                    previous_summary=summary,
                    max_summary_chars=max_summary_chars,
                    target_long_side=decision.needed_long_side,
                    jpeg_quality=image_jpeg_quality,
                    timeout=request_timeout_seconds,
                ),
                config=retry_config,
                label=f"page {page.page_number} extract",
                timeout_seconds=request_timeout_seconds,
            )
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
            continue

        try:
            format_md = call_with_retry(
                lambda: call_formatter(
                    client,
                    extract_text=extract_text,
                    timeout=request_timeout_seconds,
                ),
                config=retry_config,
                label=f"page {page.page_number} format",
                timeout_seconds=request_timeout_seconds,
            )
        except BaseException as exc:
            if not fallback_to_text or not is_transient(exc):
                raise
            log.warning(
                "  [%d/%d] page %d: formatter failed after retries (%s); "
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
            continue

        artifacts.extract_text.write_text(extract_text, encoding="utf-8")
        artifacts.format_markdown.write_text(format_md, encoding="utf-8")

        if with_summary and not no_cache.summary:
            try:
                summary = call_with_retry(
                    lambda: call_summarizer(
                        client,
                        format_text=format_md,
                        previous_summary=summary,
                        max_chars=max_summary_chars,
                        timeout=request_timeout_seconds,
                    ),
                    config=retry_config,
                    label=f"page {page.page_number} summarize",
                    timeout_seconds=request_timeout_seconds,
                )
            except BaseException as exc:
                if not fallback_to_text or not is_transient(exc):
                    raise
                log.warning(
                    "  [%d/%d] page %d: summarizer failed after retries (%s); "
                    "keeping previous summary",
                    idx,
                    total,
                    page.page_number,
                    _safe_exc_summary(exc),
                )
                # Fallback for summarizer: keep the prior summary. Format
                # + extract are already on disk; only the running summary
                # is left untouched.
                fallback_pages.append(page.page_number)
            else:
                if len(summary) > max_summary_chars:
                    summary = raw_pipeline._truncate_summary(summary, max_summary_chars)
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


def _do_extract(
    *,
    client: object,
    image_path: Path,
    text_hint: str,
    previous_summary: str,
    max_summary_chars: int,
    target_long_side: int,
    jpeg_quality: int,
    timeout: float | None,
) -> str:
    """Adapter so :func:`call_extractor`'s positional ``client`` arg works
    with :func:`call_with_retry`'s kwargs-based invocation."""
    return call_extractor(
        client,  # type: ignore[arg-type]
        image_path=image_path,
        text_hint=text_hint,
        previous_summary=previous_summary,
        max_summary_chars=max_summary_chars,
        target_long_side=target_long_side,
        jpeg_quality=jpeg_quality,
        timeout=timeout,
    )


def _run_format_summarize_only(
    *,
    client: object,
    page_number: int,
    artifacts: PageArtifacts,
    summary_in: str,
    summary_path: Path,
    with_summary: bool,
    retry_config: RetryConfig,
    fallback_to_text: bool,
    max_summary_chars: int,
    request_timeout_seconds: float | None = None,
) -> tuple[str, str, bool]:
    """Run formatter + (optional) summarizer without the extractor.

    Used by the ``--no-cache-extract`` short-circuit: when the runner
    trusts the cached ``extract.txt`` but needs a fresh formatter pass
    (e.g. a resume-after-failure retry). The formatter's user message
    inlines the on-disk ``extract.txt`` content as a fenced block,
    matching the text-hint seam.

    On retry exhaustion with ``fallback_to_text=True``, the cached
    ``extract.txt`` is written through unchanged as the new ``format.md``
    (the natural analogue of "fallback to text layer" for a page that
    never ran the extractor). With ``fallback_to_text=False`` the exception
    propagates.

    Returns ``(format_md, summary_out, did_fallback)``.
    """
    extract_text = artifacts.extract_text.read_text(encoding="utf-8")
    did_fallback = False
    format_md: str
    summary_out: str
    try:
        format_md = call_with_retry(
            lambda: call_formatter(
                client,
                extract_text=extract_text,
                timeout=request_timeout_seconds,
            ),
            config=retry_config,
            label=f"no-cache-extract page {page_number} format",
            timeout_seconds=request_timeout_seconds,
        )
    except BaseException as exc:
        if not fallback_to_text or not is_transient(exc):
            raise
        log.warning(
            "  page %d: no-cache-extract format failed after retries (%s); "
            "writing extract.txt as-is",
            page_number,
            _safe_exc_summary(exc),
        )
        format_md = extract_text
        summary_out = summary_in
        did_fallback = True
    else:
        if with_summary:
            try:
                summary_out = call_with_retry(
                    lambda: call_summarizer(
                        client,
                        format_text=format_md,
                        previous_summary=summary_in,
                        max_chars=max_summary_chars,
                        timeout=request_timeout_seconds,
                    ),
                    config=retry_config,
                    label=f"no-cache-extract page {page_number} summarize",
                    timeout_seconds=request_timeout_seconds,
                )
            except BaseException as exc:
                if not fallback_to_text or not is_transient(exc):
                    raise
                log.warning(
                    "  page %d: no-cache-extract summarize failed after retries (%s); "
                    "keeping prior summary",
                    page_number,
                    _safe_exc_summary(exc),
                )
                summary_out = summary_in
                did_fallback = True
        else:
            summary_out = summary_in

    artifacts.format_markdown.write_text(format_md, encoding="utf-8")

    if with_summary and not did_fallback:
        if len(summary_out) > max_summary_chars:
            summary_out = raw_pipeline._truncate_summary(summary_out, max_summary_chars)
        write_summary(summary_path, summary_out)

    return format_md, summary_out, did_fallback


__all__ = [
    "PageImage",  # re-exported from pdf_renderer
    "PageResult",
    "_make_client",  # re-exported from raw_pipeline (tests patch here)
    "render_pdf",  # re-exported from pdf_renderer
    "run_pipeline",
]

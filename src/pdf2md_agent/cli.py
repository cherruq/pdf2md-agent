"""CLI entry point for pdf2md-agent."""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time

import pymupdf
from pathlib import Path

from pdf2md_agent import __about__
from pdf2md_agent.cache import (
    CacheLayout,
    CacheNoCacheFlags,
    atomic_write_text,
    check_meta_matches,
    read_meta,
    write_meta,
)
from pdf2md_agent.config import (
    FALLBACK_TO_TEXT,
    IMAGE_JPEG_QUALITY,
    IMAGE_LONG_SIDE,
    IMAGE_MIN_LONG_SIDE,
    MAX_SUMMARY_CHARS,
    MODEL_NAME,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_INITIAL_DELAY,
    RETRY_JITTER,
    RETRY_MAX_ATTEMPTS,
    RETRY_MAX_DELAY,
    TOKEN_BUDGET_SAFETY,
    resolve_ctx_limit,
)
from pdf2md_agent.crew.agents import PERSONA_VERSION
from pdf2md_agent.crew.runner import run_pipeline
from pdf2md_agent.llm_retry import RetryConfig
from pdf2md_agent.pages import parse_page_spec, resolve_pages
from PIL import Image

from pdf2md_agent.pdf_renderer import PageImage, render_pdf
from pdf2md_agent.post_stream import StitchMode, stitch_pages
from pdf2md_agent.render_skip import (
    maybe_skip_render as _maybe_skip_render,
)
from pdf2md_agent.vision import make_vision_llm

log = logging.getLogger("pdf2md-agent")

from pdf2md_agent.cli_parser import _cache_key_for_pdf, build_parser, _resolve_no_cache_flags, _NO_CACHE_FLAG_NAMES, _resolve_layout, _safe_intermediates_dir, _safe_cache_stem

def _build_retry_config(args: argparse.Namespace) -> RetryConfig | None:
    """Build a RetryConfig from CLI args (override) + env (fallback). Returns None on invalid input."""
    # ``--max-retries 0`` (or env PDF2MD_AGENT_MAX_RETRIES=0) → unlimited.
    cli_max_attempts = args.max_retries
    if cli_max_attempts == 0:
        cli_max_attempts = None
    try:
        return RetryConfig(
            max_attempts=(
                cli_max_attempts
                if cli_max_attempts is not None
                else RETRY_MAX_ATTEMPTS
            ),
            initial_delay=(
                args.retry_initial_delay
                if args.retry_initial_delay is not None
                else RETRY_INITIAL_DELAY
            ),
            max_delay=(
                args.retry_max_delay
                if args.retry_max_delay is not None
                else RETRY_MAX_DELAY
            ),
            jitter=(
                args.retry_jitter if args.retry_jitter is not None else RETRY_JITTER
            ),
        )
    except ValueError as e:
        print(f"error: invalid retry argument: {e}", file=sys.stderr)
        return None


def cmd_convert(args: argparse.Namespace) -> int:
    retry_config = _build_retry_config(args)
    if retry_config is None:
        return 1
    fallback_to_text = FALLBACK_TO_TEXT and not args.no_fallback_to_text

    if not args.pdf.exists():
        print(f"error: input PDF not found: {args.pdf}", file=sys.stderr)
        return 1

    # D10-N04: fail fast on non-PDF input before any tempdir/cache work.
    try:
        with args.pdf.open("rb") as _pdf_header_fh:
            _pdf_header = _pdf_header_fh.read(5)
    except OSError as _pdf_header_exc:
        print(
            f"error: cannot read input PDF {args.pdf}: {_pdf_header_exc}",
            file=sys.stderr,
        )
        return 1
    if not _pdf_header.startswith(b"%PDF-"):
        print(
            f"error: input file is not a PDF (missing %PDF- header): {args.pdf}",
            file=sys.stderr,
        )
        return 1

    started = time.monotonic()
    keep_intermediates = not args.no_intermediates
    with_summary = not args.no_summary
    no_cache_flags = _resolve_no_cache_flags(args)

    # Resolve --pages against the PDF's actual page count so out-of-range
    # errors surface before we commit to creating a tempdir or doing render work.
    resolved_pages: list[int] | None
    if args.pages is None:
        resolved_pages = None
    else:
        doc = pymupdf.open(args.pdf)
        try:
            resolved_pages = resolve_pages(args.pages, doc.page_count)
        except ValueError as e:
            print(f"error: --pages {args.pages!r}: {e}", file=sys.stderr)
            return 1
        finally:
            doc.close()

    # Defensive empty-pages guard: ``resolve_pages`` raises on a 0-page
    # PDF, so reaching here implies ``--pages`` filtered everything out.
    if args.pages is not None and not resolved_pages:
        print("ERROR: PDF has no pages to process.", file=sys.stderr)
        raise SystemExit(1)

    if keep_intermediates:
        layout, render_target = _resolve_layout(args.pdf, args.intermediates_dir, True)
        return _run_pipeline(
            args=args,
            layout=layout,
            render_target=render_target,
            resolved_pages=resolved_pages,
            keep_intermediates=True,
            with_summary=with_summary,
            retry_config=retry_config,
            fallback_to_text=fallback_to_text,
            started=started,
            no_cache=no_cache_flags,
        )

    with tempfile.TemporaryDirectory(prefix="pdf2md_agent_") as td_str:
        td = Path(td_str)
        pages_dir = td / "pages"
        pages_dir.mkdir()
        layout = CacheLayout(
            root=td,
            pages_dir=pages_dir,
            summary_path=td / "summary.json",
            meta_path=td / "meta.json",
        )
        return _run_pipeline(
            args=args,
            layout=layout,
            render_target=pages_dir,
            resolved_pages=resolved_pages,
            keep_intermediates=False,
            with_summary=with_summary,
            retry_config=retry_config,
            fallback_to_text=fallback_to_text,
            started=started,
            no_cache=no_cache_flags,
        )


def _render_pages(
    *,
    pdf: Path,
    render_target: Path,
    dpi: int,
    resolved_pages: list[int] | None,
    keep_intermediates: bool,
    no_cache_render: bool,
    no_cache_text: bool,
) -> list[PageImage]:
    """Render the PDF, optionally reusing per-page PNG/text cache.

    When ``keep_intermediates`` is True and the no-cache flags are unset,
    pages whose PNG/text are already on disk are returned without touching
    PyMuPDF — that's the trust-cache fast path. With either flag set, the
    pipeline always re-renders / re-extracts.
    """
    if not keep_intermediates or no_cache_render or no_cache_text:
        return render_pdf(pdf, render_target, dpi=dpi, pages=resolved_pages)

    layout = CacheLayout(
        root=render_target.parent,
        pages_dir=render_target,
        summary_path=render_target.parent / "summary.json",
        meta_path=render_target.parent / "meta.json",
    )

    target_pages: list[int] = (
        list(resolved_pages) if resolved_pages is not None
        else list(range(1, _pdf_page_count(pdf) + 1))
    )
    missing: list[int] = [
        n for n in target_pages if _maybe_skip_render(layout, n, dpi) is None
    ]
    if missing:
        render_pdf(pdf, render_target, dpi=dpi, pages=missing)
    pages: list[PageImage] = []
    for n in target_pages:
        png = layout.page_png_path(n)
        with Image.open(png) as img:
            pages.append(PageImage(
                page_number=n,
                width=img.width,
                height=img.height,
                image_path=png,
            ))
    return pages


def _pdf_page_count(pdf: Path) -> int:
    doc = pymupdf.open(pdf)
    try:
        return doc.page_count
    finally:
        doc.close()


def _run_pipeline(
    *,
    args: argparse.Namespace,
    layout: CacheLayout,
    render_target: Path,
    resolved_pages: list[int] | None,
    keep_intermediates: bool,
    with_summary: bool,
    retry_config: RetryConfig,
    fallback_to_text: bool,
    started: float,
    no_cache: CacheNoCacheFlags,
) -> int:
    log.info("converting %s", args.pdf)
    log.info("  output:          %s", args.output)
    log.info("  cache:           %s", layout.root if keep_intermediates else "(tempdir, discarded)")
    log.info("  dpi:             %d", args.dpi)
    log.info("  pages:           %s", "all" if resolved_pages is None else resolved_pages)
    log.info("  no-cache:        %s", no_cache.as_dict())
    log.info("  cross-page:      %s", "summary" if with_summary else "independent")
    log.info("  text-hint:       %s", "on" if not args.no_text_hint else "off")

    if keep_intermediates:
        existing_meta = read_meta(layout.meta_path)
        # ``--no-cache-all`` discards every cached output, so the on-disk
        # fingerprint (about to be overwritten by ``write_meta`` below) is
        # no longer load-bearing — refusing on drift would create a circular
        # error the user can't escape.
        if existing_meta is not None and not no_cache.all():
            reasons = check_meta_matches(
                existing_meta,
                pdf=str(args.pdf.resolve()),
                dpi=args.dpi,
                with_summary=with_summary,
                pages=resolved_pages,
                model=args.model,
                persona_version=args.persona_version,
            )
            if reasons:
                for r in reasons:
                    print(f"error: cache invalid: {r}", file=sys.stderr)
                print(
                    "error: meta.json fingerprint drift detected. "
                    "re-run with --no-cache-all or wipe "
                    f"{layout.root} to rebuild the cache.",
                    file=sys.stderr,
                )
                return 1
        write_meta(
            layout.meta_path,
            pdf=args.pdf,
            dpi=args.dpi,
            with_summary=with_summary,
            pages=resolved_pages,
            model=args.model,
            persona_version=args.persona_version,
        )
        if not with_summary and layout.summary_path.exists():
            layout.summary_path.unlink()

    log.info("rendering PDF to PNGs at %d dpi%s...", args.dpi, " (subset)" if resolved_pages else "")
    pages = _render_pages(
        pdf=args.pdf,
        render_target=render_target,
        dpi=args.dpi,
        resolved_pages=resolved_pages,
        keep_intermediates=keep_intermediates,
        no_cache_render=no_cache.render,
        no_cache_text=no_cache.text,
    )
    log.info("rendered %d page(s) to %s", len(pages), render_target)

    log.info("running pipeline: extract + format%s", " + summarize" if with_summary else "")
    llm = make_vision_llm()
    log.info(
        "  retry:           max_attempts=%s, initial_delay=%.1fs, fibonacci, max_delay=%.1fs, jitter=±%.0f%%",
        retry_config.max_attempts if retry_config.max_attempts is not None else "\u221e",
        retry_config.initial_delay,
        retry_config.max_delay,
        retry_config.jitter * 100,
    )
    log.info("  fallback:        %s", "text layer" if fallback_to_text else "off")
    image_long_side = args.image_long_side if args.image_long_side is not None else IMAGE_LONG_SIDE
    image_jpeg_quality = args.image_quality if args.image_quality is not None else IMAGE_JPEG_QUALITY
    max_summary_chars = args.max_summary_chars if args.max_summary_chars is not None else MAX_SUMMARY_CHARS
    ctx_limit = args.ctx_limit if args.ctx_limit is not None else resolve_ctx_limit()
    log.info(
        "  budget:          ctx_limit=%d, safety=%.0f%%, image_long_side=%dpx, "
        "image_q=%d, max_summary=%d chars",
        ctx_limit,
        TOKEN_BUDGET_SAFETY * 100,
        image_long_side,
        image_jpeg_quality,
        max_summary_chars,
    )
    results = run_pipeline(
        pages=pages,
        layout=layout,
        with_summary=with_summary,
        no_cache=no_cache,
        text_hint=not args.no_text_hint,
        llm=llm,
        retry_config=retry_config,
        fallback_to_text=fallback_to_text,
        ctx_limit=ctx_limit,
        image_long_side=image_long_side,
        image_min_long_side=IMAGE_MIN_LONG_SIDE,
        image_jpeg_quality=image_jpeg_quality,
        max_summary_chars=max_summary_chars,
        token_budget_safety=TOKEN_BUDGET_SAFETY,
        request_timeout_seconds=(
            args.request_timeout
            if args.request_timeout is not None
            else REQUEST_TIMEOUT_SECONDS
        ),
    )

    stitch_mode = StitchMode(args.stitch_mode)
    if stitch_mode is StitchMode.HEURISTIC:
        markdown = stitch_pages(results)
        log.info("  stitch:          heuristic (cross-page merged)")
    else:
        markdown = stitch_pages(results, mode=stitch_mode)
        log.info("  stitch:          off (legacy '---' separator preserved)")
    atomic_write_text(args.output, markdown)
    elapsed = time.monotonic() - started
    log.info(
        "wrote %s — %d page(s), %s chars in %.1fs",
        args.output,
        len(results),
        f"{len(markdown):,}",
        elapsed,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    args = build_parser().parse_args(argv)
    return cmd_convert(args)


if __name__ == "__main__":
    sys.exit(main())
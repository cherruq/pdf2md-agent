"""CLI entry point for convertpdf."""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time

import pymupdf
from pathlib import Path

from convertpdf.cache import CacheLayout, write_meta
from convertpdf.config import (
    CTX_LIMIT,
    FALLBACK_TO_TEXT,
    IMAGE_JPEG_QUALITY,
    IMAGE_LONG_SIDE,
    IMAGE_MIN_LONG_SIDE,
    MAX_SUMMARY_CHARS,
    RETRY_BACKOFF,
    RETRY_INITIAL_DELAY,
    RETRY_JITTER,
    RETRY_MAX_ATTEMPTS,
    RETRY_MAX_DELAY,
    TOKEN_BUDGET_SAFETY,
)
from convertpdf.crew.runner import run_pipeline
from convertpdf.llm_retry import RetryConfig
from convertpdf.pages import parse_page_spec, resolve_pages
from convertpdf.pdf_renderer import render_pdf
from convertpdf.vision import make_vision_llm

log = logging.getLogger("convertpdf")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convertpdf",
        description="Convert a PDF to markdown via a CrewAI vision pipeline (MiniMax-M3).",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    cv = sub.add_parser("convert", help="Render a PDF to markdown.")
    cv.add_argument("pdf", type=Path, help="Input PDF path.")
    cv.add_argument("-o", "--output", type=Path, required=True, help="Output markdown path.")
    cv.add_argument(
        "--dpi",
        type=int,
        default=144,
        help=(
            "Rendering DPI (default: 144). Recommended: "
            "72 (PDF native, smallest), "
            "150 (text + tables), "
            "200 (small fonts / dense formulas), "
            "300+ (print, usually overkill for vision models)."
        ),
    )
    cv.add_argument(
        "-p", "--pages",
        type=parse_page_spec,
        default=None,
        metavar="SPEC",
        help=(
            "Subset of pages to convert. Accepts comma-separated pages and "
            "ranges, e.g. '1-5,8,11-13'. Pages are 1-based; output preserves "
            "original page numbers and is ordered by document position. "
            "Default: all pages."
        ),
    )
    cv.add_argument(
        "--no-intermediates",
        action="store_true",
        help="Skip writing intermediate cache files.",
    )
    cv.add_argument(
        "--intermediates-dir",
        type=Path,
        default=None,
        help="Override the intermediates cache directory (default: .convertpdf-cache/<pdf_stem>/).",
    )
    cv.add_argument(
        "--resume",
        action="store_true",
        help="Reuse cached per-page outputs when present; only re-run missing pages.",
    )
    cv.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable cross-page running summary (process each page independently).",
    )
    cv.add_argument(
        "--no-text-hint",
        action="store_true",
        help="Disable feeding the PDF's native text layer to the extractor.",
    )
    cv.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help=(
            "Total LLM call attempts per page (initial + retries). Overrides "
            "CONVERTPDF_MAX_RETRIES. Default: 4."
        ),
    )
    cv.add_argument(
        "--retry-initial-delay",
        type=float,
        default=None,
        help=(
            "Initial retry delay in seconds (exponential backoff). Overrides "
            "CONVERTPDF_RETRY_INITIAL_DELAY. Default: 1.0."
        ),
    )
    cv.add_argument(
        "--retry-backoff",
        type=float,
        default=None,
        help=(
            "Exponential backoff multiplier between retries. Overrides "
            "CONVERTPDF_RETRY_BACKOFF. Default: 2.0."
        ),
    )
    cv.add_argument(
        "--retry-max-delay",
        type=float,
        default=None,
        help=(
            "Per-attempt retry delay cap in seconds. Overrides "
            "CONVERTPDF_RETRY_MAX_DELAY. Default: 30.0."
        ),
    )
    cv.add_argument(
        "--retry-jitter",
        type=float,
        default=None,
        help=(
            "Jitter ratio in [0.0, 1.0] applied to each retry delay to avoid "
            "thundering-herd. Overrides CONVERTPDF_RETRY_JITTER. Default: 0.25."
        ),
    )
    cv.add_argument(
        "--no-fallback-to-text",
        action="store_false",
        dest="fallback_to_text",
        default=None,
        help=(
            "On retry exhaustion, raise instead of falling back to the PDF's "
            "native text layer. Default: fallback enabled."
        ),
    )
    cv.add_argument(
        "--image-long-side",
        type=int,
        default=None,
        metavar="PX",
        help=(
            "Long-side cap (pixels) for inlined page images. The runner "
            "rescales each page PNG to this size as JPEG at the configured "
            "quality before base64-encoding it. Lower values shrink the per-"
            "call token cost at the expense of OCR fidelity. Overrides "
            "CONVERTPDF_IMAGE_LONG_SIDE. Default: 1536."
        ),
    )
    cv.add_argument(
        "--image-quality",
        type=int,
        default=None,
        metavar="Q",
        help=(
            "JPEG quality (1-95) used when the runner downsamples page "
            "images. Higher values preserve detail but enlarge the per-"
            "call token cost. Overrides CONVERTPDF_IMAGE_JPEG_QUALITY. "
            "Default: 85."
        ),
    )
    cv.add_argument(
        "--max-summary-chars",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum running-summary size (characters) fed into the next "
            "page's extract call and produced by the summarizer. Overrides "
            "CONVERTPDF_MAX_SUMMARY_CHARS. Default: 800."
        ),
    )
    cv.add_argument(
        "--ctx-limit",
        type=int,
        default=None,
        metavar="TOK",
        help=(
            "Model context-window token limit the runner budgets against. "
            "Used only when CONVERTPDF_CTX_LIMIT is wrong. Default: 2013."
        ),
    )
    return parser


def _resolve_layout(
    pdf: Path,
    override: Path | None,
    keep_intermediates: bool,
) -> tuple[CacheLayout, Path]:
    """Return ``(layout, render_target_pages_dir)``.

    When ``keep_intermediates`` is False the layout lives under a tempdir that
    is removed on context exit.
    """
    if keep_intermediates:
        root = override if override is not None else Path(".convertpdf-cache") / pdf.stem
        return CacheLayout.for_pdf(root, pdf), root / "pages"

    td = Path(tempfile.mkdtemp(prefix="convertpdf_"))
    pages = td / "pages"
    pages.mkdir()
    return (
        CacheLayout(
            root=td,
            pages_dir=pages,
            summary_path=td / "summary.json",
            meta_path=td / "meta.json",
        ),
        pages,
    )


def _build_retry_config(args: argparse.Namespace) -> RetryConfig | None:
    """Build a RetryConfig from CLI args (override) + env (fallback). Returns None on invalid input."""
    try:
        return RetryConfig(
            max_attempts=(
                args.max_retries if args.max_retries is not None else RETRY_MAX_ATTEMPTS
            ),
            initial_delay=(
                args.retry_initial_delay
                if args.retry_initial_delay is not None
                else RETRY_INITIAL_DELAY
            ),
            backoff=(
                args.retry_backoff if args.retry_backoff is not None else RETRY_BACKOFF
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
    fallback_to_text = (
        args.fallback_to_text if args.fallback_to_text is not None else FALLBACK_TO_TEXT
    )

    if not args.pdf.exists():
        print(f"error: input PDF not found: {args.pdf}", file=sys.stderr)
        return 1

    started = time.monotonic()
    keep_intermediates = not args.no_intermediates
    with_summary = not args.no_summary

    layout, render_target = _resolve_layout(args.pdf, args.intermediates_dir, keep_intermediates)

    # Resolve --pages against the PDF's actual page count so out-of-range
    # errors surface before any rendering work happens.
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

    log.info("converting %s", args.pdf)
    log.info("  output:          %s", args.output)
    log.info("  cache:           %s", layout.root if keep_intermediates else "(tempdir, discarded)")
    log.info("  dpi:             %d", args.dpi)
    log.info("  pages:           %s", "all" if resolved_pages is None else resolved_pages)
    log.info("  cross-page:      %s", "summary" if with_summary else "independent")
    log.info("  resume:          %s", "yes" if args.resume else "no")
    log.info("  text-hint:       %s", "on" if not args.no_text_hint else "off")

    if keep_intermediates:
        write_meta(
            layout.meta_path,
            pdf=args.pdf,
            dpi=args.dpi,
            with_summary=with_summary,
            pages=resolved_pages,
        )

    log.info("rendering PDF to PNGs at %d dpi%s...", args.dpi, " (subset)" if resolved_pages else "")
    pages = render_pdf(args.pdf, render_target, dpi=args.dpi, pages=resolved_pages)
    log.info("rendered %d page(s) to %s", len(pages), render_target)

    log.info("running pipeline: extract + format%s", " + summarize" if with_summary else "")
    llm = make_vision_llm()
    log.info(
        "  retry:           max_attempts=%d, initial_delay=%.1fs, backoff=%.1fx, max_delay=%.1fs, jitter=±%.0f%%",
        retry_config.max_attempts,
        retry_config.initial_delay,
        retry_config.backoff,
        retry_config.max_delay,
        retry_config.jitter * 100,
    )
    log.info("  fallback:        %s", "text layer" if fallback_to_text else "off")
    image_long_side = args.image_long_side if args.image_long_side is not None else IMAGE_LONG_SIDE
    image_jpeg_quality = args.image_quality if args.image_quality is not None else IMAGE_JPEG_QUALITY
    max_summary_chars = args.max_summary_chars if args.max_summary_chars is not None else MAX_SUMMARY_CHARS
    ctx_limit = args.ctx_limit if args.ctx_limit is not None else CTX_LIMIT
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
        resume=args.resume,
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
    )

    markdown = "\n\n---\n\n".join(r.markdown for r in results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
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
    match args.command:
        case "convert":
            return cmd_convert(args)
        case _:
            raise AssertionError(f"unreachable command: {args.command!r}")  # subparser is required


if __name__ == "__main__":
    sys.exit(main())
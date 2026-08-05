"""CLI entry point for pdf2md-agent."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pymupdf

from pdf2md_agent.cache import (
    CacheLayout as CacheLayout,  # noqa: F401
    CacheNoCacheFlags as CacheNoCacheFlags,  # noqa: F401
    atomic_write_text as atomic_write_text,  # noqa: F401
    read_meta as read_meta,  # noqa: F401
    write_meta as write_meta,  # noqa: F401
)
from pdf2md_agent.cli_parser import (
    _resolve_layout,
    _resolve_no_cache_flags,
    build_parser,
    build_retry_config,
)
from pdf2md_agent.config import (
    IMAGE_JPEG_QUALITY,
    IMAGE_LONG_SIDE,
    REQUEST_TIMEOUT_SECONDS,
    ConversionConfig,
    resolve_ctx_limit,
)
from pdf2md_agent.tuning import DEFAULT_STITCH_MODE
from pdf2md_agent.crew.runner import run_pipeline as run_pipeline  # noqa: F401
from pdf2md_agent.llm_retry import RetryConfig
from pdf2md_agent.pages import resolve_pages
from pdf2md_agent.crew.types import (
    RenderedPage as RenderedPage,  # noqa: F401
)
from pdf2md_agent.pdf_renderer import (
    render_pages as _render_pages,  # noqa: F401
    render_pdf as render_pdf,  # noqa: F401
)
from pdf2md_agent.pipeline import run_unified_conversion
from pdf2md_agent.post_stream import stitch_pages as stitch_pages  # noqa: F401
from pdf2md_agent.vision import make_vision_llm as make_vision_llm  # noqa: F401

log = logging.getLogger("pdf2md-agent")


# --- Entrypoint -------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    args = build_parser().parse_args(argv)
    return cmd_convert(args)


# --- Input validation -------------------------------------------------------


def _validate_pdf_header(pdf: Path) -> int | None:
    """Return exit code if the file is missing, unreadable, or not a PDF; else None."""
    if not pdf.exists():
        print(f"error: input PDF not found: {pdf}", file=sys.stderr)
        return 1
    try:
        with pdf.open("rb") as fh:
            header = fh.read(5)
    except OSError as exc:
        print(
            f"error: cannot read input PDF {pdf}: {exc}",
            file=sys.stderr,
        )
        return 1
    if not header.startswith(b"%PDF-"):
        print(
            f"error: input file is not a PDF (missing %PDF- header): {pdf}",
            file=sys.stderr,
        )
        return 1
    return None


def _resolve_requested_pages(pdf: Path, pages_spec: object) -> tuple[list[int] | None, int | None]:
    """Validate ``pages_spec`` against the PDF's page count."""
    if pages_spec is None:
        return None, None
    doc = pymupdf.open(pdf)
    try:
        resolved = resolve_pages(pages_spec, doc.page_count)
    except ValueError as exc:
        print(f"error: --pages {pages_spec!r}: {exc}", file=sys.stderr)
        return None, 1
    finally:
        doc.close()
    if not resolved:
        print("ERROR: PDF has no pages to process.", file=sys.stderr)
        return None, 1
    return resolved, None


# --- Orchestration Gateway --------------------------------------------------


def cmd_convert(args: argparse.Namespace) -> int:
    retry_config = build_retry_config(args)
    if retry_config is None:
        return 1

    header_exit = _validate_pdf_header(args.pdf)
    if header_exit is not None:
        return header_exit

    no_cache_flags = _resolve_no_cache_flags(args)
    resolved_pages, pages_exit = _resolve_requested_pages(args.pdf, args.pages)
    if pages_exit is not None:
        return pages_exit

    layout, render_target = _resolve_layout(args.pdf, args.intermediates_dir)

    config = ConversionConfig(
        pdf=args.pdf,
        output=args.output,
        dpi=args.dpi,
        layout=layout,
        render_target=render_target,
        resolved_pages=resolved_pages,
        no_cache=no_cache_flags,
        retry_config=retry_config,
        text_hint=not getattr(args, "no_text_hint", False),
        image_long_side=getattr(args, "image_long_side", None) or IMAGE_LONG_SIDE,
        image_jpeg_quality=getattr(args, "image_quality", None) or IMAGE_JPEG_QUALITY,
        ctx_limit=getattr(args, "ctx_limit", None) or resolve_ctx_limit(),
        request_timeout_seconds=getattr(args, "request_timeout", None) or REQUEST_TIMEOUT_SECONDS,
        stitch_mode=getattr(args, "stitch_mode", DEFAULT_STITCH_MODE),
        started=time.monotonic(),
    )
    return run_unified_conversion(config)





# Step 1, 2, 3 aliases for clear 3-step architecture
step1_render_and_sync_cache = _render_pages
step2_extract_pages = run_pipeline
step3_stitch_and_clean = stitch_pages

__all__ = [
    "build_parser",
    "cmd_convert",
    "main",
    "run_unified_conversion",
    "step1_render_and_sync_cache",
    "step2_extract_pages",
    "step3_stitch_and_clean",
]

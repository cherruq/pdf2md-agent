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
from convertpdf.crew.runner import run_pipeline
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


def cmd_convert(args: argparse.Namespace) -> int:
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
    results = run_pipeline(
        pages=pages,
        layout=layout,
        with_summary=with_summary,
        resume=args.resume,
        text_hint=not args.no_text_hint,
        llm=llm,
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
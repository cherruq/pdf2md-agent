"""Render a PDF to per-page PNG images + native text layer via PyMuPDF."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True, slots=True)
class PageImage:
    """One rendered PDF page."""

    page_number: int
    width: int
    height: int
    image_path: Path


def render_pdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = 144,
    prefix: str = "page",
    pages: list[int] | None = None,
) -> list[PageImage]:
    """Render ``pdf_path`` into per-page PNGs under ``output_dir``.

    If ``pages`` is ``None`` (default), renders every page in document
    order. If ``pages`` is a list of 1-based page numbers, renders only
    those pages (still in document order — the list is sorted internally)
    and skips the rest. Output filenames always use the **original**
    1-based page number, so cache directories are stable across calls
    with different ``pages`` selections.

    For each rendered page, also writes a sibling
    ``{prefix}_{NNNN}_text.txt`` containing the PDF's native text layer
    (empty for scanned pages).

    Returns the pages in document order. Caller is responsible for
    ``output_dir`` existing; the function writes into it but does not
    create it.
    """
    doc = pymupdf.open(pdf_path)
    try:
        pages_out: list[PageImage] = []
        zoom = dpi / 72
        matrix = pymupdf.Matrix(zoom, zoom)

        if pages is None:
            page_numbers = list(range(1, doc.page_count + 1))
        else:
            page_numbers = sorted(set(pages))

        for page_number in page_numbers:
            page = doc.load_page(page_number - 1)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            png, text = _page_artifact_paths(output_dir, prefix, page_number)
            pix.save(png)
            text.write_text(page.get_text("text", sort=True), encoding="utf-8")
            pages_out.append(
                PageImage(
                    page_number=page_number,
                    width=pix.width,
                    height=pix.height,
                    image_path=png,
                )
            )
        return pages_out
    finally:
        doc.close()


def _page_artifact_paths(
    output_dir: Path, prefix: str, page_number: int
) -> tuple[Path, Path]:
    """Return ``(png_path, text_path)`` for one rendered page.

    Per-page filenames embed the 1-based ``page_number``, so each call
    produces a fresh ``Path`` pair; the helper exists to consolidate the
    construction (matches the layout used by :mod:`pdf2md_agent.cache`)
    and keep the render loop readable.
    """
    stem = f"{prefix}_{page_number:04d}"
    return output_dir / f"{stem}.png", output_dir / f"{stem}_text.txt"


def read_page_text(text_path: Path) -> str:
    """Read a per-page text file written by :func:`render_pdf`."""
    if not text_path.exists():
        return ""
    return text_path.read_text(encoding="utf-8")


def pdf_page_count(pdf: Path) -> int:
    """Return the total page count of a PDF file via PyMuPDF."""
    doc = pymupdf.open(pdf)
    try:
        return doc.page_count
    finally:
        doc.close()


def render_pages(
    *,
    pdf: Path,
    render_target: Path,
    dpi: int,
    resolved_pages: list[int] | None,
    no_cache_render: bool,
    no_cache_text: bool,
) -> list[PageImage]:
    """Render the PDF, invalidating step 2 cache on real-time text drift or no-cache flags."""
    if no_cache_render or no_cache_text:
        return render_pdf(pdf, render_target, dpi=dpi, pages=resolved_pages)

    from PIL import Image
    from pdf2md_agent.cache import CacheLayout

    layout = CacheLayout(
        root=render_target.parent,
        pages_dir=render_target,
        meta_path=render_target.parent / "meta.json",
    )

    target_pages = (
        list(resolved_pages) if resolved_pages is not None
        else list(range(1, pdf_page_count(pdf) + 1))
    )

    doc = pymupdf.open(pdf)
    try:
        missing_or_drifted: list[int] = []
        for n in target_pages:
            png = layout.page_png_path(n)
            txt = layout.page_text_path(n)
            if not png.is_file():
                missing_or_drifted.append(n)
                continue

            # Step 1 real-time check: compare freshly extracted PyMuPDF text against disk cache.
            page = doc.load_page(n - 1)
            new_text = page.get_text("text", sort=True)
            if txt.exists():
                old_text = txt.read_text(encoding="utf-8")
                if new_text != old_text:
                    # Inconsistent! Delete this page's Step 2 cache files and re-render.
                    layout.page_format_path(n).unlink(missing_ok=True)
                    layout.page_extract_path(n).unlink(missing_ok=True)
                    for jpg_path in render_target.glob(f"page_{n:04d}_*.jpg"):
                        jpg_path.unlink(missing_ok=True)
                    missing_or_drifted.append(n)
            else:
                txt.write_text(new_text, encoding="utf-8")

        if missing_or_drifted:
            render_pdf(pdf, render_target, dpi=dpi, pages=missing_or_drifted)

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
    finally:
        doc.close()


# Step 1 entry point alias for the unified conversion pipeline
step1_render_and_sync_cache = render_pages

__all__ = [
    "PageImage",
    "render_pdf",
    "read_page_text",
    "pdf_page_count",
    "render_pages",
    "step1_render_and_sync_cache",
]
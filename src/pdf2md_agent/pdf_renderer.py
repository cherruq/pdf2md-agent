"""Render a PDF to per-page PNG images + native text layer via PyMuPDF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pymupdf

if TYPE_CHECKING:
    from pdf2md_agent.config import ConversionConfig


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """One rendered PDF page: raster image artifact and native text layer."""

    page_number: int
    width: int
    height: int
    image_path: Path
    text_path: Path | None = None
    text: str = ""


def render_pdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = 144,
    prefix: str = "page",
    pages: list[int] | None = None,
) -> list[RenderedPage]:
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
        zoom = dpi / 72
        matrix = pymupdf.Matrix(zoom, zoom)
        page_numbers = list(range(1, doc.page_count + 1)) if pages is None else sorted(set(pages))
        pages_out: list[RenderedPage] = []
        for page_number in page_numbers:
            png, text = _page_artifact_paths(output_dir, prefix, page_number)
            page = doc.load_page(page_number - 1)
            pages_out.append(_render_single_page(page, page_number, png, text, matrix))
        return pages_out
    finally:
        doc.close()


def _render_single_page(
    page: pymupdf.Page,
    page_number: int,
    png_path: Path,
    text_path: Path,
    matrix: pymupdf.Matrix,
    *,
    extracted_text: str | None = None,
) -> RenderedPage:
    """Render a single PyMuPDF page to PNG and write its native text layer."""
    if extracted_text is None:
        extracted_text = page.get_text("text", sort=True)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    pix.save(png_path)
    text_path.write_text(extracted_text, encoding="utf-8")
    return RenderedPage(
        page_number=page_number,
        width=pix.width,
        height=pix.height,
        image_path=png_path,
        text_path=text_path,
        text=extracted_text,
    )


def _page_artifact_paths(output_dir: Path, prefix: str, page_number: int) -> tuple[Path, Path]:
    """Return ``(png_path, text_path)`` for one rendered page.

    Per-page filenames embed the 1-based ``page_number``, so each call
    produces a fresh ``Path`` pair; the helper exists to consolidate the
    construction (matches the layout used by :mod:`pdf2md_agent.cache`)
    and keep the render loop readable.
    """
    stem = f"{prefix}_{page_number:04d}"
    return output_dir / f"{stem}.png", output_dir / f"{stem}_text.txt"


def read_page_text(text_path: Path) -> str:
    """Read a per-page text file written by :func:`render_pdf`, safely ignoring I/O errors."""
    try:
        if not text_path.exists():
            return ""
        return text_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _is_cached_png_valid(png_path: Path) -> bool:
    """Return whether ``png_path`` exists and is non-empty without raising I/O errors."""
    try:
        return png_path.is_file() and png_path.stat().st_size > 0
    except OSError:
        return False


def _is_cached_text_valid(txt_path: Path, expected_text: str) -> bool:
    """Return whether ``txt_path`` exists and matches ``expected_text`` without raising I/O errors."""
    try:
        if not txt_path.is_file():
            return False
        return txt_path.read_text(encoding="utf-8") == expected_text
    except (OSError, UnicodeDecodeError):
        return False


def pdf_page_count(pdf: Path) -> int:
    """Return the total page count of a PDF file via PyMuPDF."""
    doc = pymupdf.open(pdf)
    try:
        return doc.page_count
    finally:
        doc.close()


def render_pages(config: ConversionConfig) -> list[RenderedPage]:
    """Render the PDF, invalidating step 2 cache on real-time text drift or no-cache flags."""
    if config.no_cache.render or config.no_cache.text:
        return render_pdf(config.pdf, config.render_target, dpi=config.dpi, pages=config.resolved_pages)

    from PIL import Image

    layout = config.layout
    with pymupdf.open(config.pdf) as doc:
        target_pages = (
            list(config.resolved_pages) if config.resolved_pages is not None else list(range(1, doc.page_count + 1))
        )
        zoom = config.dpi / 72
        matrix = pymupdf.Matrix(zoom, zoom)
        pages: list[RenderedPage] = []

        for n in target_pages:
            png = layout.page_png_path(n)
            txt = layout.page_text_path(n)
            page = doc.load_page(n - 1)

            need_png = not _is_cached_png_valid(png)
            new_text = page.get_text("text", sort=True)
            text_valid = _is_cached_text_valid(txt, new_text)

            if not text_valid:
                # 无效分支：尝试删后续资源（Step 2 产物与缩放 JPEG）
                layout.page_format_path(n).unlink(missing_ok=True)
                for jpg_path in config.render_target.glob(f"page_{n:04d}_*.jpg"):
                    jpg_path.unlink(missing_ok=True)

            if need_png or not text_valid:
                pages.append(_render_single_page(page, n, png, txt, matrix, extracted_text=new_text))
            else:
                # 有效分支：缓存直接可用
                with Image.open(png) as img:
                    pages.append(
                        RenderedPage(
                            page_number=n,
                            width=img.width,
                            height=img.height,
                            image_path=png,
                            text_path=txt,
                            text=new_text,
                        )
                    )
        return pages


__all__ = [
    "RenderedPage",
    "read_page_text",
    "render_pages",
    "render_pdf",
]

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
) -> list[PageImage]:
    """Render every page of ``pdf_path`` into a PNG under ``output_dir``.

    For each page, also writes a sibling ``{prefix}_{NNNN}_text.txt`` containing
    the PDF's native text layer (empty for scanned pages).

    Returns the pages in document order. Caller is responsible for ``output_dir``
    existing; the function writes into it but does not create it.
    """
    doc = pymupdf.open(pdf_path)
    try:
        pages: list[PageImage] = []
        zoom = dpi / 72
        matrix = pymupdf.Matrix(zoom, zoom)
        for index, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            png = output_dir / f"{prefix}_{index:04d}.png"
            pix.save(png)
            text = output_dir / f"{prefix}_{index:04d}_text.txt"
            text.write_text(page.get_text("text"), encoding="utf-8")
            pages.append(
                PageImage(
                    page_number=index,
                    width=pix.width,
                    height=pix.height,
                    image_path=png,
                )
            )
        return pages
    finally:
        doc.close()


def read_page_text(text_path: Path) -> str:
    """Read a per-page text file written by :func:`render_pdf`."""
    if not text_path.exists():
        return ""
    return text_path.read_text(encoding="utf-8")
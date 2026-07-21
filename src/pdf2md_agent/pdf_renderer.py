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
            text.write_text(page.get_text("text"), encoding="utf-8")
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
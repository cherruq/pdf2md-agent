"""Tests for pdf2md_agent.pdf_renderer."""
from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from pdf2md_agent.pdf_renderer import render_pdf


def _make_pdf(path: Path, pages: int = 2) -> Path:
    doc = pymupdf.open()
    try:
        for i in range(pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"page {i + 1}")
        doc.save(str(path))
    finally:
        doc.close()
    return path


def test_render_pdf_writes_one_png_per_page(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "tiny.pdf", pages=3)
    out = tmp_path / "pages"
    out.mkdir()

    pages = render_pdf(pdf, out, dpi=72)

    assert len(pages) == 3
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert all(p.image_path.exists() for p in pages)
    assert all(p.width > 0 and p.height > 0 for p in pages)


def test_render_pdf_higher_dpi_yields_larger_pixels(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "tiny.pdf", pages=1)
    out = tmp_path / "pages"
    out.mkdir()

    low = render_pdf(pdf, out, dpi=72, prefix="low")
    high = render_pdf(pdf, out, dpi=288, prefix="high")

    assert high[0].width > low[0].width
    assert high[0].height > low[0].height


def test_render_pdf_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        render_pdf(tmp_path / "ghost.pdf", tmp_path / "pages", dpi=72)


def test_render_pdf_subset_writes_only_requested_pages(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "tiny.pdf", pages=3)
    out = tmp_path / "pages"
    out.mkdir()

    pages = render_pdf(pdf, out, dpi=72, pages=[2])

    assert len(pages) == 1
    assert pages[0].page_number == 2
    assert (out / "page_0002.png").exists()
    assert (out / "page_0002_text.txt").exists()
    assert not (out / "page_0001.png").exists()
    assert not (out / "page_0003.png").exists()


def test_render_pdf_subset_preserves_original_page_numbers(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "tiny.pdf", pages=5)
    out = tmp_path / "pages"
    out.mkdir()

    pages = render_pdf(pdf, out, dpi=72, pages=[3, 1])

    # Sorted ascending in the returned list.
    assert [p.page_number for p in pages] == [1, 3]
    # But output filenames use the ORIGINAL page number.
    assert (out / "page_0001.png").exists()
    assert (out / "page_0003.png").exists()
    assert not (out / "page_0002.png").exists()
    assert not (out / "page_0004.png").exists()
    assert not (out / "page_0005.png").exists()


def test_render_pdf_subset_full_coverage(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "tiny.pdf", pages=2)
    out = tmp_path / "pages"
    out.mkdir()

    pages = render_pdf(pdf, out, dpi=72, pages=[1, 2])

    assert [p.page_number for p in pages] == [1, 2]
    assert (out / "page_0001.png").exists()
    assert (out / "page_0002.png").exists()
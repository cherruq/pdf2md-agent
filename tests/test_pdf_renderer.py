"""Tests for pdf2md_agent.pdf_renderer."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from pdf2md_agent.cache import CacheLayout, CacheNoCacheFlags
from pdf2md_agent.config import ConversionConfig
from pdf2md_agent.llm_retry import RetryConfig
from pdf2md_agent.pdf_renderer import RenderedPage, render_pages, render_pdf


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


def test_render_pages_invalidates_step2_cache_on_text_drift(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=1)
    out = tmp_path / "cache" / "pages"
    out.mkdir(parents=True, exist_ok=True)

    layout = CacheLayout(root=out.parent, pages_dir=out, meta_path=out.parent / "meta.json")
    config = ConversionConfig(
        pdf=pdf,
        output=tmp_path / "out.md",
        dpi=72,
        layout=layout,
        render_target=out,
        resolved_pages=None,
        no_cache=CacheNoCacheFlags(),
        retry_config=RetryConfig(),
        text_hint=True,
        image_long_side=1536,
        image_jpeg_quality=85,
        ctx_limit=256,
        request_timeout_seconds=60.0,
    )

    # First run: renders page 1 and saves text cache
    render_pages(config)
    assert (out / "page_0001.png").exists()
    assert (out / "page_0001_text.txt").exists()

    # Simulate Step 2 producing cache files
    format_path = out / "page_0001_format.md"
    format_path.write_text("# Old Content", encoding="utf-8")

    # Tamper with the saved text file on disk to simulate drift from the real PDF text
    (out / "page_0001_text.txt").write_text("Drifted old text on disk", encoding="utf-8")

    # Second run: Step 1 detects drift between real PyMuPDF text and disk text, invalidating Step 2 cache
    render_pages(config)

    assert not format_path.exists(), "Step 1 must delete format.md on text drift"
    assert "page 1" in (out / "page_0001_text.txt").read_text(encoding="utf-8")


def test_render_pages_returns_rendered_page_with_text(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=1)
    out = tmp_path / "cache" / "pages"
    out.mkdir(parents=True, exist_ok=True)
    layout = CacheLayout(root=out.parent, pages_dir=out, meta_path=out.parent / "meta.json")
    config = ConversionConfig(
        pdf=pdf,
        output=tmp_path / "out.md",
        dpi=72,
        layout=layout,
        render_target=out,
        resolved_pages=None,
        no_cache=CacheNoCacheFlags(),
        retry_config=RetryConfig(),
        text_hint=True,
        image_long_side=1536,
        image_jpeg_quality=85,
        ctx_limit=256,
        request_timeout_seconds=60.0,
    )
    pages = render_pages(config)
    assert len(pages) == 1
    assert isinstance(pages[0], RenderedPage)
    assert pages[0].text_path == out / "page_0001_text.txt"
    assert "page 1" in pages[0].text


def test_render_pages_regenerates_zero_byte_png(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=1)
    out = tmp_path / "cache" / "pages"
    out.mkdir(parents=True, exist_ok=True)
    layout = CacheLayout(root=out.parent, pages_dir=out, meta_path=out.parent / "meta.json")
    config = ConversionConfig(
        pdf=pdf,
        output=tmp_path / "out.md",
        dpi=72,
        layout=layout,
        render_target=out,
        resolved_pages=None,
        no_cache=CacheNoCacheFlags(),
        retry_config=RetryConfig(),
        text_hint=True,
        image_long_side=1536,
        image_jpeg_quality=85,
        ctx_limit=256,
        request_timeout_seconds=60.0,
    )
    render_pages(config)
    png_path = out / "page_0001.png"
    assert png_path.stat().st_size > 0

    # Simulate a corrupted / zero-byte PNG file from an aborted process
    png_path.write_bytes(b"")
    assert png_path.stat().st_size == 0

    # Running render_pages must detect the 0-byte file and regenerate it
    render_pages(config)
    assert png_path.stat().st_size > 0


def test_render_pages_handles_corrupt_text_cache_without_crashing(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=1)
    out = tmp_path / "cache" / "pages"
    out.mkdir(parents=True, exist_ok=True)
    layout = CacheLayout(root=out.parent, pages_dir=out, meta_path=out.parent / "meta.json")
    config = ConversionConfig(
        pdf=pdf,
        output=tmp_path / "out.md",
        dpi=72,
        layout=layout,
        render_target=out,
        resolved_pages=None,
        no_cache=CacheNoCacheFlags(),
        retry_config=RetryConfig(),
        text_hint=True,
        image_long_side=1536,
        image_jpeg_quality=85,
        ctx_limit=256,
        request_timeout_seconds=60.0,
    )
    render_pages(config)
    txt_path = out / "page_0001_text.txt"

    # Write invalid UTF-8 bytes to simulate a corrupt cache file
    txt_path.write_bytes(b"\xff\xfe\x00\x80\x81")

    # render_pages must safely handle the decoding error, treat the cache as invalid, and rewrite the valid text
    pages = render_pages(config)
    assert "page 1" in txt_path.read_text(encoding="utf-8")
    assert "page 1" in pages[0].text

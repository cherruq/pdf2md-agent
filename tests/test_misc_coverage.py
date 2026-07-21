"""Misc coverage: cache, pdf_renderer.read_page_text, runner._strip_think, CLI smoke."""
from __future__ import annotations

import json
import logging
import pymupdf
from pathlib import Path

import pytest

from pdf2md_agent import cli
from pdf2md_agent.cache import (
    CacheLayout,
    is_page_complete,
    read_summary,
    write_summary,
)
from pdf2md_agent.crew.runner import _strip_think
from pdf2md_agent.pdf_renderer import PageImage, read_page_text, render_pdf


# --- CacheLayout ----------------------------------------------------------


def test_cache_layout_for_pdf_creates_subdirs(tmp_path: Path) -> None:
    root = tmp_path / "out"
    layout = CacheLayout.for_pdf(root, tmp_path / "x.pdf")
    assert layout.root == root
    assert layout.pages_dir == root / "pages"
    assert layout.pages_dir.is_dir()
    assert (root / "summary.json").parent == root
    assert layout.meta_path == root / "meta.json"
    assert layout.summary_path == root / "summary.json"


def test_cache_layout_artifacts_for_round_trip(tmp_path: Path) -> None:
    layout = CacheLayout.for_pdf(tmp_path / "out", tmp_path / "x.pdf")
    page = PageImage(page_number=3, width=100, height=100, image_path=tmp_path / "p3.png")
    a = layout.artifacts_for(page)
    assert a.page_number == 3
    assert a.page_png == layout.page_png_path(3)
    assert a.page_text == layout.page_text_path(3)
    assert a.extract_text == layout.page_extract_path(3)
    assert a.format_markdown == layout.page_format_path(3)


def test_is_page_complete_true_when_both_outputs_exist(tmp_path: Path) -> None:
    layout = CacheLayout.for_pdf(tmp_path / "out", tmp_path / "x.pdf")
    layout.page_extract_path(1).write_text("extract", encoding="utf-8")
    layout.page_format_path(1).write_text("md", encoding="utf-8")
    assert is_page_complete(layout, 1) is True


def test_is_page_complete_false_when_one_output_missing(tmp_path: Path) -> None:
    layout = CacheLayout.for_pdf(tmp_path / "out", tmp_path / "x.pdf")
    layout.page_extract_path(2).write_text("extract", encoding="utf-8")
    # format_markdown missing
    assert is_page_complete(layout, 2) is False
    layout.page_format_path(3).write_text("md", encoding="utf-8")
    # extract missing
    assert is_page_complete(layout, 3) is False


# --- read_summary / write_summary -----------------------------------------


def test_read_summary_missing_returns_empty(tmp_path: Path) -> None:
    assert read_summary(tmp_path / "nope.json") == ""


def test_read_write_summary_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    write_summary(path, "running sum text 中文")
    assert read_summary(path) == "running sum text 中文"


def test_read_summary_corrupt_returns_empty(tmp_path: Path, caplog) -> None:
    path = tmp_path / "summary.json"
    path.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="pdf2md_agent.cache"):
        assert read_summary(path) == ""
    assert any("unreadable" in rec.message for rec in caplog.records)


def test_read_summary_wrong_shape_returns_empty(tmp_path: Path, caplog) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="pdf2md_agent.cache"):
        assert read_summary(path) == ""
    assert any("not a JSON object" in rec.message for rec in caplog.records)


# --- pdf_renderer.read_page_text ------------------------------------------


def _make_onepage_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    try:
        doc.new_page().insert_text((72, 72), "page 1")
        doc.save(str(path))
    finally:
        doc.close()
    return path


def test_read_page_text_missing_returns_empty(tmp_path: Path) -> None:
    assert read_page_text(tmp_path / "no.txt") == ""


def test_read_page_text_round_trip(tmp_path: Path) -> None:
    pdf = _make_onepage_pdf(tmp_path / "x.pdf")
    pages = render_pdf(pdf, tmp_path, dpi=72)
    assert len(pages) == 1
    txt = read_page_text(pages[0].image_path.with_name(
        f"page_{pages[0].page_number:04d}_text.txt"
    ))
    assert "page 1" in txt


# --- runner._strip_think ---------------------------------------------------


def test_strip_think_removes_single_block() -> None:
    assert _strip_think("before<think>scratch</think>after") == "beforeafter"


def test_strip_think_removes_multiple_blocks() -> None:
    text = "head<think>a</think>mid<think>b</think>tail"
    assert _strip_think(text) == "headmidtail"


def test_strip_think_no_block_returns_unchanged() -> None:
    assert _strip_think("plain answer") == "plain answer"


def test_strip_think_strips_whitespace() -> None:
    assert _strip_think("  answer text  \n") == "answer text"


# --- CLI smoke ------------------------------------------------------------


def test_cli_parse_known_args() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["in.pdf", "-o", "out.md"])
    assert args.pdf == Path("in.pdf")
    assert args.output == Path("out.md")
    assert args.dpi == 144
    assert args.pages is None
    assert args.no_intermediates is False
    assert args.no_summary is False
    assert args.no_text_hint is False
    assert args.no_fallback_to_text is False  # default — env may still override


def test_cli_parse_pages_spec() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["in.pdf", "-o", "x.md", "-p", "1-5,8"])
    assert args.pages == [1, 2, 3, 4, 5, 8]


def test_cli_parse_rejects_invalid_pages() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["in.pdf", "-o", "x.md", "-p", "0"])


def test_cli_main_missing_pdf_returns_1(capsys) -> None:
    rc = cli.main(["/no/such/file.pdf", "-o", "/tmp/out.md"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "input PDF not found" in err

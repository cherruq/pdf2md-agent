"""Misc coverage: cache, pdf_renderer.read_page_text, runner._strip_think, CLI smoke."""
from __future__ import annotations

import hashlib
import json
import logging
import pymupdf
from pathlib import Path
from unittest.mock import patch

import pytest

from pdf2md_agent import cli
from pdf2md_agent.cache import (
    CacheCorruptedError,
    CacheLayout,
    is_page_complete,
    read_summary,
    write_summary,
)
from pdf2md_agent.cli import _atomic_write_text, _safe_cache_stem, _safe_intermediates_dir
from pdf2md_agent.config import MODEL_NAME
from pdf2md_agent.crew import agents
from pdf2md_agent.crew.multimodal_patch import ImageEncodeError, _encode_local_image
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


def test_read_summary_corrupt_raises_and_warns(tmp_path: Path, caplog) -> None:
    path = tmp_path / "summary.json"
    path.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="pdf2md_agent.cache"):
        with pytest.raises(CacheCorruptedError):
            read_summary(path)
    assert any("unreadable" in rec.message for rec in caplog.records)


def test_read_summary_wrong_shape_raises_and_warns(tmp_path: Path, caplog) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="pdf2md_agent.cache"):
        with pytest.raises(CacheCorruptedError):
            read_summary(path)
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
    assert args.no_fallback_to_text is False
    assert args.model == MODEL_NAME
    assert args.persona_version == agents.PERSONA_VERSION


def test_help_lists_argument_groups() -> None:
    """The --help output must surface the four logical groups so users
    can discover flags without reading the README."""
    parser = cli.build_parser()
    help_text = parser.format_help()
    for group in ("Pipeline", "Cache control", "Feature disable", "Retry & tuning"):
        assert group in help_text, f"missing help group: {group}"
    assert "Diagnostic" in help_text


def test_persona_version_hashes_active_personas() -> None:
    joined = "\x00".join(
        (
            agents.EXTRACTOR_PERSONA,
            agents.FORMATTER_PERSONA_STRICT,
            agents.SUMMARIZER_PERSONA,
        )
    )
    assert agents.PERSONA_VERSION == hashlib.sha256(joined.encode()).hexdigest()[:16]
    assert "PERSONA_VERSION" in agents.__all__


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


def test_cli_version_prints_and_exits(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "pdf2md-agent" in out
    from pdf2md_agent import __about__
    assert __about__.__version__ in out


def test_cli_request_timeout_rejects_zero() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["in.pdf", "-o", "x.md", "--request-timeout", "0"])


def test_cli_request_timeout_rejects_negative() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["in.pdf", "-o", "x.md", "--request-timeout", "-1"])


def test_cli_max_retries_rejects_zero() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["in.pdf", "-o", "x.md", "--max-retries", "0"])


def test_encode_local_image_non_image_raises_image_encode_error(tmp_path: Path) -> None:
    bogus = tmp_path / "fake.jpg"
    bogus.write_text("not an image", encoding="utf-8")
    with pytest.raises(ImageEncodeError):
        _encode_local_image(bogus, target_long_side=1536, jpeg_quality=85)


# --- _atomic_write_text (D11-N01) ------------------------------------------


def test_atomic_write_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "out.md"
    _atomic_write_text(p, "hello world")
    assert p.read_text(encoding="utf-8") == "hello world"


def test_atomic_write_creates_parent(tmp_path: Path) -> None:
    p = tmp_path / "nested" / "out.md"
    _atomic_write_text(p, "data")
    assert p.read_text(encoding="utf-8") == "data"


def test_atomic_write_mode_is_0o600_on_posix(tmp_path: Path) -> None:
    """Verifies the new os.open(..., 0o600) path is exercised (D11-N01)."""
    import os
    p = tmp_path / "out.md"
    _atomic_write_text(p, "new")
    mode = os.stat(p).st_mode & 0o777
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


# --- _safe_intermediates_dir (D11-N02 / D10-N03) --------------------------


def test_safe_intermediates_dir_accepts_normal_path() -> None:
    result = _safe_intermediates_dir("out/cache")
    assert isinstance(result, Path)


def test_safe_intermediates_dir_rejects_dotdot() -> None:
    import argparse
    with pytest.raises(argparse.ArgumentTypeError, match=r"\.\."):
        _safe_intermediates_dir("foo/../etc")


def test_cli_parse_rejects_traversal_intermediates_dir() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "in.pdf", "-o", "x.md",
            "--intermediates-dir", "../escape",
        ])


# --- _safe_cache_stem (D16-001 / D16-002 / D16-003) -----------------------


def test_safe_cache_stem_regular_passthrough() -> None:
    assert _safe_cache_stem("report") == "report"
    assert _safe_cache_stem("annual-2026") == "annual-2026"


def test_safe_cache_stem_strips_trailing_dot_or_space() -> None:
    assert _safe_cache_stem("trailing.") == "trailing"
    assert _safe_cache_stem("trailing. ") == "trailing"


def test_safe_cache_stem_empty_returns_underscore() -> None:
    assert _safe_cache_stem("") == "_"
    assert _safe_cache_stem("...") == "_"


@pytest.mark.skipif(
    __import__("sys").platform != "win32",
    reason="reserved-name suffix is Windows-only behaviour",
)
def test_safe_cache_stem_reserved_name_on_windows() -> None:
    assert _safe_cache_stem("CON") == "CON_"
    assert _safe_cache_stem("nul") == "nul_"
    assert _safe_cache_stem("COM1") == "COM1_"


def test_safe_cache_stem_no_suffix_off_windows() -> None:
    if __import__("sys").platform == "win32":
        pytest.skip("non-windows variant")
    assert _safe_cache_stem("CON") == "CON"

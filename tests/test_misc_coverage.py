"""Misc coverage: cache, pdf_renderer.read_page_text, runner._strip_think, CLI smoke."""

from __future__ import annotations

import pymupdf
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from pdf2md_agent import cli, pipeline
from pdf2md_agent.cache import (
    CacheLayout,
    is_page_complete,
)
from pdf2md_agent.cli_parser import _safe_intermediates_dir
from pdf2md_agent.filesystem_safety import safe_cache_stem as _safe_cache_stem
from pdf2md_agent.cache import atomic_write_text as _atomic_write_text
from pdf2md_agent.crew.multimodal_patch import ImageEncodeError, _encode_local_image
from pdf2md_agent.crew.runner import _strip_think
from pdf2md_agent.crew.types import PageRunContext, RenderedPage
from pdf2md_agent.pdf_renderer import read_page_text, render_pdf


# --- CacheLayout ----------------------------------------------------------


def test_cache_layout_for_pdf_creates_subdirs(tmp_path: Path) -> None:
    root = tmp_path / "out"
    layout = CacheLayout.for_pdf(root, tmp_path / "x.pdf")
    assert layout.root == root
    assert layout.pages_dir == root / "pages"
    assert layout.pages_dir.is_dir()
    assert layout.meta_path == root / "meta.json"


def test_cache_layout_artifacts_for_round_trip(tmp_path: Path) -> None:
    layout = CacheLayout.for_pdf(tmp_path / "out", tmp_path / "x.pdf")
    page = RenderedPage(ctx=PageRunContext(page_number=3, idx=3, total=10, page_started=0.0), width=100, height=100, image_path=tmp_path / "p3.png")
    a = layout.artifacts_for(3)
    assert a.page_number == 3
    assert a.page_png == layout.page_png_path(3)
    assert a.page_text == layout.page_text_path(3)
    assert a.format_markdown == layout.page_format_path(3)


def test_is_page_complete_true_when_format_exists(tmp_path: Path) -> None:
    layout = CacheLayout.for_pdf(tmp_path / "out", tmp_path / "x.pdf")
    layout.page_format_path(1).write_text("md", encoding="utf-8")
    assert is_page_complete(layout, 1) is True


def test_is_page_complete_false_when_format_missing(tmp_path: Path) -> None:
    layout = CacheLayout.for_pdf(tmp_path / "out", tmp_path / "x.pdf")
    assert is_page_complete(layout, 2) is False


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
    txt = read_page_text(pages[0].image_path.with_name(f"page_{pages[0].ctx.page_number:04d}_text.txt"))
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
    assert args.no_text_hint is False


def test_help_lists_argument_groups() -> None:
    parser = cli.build_parser()
    help_text = parser.format_help()
    for group in ("Pipeline", "Cache control", "Feature disable", "Retry & tuning"):
        assert group in help_text, f"missing help group: {group}"


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
    from pdf2md_agent import __version__

    assert __version__ in out


def test_cli_request_timeout_rejects_zero() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["in.pdf", "-o", "x.md", "--request-timeout", "0"])


def test_cli_request_timeout_rejects_negative() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["in.pdf", "-o", "x.md", "--request-timeout", "-1"])


def test_cli_max_retries_accepts_zero_for_unlimited() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["in.pdf", "-o", "x.md", "--max-retries", "0"])
    assert args.max_retries == 0


def test_cli_max_retries_rejects_negative() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["in.pdf", "-o", "x.md", "--max-retries", "-1"])


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
        parser.parse_args(
            [
                "in.pdf",
                "-o",
                "x.md",
                "--intermediates-dir",
                "../escape",
            ]
        )


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


# --- meta fingerprint drift ----------------------------------------------


def test_meta_fingerprint_drift_refuses_run(tmp_path: Path) -> None:
    from pdf2md_agent.cache import write_meta

    cache_root = tmp_path / "cache"
    layout = CacheLayout.for_pdf(cache_root, tmp_path / "input.pdf")
    write_meta(
        layout.meta_path,
        pdf=tmp_path / "other-file.pdf",
    )

    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

    captured: dict[str, str] = {"stdout": "", "stderr": ""}

    def _capture_print(msg: str, *args, **kwargs) -> None:
        stream = kwargs.get("file", None)
        if stream is None or stream is sys.stdout:
            captured["stdout"] += msg + "\n"
        else:
            captured["stderr"] += msg + "\n"

    with (
        patch.object(pipeline, "_render_pages", return_value=[]),
        patch.object(pipeline, "make_vision_llm", return_value=object()),
        patch.object(pipeline, "run_pipeline", return_value=[]),
        patch.object(pipeline, "stitch_pages", return_value=""),
        patch("builtins.print", side_effect=_capture_print),
    ):
        rc = cli.main(
            [
                str(pdf_path),
                "-o",
                str(tmp_path / "out.md"),
                "--intermediates-dir",
                str(cache_root),
            ]
        )
    assert rc == 1, "drift must refuse the run with exit code 1"
    assert "pdf changed" in captured["stderr"]
    assert "--no-cache-all" in captured["stderr"]


def test_meta_fingerprint_drift_bypassed_by_no_cache_all(
    tmp_path: Path,
) -> None:
    from pdf2md_agent.cache import read_meta, write_meta

    cache_root = tmp_path / "cache"
    layout = CacheLayout.for_pdf(cache_root, tmp_path / "input.pdf")
    write_meta(
        layout.meta_path,
        pdf=tmp_path / "different-cwd" / "input.pdf",
    )

    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

    with (
        patch.object(pipeline, "_render_pages", return_value=[]),
        patch.object(pipeline, "make_vision_llm", return_value=object()),
        patch.object(pipeline, "run_pipeline", return_value=[]),
        patch.object(pipeline, "stitch_pages", return_value=""),
    ):
        rc = cli.main(
            [
                str(pdf_path),
                "-o",
                str(tmp_path / "out.md"),
                "--intermediates-dir",
                str(cache_root),
                "--no-cache-all",
            ]
        )

    assert rc == 0, "drift must NOT refuse the run when --no-cache-all is set"
    post = read_meta(layout.meta_path)
    assert post is not None
    assert Path(post.pdf) == pdf_path.resolve()

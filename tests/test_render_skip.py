"""Tests for the render-side cache reuse helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from pdf2md_agent.cache import CacheLayout
from pdf2md_agent.pdf_renderer import render_pdf
from pdf2md_agent.render_skip import (
    maybe_skip_render,
    maybe_skip_resized,
)


def _make_pdf(path: Path, pages: int = 1) -> Path:
    doc = pymupdf.open()
    try:
        for _ in range(pages):
            doc.new_page()
        doc.save(str(path))
    finally:
        doc.close()
    return path


def _make_layout(tmp_path: Path, pdf: Path) -> CacheLayout:
    return CacheLayout.for_pdf(tmp_path / "cache", pdf)


# --- maybe_skip_render ------------------------------------------------------


def test_maybe_skip_render_returns_none_when_png_missing(tmp_path: Path) -> None:
    layout = _make_layout(tmp_path, tmp_path / "x.pdf")
    assert maybe_skip_render(layout, 1, dpi=144) is None


def test_maybe_skip_render_returns_png_when_sidecar_matches(tmp_path: Path) -> None:
    layout = _make_layout(tmp_path, tmp_path / "x.pdf")
    (layout.pages_dir).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), "white").save(
        layout.page_png_path(1), "PNG"
    )
    sidecar = layout.page_png_path(1).with_name(
        f"{layout.page_png_path(1).stem}.meta.json"
    )
    sidecar.write_text(json.dumps({"dpi": 144}), encoding="utf-8")

    assert maybe_skip_render(layout, 1, dpi=144) == layout.page_png_path(1)


def test_maybe_skip_render_returns_none_when_dpi_mismatches(tmp_path: Path) -> None:
    layout = _make_layout(tmp_path, tmp_path / "x.pdf")
    (layout.pages_dir).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), "white").save(
        layout.page_png_path(1), "PNG"
    )
    sidecar = layout.page_png_path(1).with_name(
        f"{layout.page_png_path(1).stem}.meta.json"
    )
    sidecar.write_text(json.dumps({"dpi": 200}), encoding="utf-8")

    assert maybe_skip_render(layout, 1, dpi=144) is None


# --- maybe_skip_resized -----------------------------------------------------


def test_maybe_skip_resized_returns_none_when_missing(tmp_path: Path) -> None:
    layout = _make_layout(tmp_path, tmp_path / "x.pdf")
    assert maybe_skip_resized(layout, 1, needed_long_side=1536) is None


def test_maybe_skip_resized_returns_jpeg_when_long_side_matches(
    tmp_path: Path,
) -> None:
    layout = _make_layout(tmp_path, tmp_path / "x.pdf")
    (layout.pages_dir).mkdir(parents=True, exist_ok=True)
    resized = layout.pages_dir / "page_0001_resized.jpg"
    Image.new("RGB", (10, 10), "red").save(resized, "JPEG")
    sidecar = resized.with_name(f"{resized.stem}.meta.json")
    sidecar.write_text(json.dumps({"long_side": 1024}), encoding="utf-8")

    assert maybe_skip_resized(layout, 1, needed_long_side=1024) == resized


def test_maybe_skip_resized_returns_none_on_long_side_mismatch(
    tmp_path: Path,
) -> None:
    layout = _make_layout(tmp_path, tmp_path / "x.pdf")
    (layout.pages_dir).mkdir(parents=True, exist_ok=True)
    resized = layout.pages_dir / "page_0001_resized.jpg"
    Image.new("RGB", (10, 10), "red").save(resized, "JPEG")
    sidecar = resized.with_name(f"{resized.stem}.meta.json")
    sidecar.write_text(json.dumps({"long_side": 1024}), encoding="utf-8")

    assert maybe_skip_resized(layout, 1, needed_long_side=512) is None


# --- render_skip + runner integration ---------------------------------------


def test_render_skip_honours_no_cache_render_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    pdf = _make_pdf(tmp_path / "tiny.pdf", pages=1)

    call_count = {"render": 0}

    real_render = render_pdf

    def _counting_render(*args: object, **kwargs: object) -> object:
        call_count["render"] += 1
        return real_render(*args, **kwargs)

    monkeypatch.setattr(
        "pdf2md_agent.cli.render_pdf", _counting_render
    )

    from pdf2md_agent import cli
    from pdf2md_agent.crew.runner import PageResult
    import argparse

    args = argparse.Namespace(
        pdf=pdf,
        output=tmp_path / "out.md",
        dpi=144,
        pages=None,
        no_intermediates=False,
        intermediates_dir=None,
        no_summary=False,
        no_text_hint=False,
        no_fallback_to_text=False,
        no_cache_render=False,
        no_cache_text=False,
        no_cache_resized=False,
        no_cache_extract=False,
        no_cache_format=False,
        no_cache_summary=False,
        no_cache_all=False,
        max_retries=None,
        retry_initial_delay=None,
        retry_max_delay=None,
        retry_jitter=None,
        image_long_side=None,
        image_quality=None,
        max_summary_chars=None,
        ctx_limit=None,
        stitch_mode="heuristic",
        request_timeout=None,
        model="m",
        persona_version="0123456789abcdef",
    )

    def _fake_run_pipeline(*_args: object, **_kwargs: object) -> list[PageResult]:
        return [PageResult(page_number=1, markdown="hello", summary="")]

    monkeypatch.setattr("pdf2md_agent.cli.run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(
        "pdf2md_agent.cli.stitch_pages", lambda *_a, **_k: "hello"
    )
    monkeypatch.setattr("pdf2md_agent.cli.make_vision_llm", lambda: object())
    monkeypatch.setattr(
        "pdf2md_agent.cli.write_meta", lambda *_a, **_k: None
    )

    rc = cli.cmd_convert(args)
    assert rc == 0
    assert call_count["render"] == 1, "first run renders the PNG"
    rc = cli.cmd_convert(args)
    assert rc == 0
    assert call_count["render"] == 1, "trust-cache run must not re-render"

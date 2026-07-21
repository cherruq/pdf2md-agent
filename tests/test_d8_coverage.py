"""D8 batch coverage: previously-untested internal helpers + strengthened assertion.

Targets (per findings.md):
- D8-003  ``cli._resolve_layout``
- D8-005  ``cli.cmd_convert`` (integration with mocked LLM + IO)
- D8-006  ``cli._run_pipeline`` (mocked render/run_pipeline/stitch + atomic write)
- D8-007  ``runner._resize_page_png`` (Pillow resize, long-side applied)
- D8-009  ``runner._record_text_layer_fallback`` (extract.txt + format.md writes)
- D8-010  ``runner._text_layer_fallback`` (markdown stub shape)
- D8-011  ``runner._output`` (attribute extraction)
- D8-014  ``multimodal_patch._encode_local_image`` (JPEG bytes shape)
- D8-015  ``multimodal_patch._to_data_url`` (data-URL wrapping + pass-through)
- D8-016  ``multimodal_patch._to_sentinel`` (VISION_IMAGE: prefix + b64 round-trip)

D8-002 ``cli._atomic_write_text`` (alias for ``cache.atomic_write_text``) is
already covered transitively by ``tests/test_misc_coverage.py::test_atomic_write_*``
and ``tests/test_cache.py::test_atomic_write_text_*``.

D8-012 is strengthened in-place in ``tests/test_runner.py`` (kept there so the
test sits next to the seam it guards).
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf
import pytest
from PIL import Image

from pdf2md_agent import cli
from pdf2md_agent.cache import CacheLayout
from pdf2md_agent.cli import _resolve_layout
from pdf2md_agent.crew.multimodal_patch import (
    _encode_local_image,
    _to_data_url,
    _to_sentinel,
)
from pdf2md_agent.crew.runner import (
    PageResult,
    _output,
    _record_text_layer_fallback,
    _resize_page_png,
    _text_layer_fallback,
)
from pdf2md_agent.pdf_renderer import PageImage


# --- helpers ---------------------------------------------------------------


def _make_onepage_pdf(path: Path) -> Path:
    """Synthesize a 1-page PDF on disk (no network, no fixture deps)."""
    doc = pymupdf.open()
    try:
        doc.new_page().insert_text((72, 72), "page 1")
        doc.save(str(path))
    finally:
        doc.close()
    return path


def _write_png(path: Path, *, width: int, height: int, color: str = "red") -> Path:
    """Write a small PNG of given dimensions to ``path`` (deterministic for tests)."""
    img = Image.new("RGB", (width, height), color)
    img.save(path, "PNG")
    return path


# ===========================================================================
# D8-003 — cli._resolve_layout
# ===========================================================================


def test_resolve_layout_keep_intermediates_default_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default root is the literal relative ``.pdf2md-agent-cache/<safe_stem>``."""
    monkeypatch.chdir(tmp_path)
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    layout, render_target = _resolve_layout(pdf, override=None, keep_intermediates=True)

    assert layout.root == Path(".pdf2md-agent-cache") / "report"
    assert render_target == layout.root / "pages"
    assert (tmp_path / layout.root).is_dir()
    assert (tmp_path / layout.pages_dir).is_dir()


def test_resolve_layout_keep_intermediates_with_override(tmp_path: Path) -> None:
    """``--intermediates-dir`` is honored verbatim (path-traversal pre-validated)."""
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    override = tmp_path / "custom" / "cache"

    layout, render_target = _resolve_layout(pdf, override=override, keep_intermediates=True)

    assert layout.root == override
    assert render_target == override / "pages"


def test_resolve_layout_no_intermediates_uses_tempdir(tmp_path: Path) -> None:
    """``--no-intermediates`` builds the layout under a tempdir; pages dir is pre-created."""
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    layout, render_target = _resolve_layout(pdf, override=None, keep_intermediates=False)

    # Layout root lives under the system tempdir — NOT under CWD.
    assert ".pdf2md-agent-cache" not in str(layout.root)
    assert "pdf2md_agent_" in layout.root.name  # mkdtemp prefix preserved
    assert layout.root.is_dir()
    assert render_target == layout.pages_dir
    assert layout.pages_dir.is_dir()


# ===========================================================================
# D8-005 — cli.cmd_convert (end-to-end with mocked IO)
# ===========================================================================


class _FakeOutput:
    def __init__(self, raw: str) -> None:
        self.raw = raw


class _FakeTask:
    def __init__(self, raw: str = "") -> None:
        self.output = _FakeOutput(raw)


def _build_minimal_args(tmp_path: Path, pdf: Path) -> argparse.Namespace:
    """Return the minimal ``argparse.Namespace`` cmd_convert consumes."""
    return argparse.Namespace(
        pdf=pdf,
        output=tmp_path / "out.md",
        dpi=144,
        pages=None,
        no_intermediates=False,
        reformat=False,
        intermediates_dir=None,
        resume=False,
        no_summary=False,
        no_text_hint=False,
        no_fallback_to_text=False,
        max_retries=None,
        retry_initial_delay=None,
        retry_backoff=None,
        retry_max_delay=None,
        retry_jitter=None,
        image_long_side=None,
        image_quality=None,
        max_summary_chars=None,
        ctx_limit=None,
        stitch_mode="heuristic",
    )


def test_cmd_convert_happy_path_writes_output_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """End-to-end: synthesize PDF, mock LLM + render + crew + stitch, verify output."""
    pdf = _make_onepage_pdf(tmp_path / "in.pdf")
    args = _build_minimal_args(tmp_path, pdf)

    page = PageImage(
        page_number=1, width=72, height=72, image_path=tmp_path / "page_0001.png",
    )

    with patch.object(cli, "render_pdf", return_value=[page]) as mock_render, \
         patch.object(cli, "make_vision_llm", return_value=object()), \
         patch.object(cli, "run_pipeline", return_value=[
             PageResult(page_number=1, markdown="# Title\n\n- item\n", summary="summary line"),
         ]) as mock_run, \
         patch.object(cli, "stitch_pages", return_value="# Title\n\n- item\n") as mock_stitch, \
         patch.object(cli, "write_meta") as mock_write_meta:
        rc = cli.cmd_convert(args)

    assert rc == 0
    out = args.output
    assert out.read_text(encoding="utf-8") == "# Title\n\n- item\n"
    # meta.json was emitted (intermediates kept by default)
    assert mock_write_meta.called
    # every stage got called exactly once
    assert mock_render.call_count == 1
    assert mock_run.call_count == 1
    assert mock_stitch.call_count == 1


def test_cmd_convert_no_intermediates_does_not_emit_meta(
    tmp_path: Path,
) -> None:
    """``--no-intermediates`` short-circuits meta.json emission (tempdir-only cache)."""
    pdf = _make_onepage_pdf(tmp_path / "in.pdf")
    args = _build_minimal_args(tmp_path, pdf)
    args.no_intermediates = True

    page = PageImage(
        page_number=1, width=72, height=72, image_path=tmp_path / "page_0001.png",
    )

    with patch.object(cli, "render_pdf", return_value=[page]), \
         patch.object(cli, "make_vision_llm", return_value=object()), \
         patch.object(cli, "run_pipeline", return_value=[
             PageResult(page_number=1, markdown="hi", summary=""),
         ]), \
         patch.object(cli, "stitch_pages", return_value="hi"), \
         patch.object(cli, "write_meta") as mock_write_meta:
        rc = cli.cmd_convert(args)

    assert rc == 0
    assert not mock_write_meta.called  # no meta.json when intermediates disabled
    assert args.output.read_text(encoding="utf-8") == "hi"


def test_cmd_convert_missing_pdf_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing PDF exits 1 with a stderr message — no crash, no partial output."""
    args = _build_minimal_args(tmp_path, tmp_path / "absent.pdf")
    rc = cli.cmd_convert(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "input PDF not found" in err


def test_cmd_convert_reformat_with_no_intermediates_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--reformat`` requires ``--intermediates``; the CLI rejects the combo early."""
    pdf = _make_onepage_pdf(tmp_path / "in.pdf")
    args = _build_minimal_args(tmp_path, pdf)
    args.reformat = True
    args.no_intermediates = True

    rc = cli.cmd_convert(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "--reformat requires --intermediates" in err


# ===========================================================================
# D8-006 — cli._run_pipeline (atomic write on the output path)
# ===========================================================================


def test_run_pipeline_calls_atomic_write_text_with_stitched_markdown(
    tmp_path: Path,
) -> None:
    """``_run_pipeline`` writes the stitched output via the atomic-write helper."""
    pdf = _make_onepage_pdf(tmp_path / "in.pdf")
    out_path = tmp_path / "out.md"
    args = _build_minimal_args(tmp_path, pdf)
    args.output = out_path

    page = PageImage(
        page_number=1, width=72, height=72, image_path=tmp_path / "page_0001.png",
    )
    layout = CacheLayout.for_pdf(tmp_path / ".pdf2md-agent-cache" / "in", pdf)

    with patch.object(cli, "render_pdf", return_value=[page]) as mock_render, \
         patch.object(cli, "make_vision_llm", return_value=object()), \
         patch.object(cli, "run_pipeline", return_value=[
             PageResult(page_number=1, markdown="stitched body", summary=""),
         ]), \
         patch.object(cli, "stitch_pages", return_value="stitched body") as mock_stitch, \
         patch.object(cli, "_atomic_write_text") as mock_atomic, \
         patch.object(cli, "write_meta"):
        rc = cli._run_pipeline(
            args=args,
            layout=layout,
            render_target=layout.pages_dir,
            resolved_pages=None,
            keep_intermediates=True,
            with_summary=False,
            retry_config=__import__(
                "pdf2md_agent.llm_retry", fromlist=["RetryConfig"]
            ).RetryConfig(),
            fallback_to_text=True,
            started=__import__("time").monotonic(),
        )

    assert rc == 0
    assert mock_atomic.call_count == 1
    # The atomic write must have been called with the stitched markdown and the
    # user-supplied output path — never a Path.write_text() shortcut.
    written_path, written_text = mock_atomic.call_args.args
    assert written_path == out_path
    assert written_text == "stitched body"
    assert mock_render.called
    assert mock_stitch.called


# ===========================================================================
# D8-007 — runner._resize_page_png
# ===========================================================================


def test_resize_page_png_applies_target_long_side(tmp_path: Path) -> None:
    """Output JPEG's longest side equals ``target_long_side`` (LANCZOS thumbnail)."""
    src = _write_png(tmp_path / "big.png", width=2000, height=1000)
    dst = tmp_path / "small.jpg"

    _resize_page_png(src, dst, target_long_side=400, jpeg_quality=80)

    assert dst.is_file()
    with Image.open(dst) as out:
        w, h = out.size
    assert max(w, h) == 400
    assert min(w, h) == 200  # aspect ratio preserved by thumbnail()


def test_resize_page_png_preserves_aspect_ratio(tmp_path: Path) -> None:
    """Square source resized to a non-square target stays square only when target matches."""
    src = _write_png(tmp_path / "square.png", width=800, height=800)
    dst = tmp_path / "shrunk.jpg"

    _resize_page_png(src, dst, target_long_side=200, jpeg_quality=70)

    with Image.open(dst) as out:
        w, h = out.size
    assert w == 200 and h == 200  # thumbnail keeps aspect; square stays square


def test_resize_page_png_no_op_when_target_exceeds_source(tmp_path: Path) -> None:
    """A target larger than the source is a no-op (thumbnail never upscales)."""
    src = _write_png(tmp_path / "tiny.png", width=100, height=50)
    dst = tmp_path / "still_tiny.jpg"

    _resize_page_png(src, dst, target_long_side=500, jpeg_quality=70)

    with Image.open(dst) as out:
        w, h = out.size
    assert (w, h) == (100, 50)


# ===========================================================================
# D8-009 — runner._record_text_layer_fallback
# ===========================================================================


def _make_page_artifacts(tmp_path: Path, page_number: int, text: str):
    """Build a ``PageArtifacts`` with a populated ``page_text.txt`` for fallback testing."""
    layout = CacheLayout.for_pdf(tmp_path / "cache", tmp_path / "x.pdf")
    page = PageImage(
        page_number=page_number, width=100, height=100,
        image_path=tmp_path / "page.png",
    )
    artifacts = layout.artifacts_for(page)
    artifacts.page_text.write_text(text, encoding="utf-8")
    return artifacts


def test_record_text_layer_fallback_writes_extract_and_format(tmp_path: Path) -> None:
    """Both cache files written; extract.txt is empty; format.md has the stub marker."""
    artifacts = _make_page_artifacts(tmp_path, 1, "raw pdf text\n")

    result = _record_text_layer_fallback(
        idx=1, total=1, page_number=1, page_started=0.0,
        artifacts=artifacts, summary="prior summary",
        completion_label="fallback",
    )

    assert isinstance(result, PageResult)
    assert result.page_number == 1
    assert result.summary == "prior summary"
    assert "vision model unavailable" in result.markdown
    # extract.txt is the canonical "this page was attempted but vision failed" marker.
    assert artifacts.extract_text.read_text(encoding="utf-8") == ""
    # format.md carries the recoverable fallback content.
    assert "raw pdf text" in artifacts.format_markdown.read_text(encoding="utf-8")


def test_record_text_layer_fallback_returns_consistent_page_result(tmp_path: Path) -> None:
    """``PageResult`` shape preserves page_number and the (unchanged) summary."""
    artifacts = _make_page_artifacts(tmp_path, 7, "page 7 text\n")
    result = _record_text_layer_fallback(
        idx=7, total=10, page_number=7, page_started=0.0,
        artifacts=artifacts, summary="carry-over summary 中文",
        completion_label="validation-fallback",
    )
    assert result.page_number == 7
    assert result.summary == "carry-over summary 中文"


# ===========================================================================
# D8-010 — runner._text_layer_fallback
# ===========================================================================


def test_text_layer_fallback_with_text_contains_stub_marker(tmp_path: Path) -> None:
    """Markdown stub shape: leading italic marker + fenced ``text`` block."""
    artifacts = _make_page_artifacts(tmp_path, 1, "recovered line 1\nrecovered line 2\n")

    md = _text_layer_fallback(artifacts)

    assert "*(vision model unavailable" in md
    assert "PDF text layer" in md
    assert md.count("```") == 2  # opens + closes the fenced block
    assert "recovered line 1" in md
    assert "recovered line 2" in md


def test_text_layer_fallback_empty_text_returns_no_content_marker(tmp_path: Path) -> None:
    """Empty text layer emits a distinct 'no content recovered' message — no fenced block."""
    artifacts = _make_page_artifacts(tmp_path, 1, "")

    md = _text_layer_fallback(artifacts)

    assert "no content recovered" in md
    assert "vision model unavailable" in md
    assert "```" not in md  # no fenced block when there is no text to put in one


def test_text_layer_fallback_strips_text_layer_whitespace(tmp_path: Path) -> None:
    """Whitespace-only text layer is treated the same as empty (no fenced block)."""
    artifacts = _make_page_artifacts(tmp_path, 1, "   \n\n   \n")
    md = _text_layer_fallback(artifacts)
    assert "no content recovered" in md


# ===========================================================================
# D8-011 — runner._output
# ===========================================================================


def test_output_extracts_raw_from_nested_object() -> None:
    """Standard CrewAI ``TaskOutput`` shape: ``task.output.raw`` -> the raw string."""
    out_obj = MagicMock()
    out_obj.raw = "page text 1\n"
    task = MagicMock(spec=["output"])
    task.output = out_obj

    assert _output(task) == "page text 1"


def test_output_returns_empty_string_when_output_attr_missing() -> None:
    """Tasks that never produced output (``task.output is None``) -> empty string."""
    task = MagicMock(spec=["output"])
    task.output = None
    assert _output(task) == ""


def test_output_strips_think_blocks_from_raw() -> None:
    """``_output`` runs ``_strip_think`` defensively on the raw payload."""
    out_obj = MagicMock()
    out_obj.raw = "before<think>scratch</think>after"
    task = MagicMock(spec=["output"])
    task.output = out_obj
    assert _output(task) == "beforeafter"


def test_output_falls_back_to_str_when_raw_is_not_string() -> None:
    """Legacy/shim output objects without a ``raw`` string are stringified."""
    out_obj = MagicMock(spec=["raw"])
    out_obj.raw = 12345
    task = MagicMock(spec=["output"])
    task.output = out_obj
    result = _output(task)
    assert isinstance(result, str)
    assert result


# ===========================================================================
# D8-014 / D8-015 / D8-016 — multimodal_patch encoding helpers
# ===========================================================================


# --- D8-014 ---


def test_encode_local_image_produces_valid_jpeg_bytes(tmp_path: Path) -> None:
    """Output starts with the JPEG SOI marker (``FFD8FF``) and decodes back as JPEG."""
    src = _write_png(tmp_path / "in.png", width=64, height=64)
    encoded = _encode_local_image(src, target_long_side=64, jpeg_quality=85)

    assert encoded[:3] == b"\xff\xd8\xff", "expected JPEG SOI magic bytes"
    # Round-trip through Pillow: should open as RGB JPEG.
    with Image.open(io.BytesIO(encoded)) as img:
        assert img.format == "JPEG"
        assert img.mode == "RGB"


def test_encode_local_image_respects_target_long_side(tmp_path: Path) -> None:
    """Resized JPEG's longest side is bounded by ``target_long_side``."""
    src = _write_png(tmp_path / "big.png", width=1200, height=300)
    encoded = _encode_local_image(src, target_long_side=150, jpeg_quality=70)

    with Image.open(io.BytesIO(encoded)) as img:
        w, h = img.size
    assert max(w, h) == 150
    assert min(w, h) <= 150


# --- D8-015 ---


def test_to_data_url_wraps_local_image_as_data_url(tmp_path: Path) -> None:
    """A real local image becomes ``data:image/jpeg;base64,<b64>``."""
    src = _write_png(tmp_path / "page.png", width=48, height=48)

    url = _to_data_url(str(src), target_long_side=48, jpeg_quality=80)

    assert url.startswith("data:image/jpeg;base64,")
    # The b64 suffix is valid base64 and decodes back to JPEG bytes.
    b64 = url.split(",", 1)[1]
    decoded = base64.b64decode(b64, validate=True)
    assert decoded[:3] == b"\xff\xd8\xff"


@pytest.mark.parametrize(
    "remote",
    [
        "http://example.test/img.png",
        "https://example.test/img.png",
        "data:image/png;base64,AAAAAAAA",
    ],
)
def test_to_data_url_passes_remote_values_through_unchanged(remote: str) -> None:
    """HTTP(S) and ``data:`` inputs are returned verbatim — never re-encoded."""
    assert _to_data_url(remote, target_long_side=1536, jpeg_quality=85) == remote


def test_to_data_url_returns_unknown_path_unchanged(tmp_path: Path) -> None:
    """A path that doesn't resolve to a file is returned as-is (no exception)."""
    missing = tmp_path / "no-such.png"
    assert _to_data_url(str(missing)) == str(missing)


def test_to_data_url_empty_string_returns_empty(tmp_path: Path) -> None:
    """Empty input -> empty output (defensive guard)."""
    assert _to_data_url("") == ""


# --- D8-016 ---


def test_to_sentinel_format_with_action(tmp_path: Path) -> None:
    """Sentinel shape: ``<action>\\nVISION_IMAGE:<media_type>:<b64>`` for local images."""
    missing = tmp_path / "no-such.png"
    sentinel = _to_sentinel(
        str(missing),
        action="describe this page",
        target_long_side=64,
        jpeg_quality=70,
    )
    assert sentinel == "describe this page"


def test_to_sentinel_format_without_action(tmp_path: Path) -> None:
    """Sentinel shape for a local image: ``VISION_IMAGE:image/jpeg:<b64>``."""
    src = _write_png(tmp_path / "page.png", width=64, height=64)

    sentinel = _to_sentinel(str(src), action=None, target_long_side=64, jpeg_quality=80)

    assert sentinel.startswith("VISION_IMAGE:image/jpeg:")
    assert sentinel.split(":")[1] == "image/jpeg"
    b64 = sentinel.rsplit(":", 1)[-1]
    decoded = base64.b64decode(b64, validate=True)
    assert decoded[:3] == b"\xff\xd8\xff"


def test_to_sentinel_b64_decodes_round_trip(tmp_path: Path) -> None:
    """The b64 suffix decodes to a valid JPEG and the bytes re-open in Pillow."""
    src = _write_png(tmp_path / "x.png", width=32, height=32, color="blue")
    sentinel = _to_sentinel(str(src), action=None, target_long_side=32, jpeg_quality=70)
    b64 = sentinel.rsplit(":", 1)[-1]
    decoded = base64.b64decode(b64)
    with Image.open(io.BytesIO(decoded)) as img:
        assert img.format == "JPEG"
        assert img.size == (32, 32)


def test_to_sentinel_returns_action_or_fallback_for_remote_url() -> None:
    """URL inputs (no inline) yield the action text (or a clear fallback message)."""
    # With action: action text is surfaced.
    assert (
        _to_sentinel("https://example.test/x.png", action="caption text")
        == "caption text"
    )
    # Without action: a clear '(could not inline image at <url>)' marker.
    sentinel = _to_sentinel("https://example.test/x.png", action=None)
    assert "could not inline image" in sentinel
    assert "https://example.test/x.png" in sentinel
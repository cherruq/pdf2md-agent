"""Tests for the --reformat feature.

Skips the extractor on any page whose extract.txt is already on disk and
runs the layout-aware formatter (+ summarizer) instead. Pages whose
extract.txt is missing fall through to the standard extract→format pipeline.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from convertpdf.crew.agents import (
    FORMATTER_PERSONA_STRICT,
    FORMATTER_PERSONA_REFORMAT,
    make_formatter,
)
from convertpdf.cache import CacheLayout, has_cached_extract
from convertpdf.cli import build_parser
from convertpdf.crew.runner import _run_format_summarize_only, run_pipeline
from convertpdf.crew.tasks import (
    make_format_task,
    make_format_task_from_extract_file,
)
from convertpdf.llm_retry import RetryConfig
from openai import APITimeoutError
from pydantic import BaseModel, ValidationError


def test_formatter_persona_strict_unchanged():
    """The existing strict persona must keep its byte-for-byte text."""
    assert "Never drop, translate, or rewrite content" in FORMATTER_PERSONA_STRICT
    assert "Layout-Aware" not in FORMATTER_PERSONA_STRICT


def test_formatter_persona_reformat_drops_layout_artifacts():
    assert "Layout-Aware" in FORMATTER_PERSONA_REFORMAT
    # The reformat persona must instruct dropping all three artifact types
    # the user asked about (headers, footers, page numbers).
    assert "page headers" in FORMATTER_PERSONA_REFORMAT.lower()
    assert "footers" in FORMATTER_PERSONA_REFORMAT.lower()
    assert "page numbers" in FORMATTER_PERSONA_REFORMAT.lower()
    # And must preserve body content verbatim.
    assert "preserve" in FORMATTER_PERSONA_REFORMAT.lower()
    # Must NOT mention the original (un-split) persona name.
    assert "OCR noise" not in FORMATTER_PERSONA_REFORMAT


def test_make_formatter_reformat_uses_layout_aware_role():
    with patch("convertpdf.crew.agents.Agent") as MockAgent:
        make_formatter(MagicMock(), reformat=True)
    # role carries "Layout-Aware"; backstory is the extracted \n\n text
    assert MockAgent.call_args.kwargs["role"] == "Markdown Formatter (Layout-Aware)"
    assert "Layout-Aware" not in MockAgent.call_args.kwargs["backstory"]
    # verify the reformat backstory was used (contains page-artifact instruction)
    assert "page headers" in MockAgent.call_args.kwargs["backstory"].lower()


def test_make_formatter_default_unchanged():
    """reformat=False default must match today's agent exactly."""
    with patch("convertpdf.crew.agents.Agent") as MockAgent:
        make_formatter(MagicMock())
    assert MockAgent.call_args.kwargs["role"] == "Markdown Formatter"
    assert "Layout-Aware" not in MockAgent.call_args.kwargs["backstory"]

    with patch("convertpdf.crew.agents.Agent") as MockAgent:
        make_formatter(MagicMock(), reformat=False)
    assert MockAgent.call_args.kwargs["role"] == "Markdown Formatter"
    assert "Layout-Aware" not in MockAgent.call_args.kwargs["backstory"]


def test_make_format_task_reformat_appends_strip_instruction():
    """make_format_task(reformat=True) must append strip-headers/footers/numbers."""
    extract_t = MagicMock()
    with patch("convertpdf.crew.tasks.Task") as MockTask:
        MockTask.return_value = MagicMock(description="", context=[])
        make_format_task(MagicMock(), extract_t, reformat=True)
        desc = MockTask.call_args.kwargs["description"]
        assert "drop headers/footers/page numbers" in desc
        assert MockTask.call_args.kwargs["context"] == [extract_t]


def test_make_format_task_default_description_unchanged():
    """Default (reformat=False) description must match today's byte-for-byte."""
    extract_t = MagicMock()
    with patch("convertpdf.crew.tasks.Task") as MockTask:
        MockTask.return_value = MagicMock(description="", context=[])
        make_format_task(MagicMock(), extract_t)
        desc = MockTask.call_args.kwargs["description"]
        assert "drop headers/footers/page numbers" not in desc
        assert MockTask.call_args.kwargs["context"] == [extract_t]


def test_make_format_task_from_extract_file_inlines_text(tmp_path: Path):
    extract_path = tmp_path / "page_0001_extract.txt"
    extract_path.write_text("# Heading\n\nPage 7 of 100\n\nBody.\n", encoding="utf-8")
    with patch("convertpdf.crew.tasks.Task") as MockTask:
        MockTask.return_value = MagicMock(description="", context=[])
        make_format_task_from_extract_file(MagicMock(), extract_path)
        desc = MockTask.call_args.kwargs["description"]
        assert "Page 7 of 100" in desc
        assert "# Heading" in desc
        assert "```\n# Heading\n\nPage 7 of 100\n\nBody.\n\n```" in desc
        assert MockTask.call_args.kwargs["context"] == []


def test_make_format_task_from_extract_file_missing_file_raises(tmp_path: Path):
    """Missing extract.txt must propagate FileNotFoundError (callers should
    gate on has_cached_extract() before calling this)."""
    with pytest.raises(FileNotFoundError):
        make_format_task_from_extract_file(MagicMock(), tmp_path / "nope.txt")


def test_has_cached_extract_true_when_extract_exists(tmp_path: Path):
    root = tmp_path / "cache"
    layout = CacheLayout.for_pdf(root, tmp_path / "fake.pdf")
    layout.page_extract_path(1).write_text("body", encoding="utf-8")
    assert has_cached_extract(layout, 1) is True


def test_has_cached_extract_false_when_extract_missing(tmp_path: Path):
    root = tmp_path / "cache"
    layout = CacheLayout.for_pdf(root, tmp_path / "fake.pdf")
    assert has_cached_extract(layout, 1) is False


def test_has_cached_extract_independent_of_format(tmp_path: Path):
    """has_cached_extract looks at extract.txt only, never at format.md."""
    root = tmp_path / "cache"
    layout = CacheLayout.for_pdf(root, tmp_path / "fake.pdf")
    # Format exists, extract doesn't -> False (not True)
    layout.page_format_path(1).write_text("md", encoding="utf-8")
    assert has_cached_extract(layout, 1) is False
    # Extract exists, format doesn't -> True (not False)
    layout.page_extract_path(1).write_text("body", encoding="utf-8")
    layout.page_format_path(1).unlink()
    assert has_cached_extract(layout, 1) is True


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _make_artifacts(tmp_path: Path, page_number: int = 1):
    from convertpdf.cache import CacheLayout
    root = tmp_path / "cache"
    layout = CacheLayout.for_pdf(root, tmp_path / "fake.pdf")
    artifacts = layout.artifacts_for(
        type("P", (), {"page_number": page_number})()  # duck-typed PageImage
    )
    return layout, artifacts


def test_run_format_summarize_only_writes_format_md_from_extract(
    tmp_path: Path,
):
    """The helper must overwrite format.md with LLM-formatted content and
    leave extract.txt untouched."""
    layout, artifacts = _make_artifacts(tmp_path)
    _write_text(
        artifacts.extract_text,
        "# Body Heading\n\nPage 1 of 10\n\nReal body paragraph.\n",
    )
    # Snapshot the on-disk bytes so we can prove exact preservation after the run.
    before_bytes = artifacts.extract_text.read_bytes()

    llm = MagicMock()
    retry = RetryConfig(max_attempts=1, initial_delay=0.0, backoff=1.0, max_delay=0.0, jitter=0.0)

    with patch("convertpdf.crew.runner.make_formatter") as mk_fmt, \
         patch("convertpdf.crew.runner.make_summarizer") as mk_sum, \
         patch("convertpdf.crew.runner.make_format_task_from_extract_file") as mk_fmt_task, \
         patch("convertpdf.crew.runner.make_summarize_task") as mk_sum_task, \
         patch("convertpdf.crew.runner.make_extractor") as mk_extractor, \
         patch("convertpdf.crew.runner.Crew") as crew_cls, \
         patch("convertpdf.crew.runner._output") as output_fn:
        formatter = MagicMock()
        summarizer = MagicMock()
        mk_fmt.return_value = formatter
        mk_sum.return_value = summarizer
        format_t = MagicMock()
        summarize_t = MagicMock()
        mk_fmt_task.return_value = format_t
        mk_sum_task.return_value = summarize_t
        # Simulate the LLM "dropping" the page number from the extract.
        output_fn.side_effect = lambda task: (
            "# Body Heading\n\nReal body paragraph.\n" if task is format_t else "summary text"
        )
        crew = MagicMock()
        crew.kickoff.return_value = None
        crew_cls.return_value = crew

        fmt_out, sum_out, did_fb = _run_format_summarize_only(
            page_number=1,
            artifacts=artifacts,
            summary_in="prev summary",
            summary_path=layout.summary_path,
            with_summary=True,
            llm=llm,
            retry_config=retry,
            fallback_to_text=False,
            max_summary_chars=800,
        )

    assert "Page 1 of 10" not in fmt_out
    assert "Body Heading" in fmt_out
    assert "Real body paragraph" in fmt_out
    assert artifacts.format_markdown.read_text(encoding="utf-8") == fmt_out
    # extract.txt bytes must be byte-identical to the snapshot — the helper
    # must never overwrite the extractor's output.
    assert artifacts.extract_text.read_bytes() == before_bytes
    # mk_formatter was called with reformat=True
    mk_fmt.assert_called_once_with(llm, reformat=True)
    # No extract step ran — patch make_extractor and prove it was never called.
    assert mk_extractor.call_count == 0
    # summary.json was written
    assert layout.summary_path.exists()
    assert did_fb is False


def test_run_pipeline_reformat_skips_extractor_and_rewrites_format(
    tmp_path: Path,
):
    """With reformat=True and a cached extract.txt, run_pipeline must NOT
    call the extractor and must overwrite format.md."""
    from convertpdf.cache import CacheLayout, write_meta

    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake\n")
    root = tmp_path / "cache"
    layout = CacheLayout.for_pdf(root, pdf)
    write_meta(layout.meta_path, pdf=pdf, dpi=144, with_summary=True)

    # Seed a cached extract + a stale format that we expect to be overwritten.
    layout.page_extract_path(1).write_text(
        "# Real Body\n\nFooter: Page 1 of 999\n", encoding="utf-8"
    )
    layout.page_format_path(1).write_text("STALE OLD OUTPUT", encoding="utf-8")

    fake_page = type(
        "P", (), {"page_number": 1, "image_path": tmp_path / "page_0001.png",
                  "width": 100, "height": 100}
    )()
    (tmp_path / "page_0001.png").write_bytes(b"PNGDATA")
    pages = [fake_page]

    llm = MagicMock()
    retry = RetryConfig(max_attempts=1, initial_delay=0.0, backoff=1.0, max_delay=0.0, jitter=0.0)

    with patch("convertpdf.crew.runner.render_pdf", return_value=pages), \
         patch("convertpdf.crew.runner.plan_for_image") as plan, \
         patch("convertpdf.crew.runner.estimate_text_tokens", return_value=10), \
         patch("convertpdf.crew.runner.estimate_image_tokens", return_value=10), \
         patch("convertpdf.crew.runner.make_vision_llm", return_value=llm), \
         patch("convertpdf.crew.runner._run_format_summarize_only") as helper, \
         patch("convertpdf.crew.runner.call_with_retry"), \
         patch("convertpdf.crew.runner.make_extractor") as extractor_factory, \
         patch("convertpdf.crew.runner.make_formatter"), \
         patch("convertpdf.crew.runner.make_summarize_task"):
        plan.return_value = type("D", (), {"total": 100, "fits": True, "needed_long_side": 1536, "reason": "test"})()
        helper.return_value = ("# Real Body\n\n(no footer)\n", "sum", False)

        results = run_pipeline(
            pages=pages, layout=layout, with_summary=True, resume=False,
            text_hint=False, llm=llm, retry_config=retry,
            fallback_to_text=False, ctx_limit=2013,
            image_long_side=1536, image_min_long_side=64,
            image_jpeg_quality=85, max_summary_chars=800,
            token_budget_safety=0.8, reformat=True,
        )

    assert len(results) == 1
    assert extractor_factory.call_count == 0  # SKIPPED
    assert helper.call_count == 1
    assert "Footer: Page 1 of 999" not in results[0].markdown
    assert "Page 1 of 999" in layout.page_extract_path(1).read_text(encoding="utf-8")
    assert layout.page_format_path(1).read_text(encoding="utf-8") == "# Real Body\n\n(no footer)\n"


def test_run_pipeline_reformat_falls_back_when_extract_missing(
    tmp_path: Path,
):
    """If extract.txt is missing, --reformat falls through to the full pipeline."""
    from convertpdf.cache import CacheLayout, write_meta

    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake\n")
    root = tmp_path / "cache"
    layout = CacheLayout.for_pdf(root, pdf)
    write_meta(layout.meta_path, pdf=pdf, dpi=144, with_summary=True)

    fake_page = type(
        "P", (), {"page_number": 1, "image_path": tmp_path / "page_0001.png",
                  "width": 100, "height": 100}
    )()
    (tmp_path / "page_0001.png").write_bytes(b"PNG")
    pages = [fake_page]

    llm = MagicMock()
    retry = RetryConfig(max_attempts=1, initial_delay=0.0, backoff=1.0, max_delay=0.0, jitter=0.0)

    with patch("convertpdf.crew.runner.render_pdf", return_value=pages), \
         patch("convertpdf.crew.runner.plan_for_image") as plan, \
         patch("convertpdf.crew.runner.estimate_text_tokens", return_value=10), \
         patch("convertpdf.crew.runner.estimate_image_tokens", return_value=10), \
         patch("convertpdf.crew.runner.make_vision_llm", return_value=llm), \
         patch("convertpdf.crew.runner._run_format_summarize_only") as helper, \
         patch("convertpdf.crew.runner.call_with_retry"), \
         patch("convertpdf.crew.runner.make_extractor"), \
         patch("convertpdf.crew.runner.make_formatter"), \
         patch("convertpdf.crew.runner.make_format_task") as format_task_factory, \
         patch("convertpdf.crew.runner.make_extract_task") as extract_task_factory, \
         patch("convertpdf.crew.runner.make_summarize_task"):
        plan.return_value = type("D", (), {"total": 100, "fits": True, "needed_long_side": 1536, "reason": "test"})()
        with patch("convertpdf.crew.runner.Crew") as crew_cls, \
             patch("convertpdf.crew.runner._output", return_value="STUB MD"):
            crew = MagicMock()
            crew.kickoff.return_value = None
            crew_cls.return_value = crew
            run_pipeline(
                pages=pages, layout=layout, with_summary=False, resume=False,
                text_hint=False, llm=llm, retry_config=retry,
                fallback_to_text=False, ctx_limit=2013,
                image_long_side=1536, image_min_long_side=64,
                image_jpeg_quality=85, max_summary_chars=800,
                token_budget_safety=0.8, reformat=True,
            )

    assert helper.call_count == 0
    assert extract_task_factory.call_count >= 1
    assert format_task_factory.call_count >= 1


def test_run_pipeline_reformat_does_not_call_extract_when_resume_triggers(
    tmp_path: Path,
):
    """--resume --reformat: when both extract.txt AND format.md exist, the
    resume short-circuit fires and reformat is not even consulted."""
    from convertpdf.cache import CacheLayout, write_meta

    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake\n")
    root = tmp_path / "cache"
    layout = CacheLayout.for_pdf(root, pdf)
    write_meta(layout.meta_path, pdf=pdf, dpi=144, with_summary=True)
    layout.page_extract_path(1).write_text("cached extract", encoding="utf-8")
    layout.page_format_path(1).write_text("CACHED MD", encoding="utf-8")

    fake_page = type(
        "P", (), {"page_number": 1, "image_path": tmp_path / "page_0001.png",
                  "width": 100, "height": 100}
    )()
    (tmp_path / "page_0001.png").write_bytes(b"PNG")
    pages = [fake_page]

    llm = MagicMock()
    retry = RetryConfig(max_attempts=1, initial_delay=0.0, backoff=1.0, max_delay=0.0, jitter=0.0)

    with patch("convertpdf.crew.runner.render_pdf", return_value=pages), \
         patch("convertpdf.crew.runner._run_format_summarize_only") as helper, \
         patch("convertpdf.crew.runner.make_extractor") as extractor_factory:
        results = run_pipeline(
            pages=pages, layout=layout, with_summary=False, resume=True,
            text_hint=False, llm=llm, retry_config=retry,
            fallback_to_text=False, ctx_limit=2013,
            image_long_side=1536, image_min_long_side=64,
            image_jpeg_quality=85, max_summary_chars=800,
            token_budget_safety=0.8, reformat=True,
        )

    assert len(results) == 1
    assert helper.call_count == 0
    assert extractor_factory.call_count == 0
    assert layout.page_format_path(1).read_text(encoding="utf-8") == "CACHED MD"


def test_cli_parser_accepts_reformat_flag():
    args = build_parser().parse_args(["in.pdf", "-o", "out.md", "--reformat"])
    assert args.reformat is True


def test_cli_parser_reformat_defaults_to_false():
    args = build_parser().parse_args(["in.pdf", "-o", "out.md"])
    assert args.reformat is False


def test_cli_rejects_reformat_with_no_intermediates(tmp_path: Path):
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake\n")
    out = tmp_path / "out.md"
    proc = subprocess.run(
        [sys.executable, "-m", "convertpdf", str(pdf),
         "-o", str(out), "--reformat", "--no-intermediates"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert "--reformat requires --intermediates" in proc.stderr


def test_run_format_summarize_only_falls_back_on_validation_error(
    tmp_path: Path,
):
    """ValidationError from crew.kickoff → extract.txt passes through as
    format.md, summary.json NOT overwritten, did_fallback=True.

    Locks down the contract that a malformed LLM response (after retry
    exhaustion) does NOT silently corrupt the cache: the cached extract
    survives, format.md gets a passthrough copy, and the running summary
    is preserved (summarizer skipped — it depends on the formatted output).
    """

    class _StubModel(BaseModel):
        x: int

    def _raise_validation_error() -> ValidationError:
        try:
            _StubModel.model_validate({"x": "not_int"})  # type: ignore[arg-type]
        except ValidationError as exc:
            return exc
        raise AssertionError("expected ValidationError")

    layout, artifacts = _make_artifacts(tmp_path)
    cached_extract = "# Body\n\nPage 1 of 10\n\nReal body.\n"
    artifacts.extract_text.write_text(cached_extract, encoding="utf-8")
    # Pre-existing summary.json — must NOT be overwritten by the fallback path.
    layout.summary_path.write_text(
        '{"summary": "old summary from prior page"}', encoding="utf-8"
    )

    llm = MagicMock()
    retry = RetryConfig(
        max_attempts=1, initial_delay=0.0, backoff=1.0, max_delay=0.0, jitter=0.0
    )

    with patch("convertpdf.crew.runner.make_formatter"), \
         patch("convertpdf.crew.runner.make_summarizer"), \
         patch("convertpdf.crew.runner.make_format_task_from_extract_file") as mk_fmt_task, \
         patch("convertpdf.crew.runner.make_summarize_task") as mk_sum_task, \
         patch("convertpdf.crew.runner.Crew") as crew_cls:
        format_t = MagicMock()
        summarize_t = MagicMock()
        mk_fmt_task.return_value = format_t
        mk_sum_task.return_value = summarize_t
        crew = MagicMock()
        crew.kickoff.side_effect = _raise_validation_error()
        crew_cls.return_value = crew

        fmt_out, sum_out, did_fb = _run_format_summarize_only(
            page_number=1,
            artifacts=artifacts,
            summary_in="prev summary",
            summary_path=layout.summary_path,
            with_summary=True,
            llm=llm,
            retry_config=retry,
            fallback_to_text=True,
            max_summary_chars=800,
        )

    assert did_fb is True
    assert fmt_out == cached_extract
    assert sum_out == "prev summary"
    assert artifacts.format_markdown.read_text(encoding="utf-8") == cached_extract
    assert artifacts.extract_text.read_text(encoding="utf-8") == cached_extract
    assert (
        layout.summary_path.read_text(encoding="utf-8")
        == '{"summary": "old summary from prior page"}'
    )


def test_run_format_summarize_only_falls_back_on_transient_error(
    tmp_path: Path,
):
    """Transient LLM error (e.g. APITimeoutError) → same fallback contract
    as ValidationError. Non-transient BaseExceptions would re-raise, which
    is covered implicitly by the happy-path test using fallback_to_text=False
    + a successful crew.kickoff."""
    layout, artifacts = _make_artifacts(tmp_path)
    cached_extract = "# Body\n\nFooter: 99\n\nReal body.\n"
    artifacts.extract_text.write_text(cached_extract, encoding="utf-8")
    layout.summary_path.write_text(
        '{"summary": "old summary"}', encoding="utf-8"
    )

    llm = MagicMock()
    retry = RetryConfig(
        max_attempts=1, initial_delay=0.0, backoff=1.0, max_delay=0.0, jitter=0.0
    )

    with patch("convertpdf.crew.runner.make_formatter"), \
         patch("convertpdf.crew.runner.make_summarizer"), \
         patch("convertpdf.crew.runner.make_format_task_from_extract_file") as mk_fmt_task, \
         patch("convertpdf.crew.runner.make_summarize_task") as mk_sum_task, \
         patch("convertpdf.crew.runner.Crew") as crew_cls:
        format_t = MagicMock()
        summarize_t = MagicMock()
        mk_fmt_task.return_value = format_t
        mk_sum_task.return_value = summarize_t
        crew = MagicMock()
        # APITimeoutError is in the transient set per llm_retry.is_transient.
        crew.kickoff.side_effect = APITimeoutError(request=MagicMock())
        crew_cls.return_value = crew

        fmt_out, sum_out, did_fb = _run_format_summarize_only(
            page_number=1,
            artifacts=artifacts,
            summary_in="prev summary",
            summary_path=layout.summary_path,
            with_summary=True,
            llm=llm,
            retry_config=retry,
            fallback_to_text=True,
            max_summary_chars=800,
        )

    assert did_fb is True
    assert fmt_out == cached_extract
    assert sum_out == "prev summary"
    assert artifacts.format_markdown.read_text(encoding="utf-8") == cached_extract
    assert (
        layout.summary_path.read_text(encoding="utf-8")
        == '{"summary": "old summary"}'
    )

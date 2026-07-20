"""Tests for convertpdf.crew.runner's retry + fallback wiring."""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from openai import APITimeoutError, BadRequestError
from pydantic import ValidationError

from convertpdf.cache import CacheLayout
from convertpdf.crew import runner
from convertpdf.crew.runner import PageImage, run_pipeline
from convertpdf.llm_retry import RetryConfig


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", "https://example.test"))


class _FakeOutput:
    def __init__(self, raw: str) -> None:
        self.raw = raw


class _FakeTask:
    def __init__(self, raw: str = "") -> None:
        self.output = _FakeOutput(raw)


def _make_layout(tmp_path: Path, page_number: int, text: str) -> CacheLayout:
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / f"page_{page_number:04d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (pages_dir / f"page_{page_number:04d}_text.txt").write_text(text, encoding="utf-8")
    return CacheLayout(
        root=tmp_path,
        pages_dir=pages_dir,
        summary_path=tmp_path / "summary.json",
        meta_path=tmp_path / "meta.json",
    )


def _page(page_number: int) -> PageImage:
    return PageImage(
        page_number=page_number,
        width=100,
        height=100,
        image_path=Path(f"page_{page_number:04d}.png"),
    )


def test_run_pipeline_falls_back_to_text_layer_after_transient_retries(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    page = _page(1)
    layout = _make_layout(tmp_path, 1, "hello world\nfrom pdf text layer\n")

    extract_t = _FakeTask()
    format_t = _FakeTask()
    summarize_t = _FakeTask()

    with patch.object(runner, "make_extractor"), \
         patch.object(runner, "make_formatter"), \
         patch.object(runner, "make_summarizer"), \
         patch.object(runner, "make_extract_task", return_value=extract_t), \
         patch.object(runner, "make_format_task", return_value=format_t), \
         patch.object(runner, "make_summarize_task", return_value=summarize_t):
        def _always_timeout() -> None:
            raise APITimeoutError(request=httpx.Request("GET", "https://example.test"))

        with patch.object(runner, "Crew") as crew_cls:
            crew_cls.return_value.kickoff = _always_timeout
            caplog.set_level(logging.INFO, logger="convertpdf.runner")
            results = run_pipeline(
                pages=[page],
                layout=layout,
                with_summary=False,
                resume=False,
                text_hint=False,
                llm=object(),  # type: ignore[arg-type]
                retry_config=RetryConfig(
                    max_attempts=2, initial_delay=0.0, backoff=2.0, jitter=0.0
                ),
                fallback_to_text=True,
            )

    assert len(results) == 1
    md = results[0].markdown
    assert "vision model unavailable" in md
    assert "hello world" in md
    assert "from pdf text layer" in md
    # Cache files are written for resume to work on the next run.
    assert layout.page_extract_path(1).exists()
    assert layout.page_format_path(1).exists()
    assert any("falling back to text layer" in rec.message for rec in caplog.records)


def test_run_pipeline_does_not_fall_back_for_permanent_errors(
    tmp_path: Path,
) -> None:
    page = _page(1)
    layout = _make_layout(tmp_path, 1, "text layer content")

    extract_t = _FakeTask()
    format_t = _FakeTask()

    with patch.object(runner, "make_extractor"), \
         patch.object(runner, "make_formatter"), \
         patch.object(runner, "make_extract_task", return_value=extract_t), \
         patch.object(runner, "make_format_task", return_value=format_t):
        with patch.object(runner, "Crew") as crew_cls:
            crew_cls.return_value.kickoff = lambda: (_ for _ in ()).throw(
                BadRequestError(message="bad", response=_response(400), body=None)
            )
            with pytest.raises(BadRequestError):
                run_pipeline(
                    pages=[page],
                    layout=layout,
                    with_summary=False,
                    resume=False,
                    text_hint=False,
                    llm=object(),  # type: ignore[arg-type]
                    retry_config=RetryConfig(
                        max_attempts=2, initial_delay=0.0, backoff=2.0, jitter=0.0
                    ),
                    fallback_to_text=True,
                )


def test_run_pipeline_propagates_when_fallback_disabled(
    tmp_path: Path,
) -> None:
    page = _page(1)
    layout = _make_layout(tmp_path, 1, "text layer content")

    extract_t = _FakeTask()
    format_t = _FakeTask()

    with patch.object(runner, "make_extractor"), \
         patch.object(runner, "make_formatter"), \
         patch.object(runner, "make_extract_task", return_value=extract_t), \
         patch.object(runner, "make_format_task", return_value=format_t):
        with patch.object(runner, "Crew") as crew_cls:
            crew_cls.return_value.kickoff = lambda: (_ for _ in ()).throw(
                APITimeoutError(request=httpx.Request("GET", "https://example.test"))
            )
            with pytest.raises(APITimeoutError):
                run_pipeline(
                    pages=[page],
                    layout=layout,
                    with_summary=False,
                    resume=False,
                    text_hint=False,
                    llm=object(),  # type: ignore[arg-type]
                    retry_config=RetryConfig(
                        max_attempts=2, initial_delay=0.0, backoff=2.0, jitter=0.0
                    ),
                    fallback_to_text=False,
                )


def _raise_task_output_validation_error() -> None:
    err = ValidationError.from_exception_data(
        title="TaskOutput",
        line_errors=[
            {
                "type": "string_type",
                "loc": ("raw",),
                "input": ["chat completion message with tool_calls"],
                "ctx": {"expected": "string"},
            }
        ],
    )
    raise err


def test_run_pipeline_falls_back_after_task_output_validation_error(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    page = _page(1)
    layout = _make_layout(tmp_path, 1, "recovered text layer content\n")

    extract_t = _FakeTask()
    format_t = _FakeTask()

    with patch.object(runner, "make_extractor"), \
         patch.object(runner, "make_formatter"), \
         patch.object(runner, "make_extract_task", return_value=extract_t), \
         patch.object(runner, "make_format_task", return_value=format_t):
        with patch.object(runner, "Crew") as crew_cls:
            crew_cls.return_value.kickoff = _raise_task_output_validation_error
            caplog.set_level(logging.INFO, logger="convertpdf.runner")
            results = run_pipeline(
                pages=[page],
                layout=layout,
                with_summary=False,
                resume=False,
                text_hint=False,
                llm=object(),  # type: ignore[arg-type]
                retry_config=RetryConfig(
                    max_attempts=1, initial_delay=0.0, backoff=2.0, jitter=0.0
                ),
                fallback_to_text=True,
            )

    assert len(results) == 1
    md = results[0].markdown
    assert "vision model unavailable" in md
    assert "recovered text layer content" in md
    assert layout.page_extract_path(1).exists()
    assert layout.page_format_path(1).exists()
    assert any(
        "validation-fallback" in rec.message or "falling back to text layer" in rec.message
        for rec in caplog.records
    )


def test_run_pipeline_propagates_validation_error_when_fallback_disabled(
    tmp_path: Path,
) -> None:
    page = _page(1)
    layout = _make_layout(tmp_path, 1, "text layer content")

    extract_t = _FakeTask()
    format_t = _FakeTask()

    with patch.object(runner, "make_extractor"), \
         patch.object(runner, "make_formatter"), \
         patch.object(runner, "make_extract_task", return_value=extract_t), \
         patch.object(runner, "make_format_task", return_value=format_t):
        with patch.object(runner, "Crew") as crew_cls:
            crew_cls.return_value.kickoff = _raise_task_output_validation_error
            with pytest.raises(ValidationError):
                run_pipeline(
                    pages=[page],
                    layout=layout,
                    with_summary=False,
                    resume=False,
                    text_hint=False,
                    llm=object(),  # type: ignore[arg-type]
                    retry_config=RetryConfig(
                        max_attempts=1, initial_delay=0.0, backoff=2.0, jitter=0.0
                    ),
                    fallback_to_text=False,
                )


def test_default_run_uses_strict_formatter(tmp_path: Path) -> None:
    """The non-reformat path must continue to use the strict, verbatim
    formatter persona — never the layout-aware one. This guards against
    accidental swap when make_formatter() is refactored."""
    page = _page(1)
    layout = _make_layout(tmp_path, 1, "text layer content\n")

    extract_t = _FakeTask(raw="extracted markdown")
    format_t = _FakeTask(raw="formatted markdown")
    summarize_t = _FakeTask(raw="running summary")

    with patch.object(runner, "make_extractor"), \
         patch.object(runner, "make_summarizer"), \
         patch.object(runner, "make_extract_task", return_value=extract_t), \
         patch.object(runner, "make_format_task", return_value=format_t), \
         patch.object(runner, "make_summarize_task", return_value=summarize_t):
        with patch.object(runner, "make_formatter") as mk:
            with patch.object(runner, "Crew") as crew_cls:
                crew_cls.return_value.kickoff = lambda: None
                run_pipeline(
                    pages=[page],
                    layout=layout,
                    with_summary=False,
                    resume=False,
                    text_hint=False,
                    llm=object(),  # type: ignore[arg-type]
                    retry_config=RetryConfig(
                        max_attempts=1, initial_delay=0.0, backoff=2.0, jitter=0.0
                    ),
                    fallback_to_text=True,
                )
            called_reformat = any(
                call.kwargs.get("reformat") is True
                for call in mk.call_args_list
            )
            assert called_reformat is False, (
                "Default run_pipeline must not call make_formatter(reformat=True)"
            )
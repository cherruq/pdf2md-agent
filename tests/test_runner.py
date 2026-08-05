"""Tests for pdf2md_agent.crew.runner's retry + fallback wiring."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from openai import APITimeoutError, BadRequestError
from pydantic import ValidationError

from pdf2md_agent.cache import CacheLayout, CacheNoCacheFlags
from pdf2md_agent.config import ConversionConfig
from pdf2md_agent.crew import runner
from pdf2md_agent.crew.runner import run_pipeline
from pdf2md_agent.crew.types import PageRunContext, RenderedPage
from pdf2md_agent.llm_retry import RetryConfig


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
        meta_path=tmp_path / "meta.json",
    )


def _page(page_number: int) -> RenderedPage:
    return RenderedPage(
        ctx=PageRunContext(page_number=page_number, idx=page_number, total=10, page_started=0.0),
        width=100,
        height=100,
        image_path=Path(f"page_{page_number:04d}.png"),
    )


def _no_cache() -> CacheNoCacheFlags:
    return CacheNoCacheFlags()


def _make_config(
    layout: CacheLayout,
    no_cache: CacheNoCacheFlags = CacheNoCacheFlags(),
    retry_config: RetryConfig | None = None,
    fallback_to_text: bool = True,
) -> ConversionConfig:
    return ConversionConfig(
        pdf=Path("dummy.pdf"),
        output=Path("out.md"),
        dpi=144,
        layout=layout,
        render_target=Path("render"),
        resolved_pages=None,
        no_cache=no_cache,
        retry_config=retry_config or RetryConfig(max_attempts=2, initial_delay=0.001, jitter=0.0),
        text_hint=False,
        image_long_side=1536,
        image_jpeg_quality=85,
        ctx_limit=100000,
        request_timeout_seconds=60.0,
        fallback_to_text=fallback_to_text,
    )


def test_run_pipeline_falls_back_to_text_layer_after_transient_retries(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    page = _page(1)
    layout = _make_layout(tmp_path, 1, "hello world\nfrom pdf text layer\n")

    extract_t = _FakeTask()

    with patch.object(runner, "make_extractor"), patch.object(runner, "make_extract_task", return_value=extract_t):

        def _always_timeout() -> None:
            raise APITimeoutError(request=httpx.Request("GET", "https://example.test"))

        with patch.object(runner, "Crew") as crew_cls:
            crew_cls.return_value.kickoff = _always_timeout
            caplog.set_level(logging.INFO, logger="pdf2md_agent.runner")
            results = run_pipeline(
                pages=[page],
                config=_make_config(
                    layout=layout,
                    retry_config=RetryConfig(max_attempts=2, initial_delay=0.001, jitter=0.0),
                ),
                llm=object(),  # type: ignore[arg-type]
            )

    assert len(results) == 1
    md = results[0].markdown
    assert "vision model unavailable" in md
    assert "hello world" in md
    assert "from pdf text layer" in md
    assert layout.page_format_path(1).exists()
    assert any("falling back to text layer" in rec.message for rec in caplog.records)


def test_run_pipeline_does_not_fall_back_for_permanent_errors(
    tmp_path: Path,
) -> None:
    page = _page(1)
    layout = _make_layout(tmp_path, 1, "text layer content")

    extract_t = _FakeTask()

    with patch.object(runner, "make_extractor"), patch.object(runner, "make_extract_task", return_value=extract_t):
        with patch.object(runner, "Crew") as crew_cls:
            crew_cls.return_value.kickoff = lambda: (_ for _ in ()).throw(
                BadRequestError(message="bad", response=_response(400), body=None)
            )
            with pytest.raises(BadRequestError):
                run_pipeline(
                    pages=[page],
                    config=_make_config(
                        layout=layout,
                        retry_config=RetryConfig(max_attempts=2, initial_delay=0.001, jitter=0.0),
                    ),
                    llm=object(),  # type: ignore[arg-type]
                )


def test_run_pipeline_propagates_when_fallback_disabled(
    tmp_path: Path,
) -> None:
    page = _page(1)
    layout = _make_layout(tmp_path, 1, "text layer content")

    extract_t = _FakeTask()

    with patch.object(runner, "make_extractor"), patch.object(runner, "make_extract_task", return_value=extract_t):
        with patch.object(runner, "Crew") as crew_cls:
            crew_cls.return_value.kickoff = lambda: (_ for _ in ()).throw(
                APITimeoutError(request=httpx.Request("GET", "https://example.test"))
            )
            with pytest.raises(APITimeoutError):
                run_pipeline(
                    pages=[page],
                    config=_make_config(
                        layout=layout,
                        retry_config=RetryConfig(max_attempts=2, initial_delay=0.001, jitter=0.0),
                        fallback_to_text=False,
                    ),
                    llm=object(),  # type: ignore[arg-type]
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

    with patch.object(runner, "make_extractor"), patch.object(runner, "make_extract_task", return_value=extract_t):
        with patch.object(runner, "Crew") as crew_cls:
            crew_cls.return_value.kickoff = _raise_task_output_validation_error
            caplog.set_level(logging.INFO, logger="pdf2md_agent.runner")
            results = run_pipeline(
                pages=[page],
                config=_make_config(
                    layout=layout,
                    retry_config=RetryConfig(max_attempts=1, initial_delay=0.001, jitter=0.0),
                ),
                llm=object(),  # type: ignore[arg-type]
            )

    assert len(results) == 1
    md = results[0].markdown
    assert "vision model unavailable" in md
    assert "recovered text layer content" in md
    assert layout.page_format_path(1).exists()
    assert any(
        "validation-fallback" in rec.message or "falling back to text layer" in rec.message for rec in caplog.records
    )


def test_run_pipeline_propagates_validation_error_when_fallback_disabled(
    tmp_path: Path,
) -> None:
    page = _page(1)
    layout = _make_layout(tmp_path, 1, "text layer content")

    extract_t = _FakeTask()

    with patch.object(runner, "make_extractor"), patch.object(runner, "make_extract_task", return_value=extract_t):
        with patch.object(runner, "Crew") as crew_cls:
            crew_cls.return_value.kickoff = _raise_task_output_validation_error
            with pytest.raises(ValidationError):
                run_pipeline(
                    pages=[page],
                    config=_make_config(
                        layout=layout,
                        retry_config=RetryConfig(max_attempts=1, initial_delay=0.001, jitter=0.0),
                        fallback_to_text=False,
                    ),
                    llm=object(),  # type: ignore[arg-type]
                )

"""Tests for pdf2md_agent.crew.runner per-call retry + fallback wiring.

The runner now uses ``pdf2md_agent.raw_pipeline`` for explicit LLM calls
(no CrewAI state). Tests stub ``runner._make_client`` with a fake OpenAI
client whose ``chat.completions.create`` returns canned strings or
raises injected exceptions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from openai import APITimeoutError, BadRequestError

from pdf2md_agent.cache import CacheLayout, CacheNoCacheFlags
from pdf2md_agent.crew import runner
from pdf2md_agent.crew.runner import PageImage, run_pipeline
from pdf2md_agent.llm_retry import RetryConfig


# --- OpenAI fake client -------------------------------------------------


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """Captures every ``create`` call and returns canned responses in order.

    Pass ``fail_after_call_index=N`` to make the (N+1)-th call raise an
    injected exception (default: ``APITimeoutError``).
    """

    def __init__(
        self,
        responses: list[str],
        *,
        fail_after_call_index: int | None = None,
        exception_factory: Any = None,
    ) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self._fail_after = fail_after_call_index
        self._exception_factory = exception_factory

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if self._fail_after is not None and len(self.calls) > self._fail_after:
            if self._exception_factory is not None:
                raise self._exception_factory()
            raise APITimeoutError(
                request=httpx.Request("GET", "https://example.test")
            )
        content = self._responses.pop(0) if self._responses else ""
        return _FakeResponse(content)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = _FakeChat(completions)


def _make_fake_client(
    responses: list[str],
    *,
    fail_after_call_index: int | None = None,
    exception_factory: Any = None,
) -> _FakeClient:
    """Build a fake OpenAI client whose completions return ``responses``.

    With ``fail_after_call_index=N``, every call *after* the Nth raises
    the configured exception (default ``APITimeoutError``). Use
    ``exception_factory`` to inject ``BadRequestError`` etc.
    """
    completions = _FakeCompletions(
        responses,
        fail_after_call_index=fail_after_call_index,
        exception_factory=exception_factory,
    )
    return _FakeClient(completions)


# --- layout / page helpers ----------------------------------------------


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


def _page(page_number: int, tmp_path: Path | None = None) -> PageImage:
    """Build a PageImage whose path points to a real PNG so _encode_local_image succeeds."""
    base = tmp_path or Path("/tmp")
    png_path = base / f"page_{page_number:04d}.png"
    if not png_path.exists():
        from PIL import Image
        png_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 40), "white").save(png_path, "PNG")
    return PageImage(
        page_number=page_number,
        width=100,
        height=100,
        image_path=png_path,
    )


def _no_cache() -> CacheNoCacheFlags:
    return CacheNoCacheFlags()


# --- tests --------------------------------------------------------------


def test_run_pipeline_falls_back_to_text_layer_after_transient_retries(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A persistent APITimeoutError across all calls triggers text-layer fallback."""
    page = _page(1, tmp_path)
    layout = _make_layout(tmp_path, 1, "hello world\nfrom pdf text layer\n")

    fake = _make_fake_client(
        responses=[""],
        fail_after_call_index=0,
        exception_factory=lambda: APITimeoutError(
            request=httpx.Request("GET", "https://example.test")
        ),
    )

    with patch.object(runner, "_make_client", return_value=fake):
        caplog.set_level(logging.INFO, logger="pdf2md_agent.runner")
        results = run_pipeline(
            pages=[page],
            layout=layout,
            with_summary=False,
            no_cache=_no_cache(),
            text_hint=False,
            retry_config=RetryConfig(
                max_attempts=2, initial_delay=0.001, jitter=0.0
            ),
            fallback_to_text=True,
            image_long_side=40,
            image_min_long_side=40,
            image_jpeg_quality=70,
        )

    assert len(results) == 1
    md = results[0].markdown
    assert "vision model unavailable" in md
    assert "hello world" in md
    assert "from pdf text layer" in md
    assert layout.page_extract_path(1).exists()
    assert layout.page_format_path(1).exists()
    assert any("falling back to text layer" in rec.message for rec in caplog.records)


def test_run_pipeline_does_not_fall_back_for_permanent_errors(
    tmp_path: Path,
) -> None:
    """BadRequestError (non-transient) propagates even with fallback enabled."""
    page = _page(1, tmp_path)
    layout = _make_layout(tmp_path, 1, "text layer content")

    fake = _make_fake_client(
        responses=[""],
        fail_after_call_index=0,
        exception_factory=lambda: BadRequestError(
            message="bad",
            response=httpx.Response(
                400, request=httpx.Request("GET", "https://example.test")
            ),
            body=None,
        ),
    )

    with patch.object(runner, "_make_client", return_value=fake):
        with pytest.raises(BadRequestError):
            run_pipeline(
                pages=[page],
                layout=layout,
                with_summary=False,
                no_cache=_no_cache(),
                text_hint=False,
                retry_config=RetryConfig(
                    max_attempts=2, initial_delay=0.001, jitter=0.0
                ),
                fallback_to_text=True,
                image_long_side=40,
                image_min_long_side=40,
                image_jpeg_quality=70,
            )


def test_run_pipeline_propagates_when_fallback_disabled(
    tmp_path: Path,
) -> None:
    """Even transient errors propagate when ``fallback_to_text=False``."""
    page = _page(1, tmp_path)
    layout = _make_layout(tmp_path, 1, "text layer content")

    fake = _make_fake_client(
        responses=[""],
        fail_after_call_index=0,
        exception_factory=lambda: APITimeoutError(
            request=httpx.Request("GET", "https://example.test")
        ),
    )

    with patch.object(runner, "_make_client", return_value=fake):
        with pytest.raises(APITimeoutError):
            run_pipeline(
                pages=[page],
                layout=layout,
                with_summary=False,
                no_cache=_no_cache(),
                text_hint=False,
                retry_config=RetryConfig(
                    max_attempts=2, initial_delay=0.001, jitter=0.0
                ),
                fallback_to_text=False,
                image_long_side=40,
                image_min_long_side=40,
                image_jpeg_quality=70,
            )


def test_default_run_uses_strict_formatter(tmp_path: Path) -> None:
    """The default formatter is the strict, verbatim CommonMark formatter.

    Strengthened (D8-012): verifies the formatter's CommonMark-shaped
    output propagates through to ``PageResult.markdown`` and the on-disk
    ``format.md``. Each LLM call gets a canned response and the
    formatter call's payload is asserted verbatim on disk and in the
    returned PageResult.
    """
    page = _page(1, tmp_path)
    layout = _make_layout(tmp_path, 1, "text layer content\n")

    commonmark_payload = (
        "# Section\n\n"
        "First paragraph with **bold** text.\n\n"
        "- bullet one\n"
        "- bullet two\n\n"
        "```python\n"
        "print('hi')\n"
        "```"
    )
    fake = _make_fake_client(
        responses=["extracted markdown", commonmark_payload, "running summary"]
    )

    with patch.object(runner, "_make_client", return_value=fake):
        results = run_pipeline(
            pages=[page],
            layout=layout,
            with_summary=False,
            no_cache=_no_cache(),
            text_hint=False,
            retry_config=RetryConfig(
                max_attempts=1, initial_delay=0.001, jitter=0.0
            ),
            fallback_to_text=True,
            image_long_side=40,
            image_min_long_side=40,
            image_jpeg_quality=70,
        )

    assert len(results) == 1
    assert results[0].page_number == 1
    assert results[0].markdown == commonmark_payload, (
        "formatter output must propagate to PageResult.markdown verbatim"
    )
    assert "# Section" in results[0].markdown
    assert "- bullet one" in results[0].markdown
    assert "```python" in results[0].markdown
    on_disk = layout.page_format_path(1).read_text(encoding="utf-8")
    assert on_disk == commonmark_payload

    completions = fake.chat.completions
    assert len(completions.calls) == 2, (
        "without summary, runner makes exactly 2 LLM calls "
        "(extract + format)"
    )
    extract_call, format_call = completions.calls
    # Extractor: system=persona, user=[text + image_url data URL]
    assert extract_call["model"]  # any non-empty model name
    assert extract_call["messages"][0]["role"] == "system"
    assert extract_call["messages"][1]["role"] == "user"
    user_content = extract_call["messages"][1]["content"]
    assert isinstance(user_content, list)
    assert any(
        part.get("type") == "image_url"
        and part["image_url"]["url"].startswith("data:image/jpeg;base64,")
        for part in user_content
    )
    # Formatter: text-only, no image
    assert format_call["messages"][1]["role"] == "user"
    assert isinstance(format_call["messages"][1]["content"], str)
    assert "extracted markdown" in format_call["messages"][1]["content"]

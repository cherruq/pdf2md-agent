"""Tests for the ``--no-cache-*`` family and ``CacheNoCacheFlags`` plumbing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from pdf2md_agent import cli
from pdf2md_agent.cache import CacheLayout, CacheNoCacheFlags
from pdf2md_agent.config import ConversionConfig
from pdf2md_agent.crew import runner
from pdf2md_agent.crew.runner import run_pipeline
from pdf2md_agent.crew.types import PageRunContext, RenderedPage
from pdf2md_agent.llm_retry import RetryConfig


class _FakeOutput:
    def __init__(self, raw: str) -> None:
        self.raw = raw


class _FakeTask:
    def __init__(self, raw: str = "") -> None:
        self.output = _FakeOutput(raw)


def _page(page_number: int) -> RenderedPage:
    return RenderedPage(
        ctx=PageRunContext(page_number=page_number, idx=page_number, total=10, page_started=0.0),
        width=100,
        height=100,
        image_path=Path(f"page_{page_number:04d}.png"),
    )


def _layout(tmp_path: Path, page_number: int) -> CacheLayout:
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / f"page_{page_number:04d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return CacheLayout(
        root=tmp_path,
        pages_dir=pages_dir,
        meta_path=tmp_path / "meta.json",
    )


def _make_config(
    layout: CacheLayout,
    no_cache: CacheNoCacheFlags = CacheNoCacheFlags(),
    retry_config: RetryConfig | None = None,
) -> ConversionConfig:
    return ConversionConfig(
        pdf=Path("dummy.pdf"),
        output=Path("out.md"),
        dpi=144,
        layout=layout,
        render_target=Path("render"),
        resolved_pages=None,
        no_cache=no_cache,
        retry_config=retry_config or RetryConfig(max_attempts=1, initial_delay=0.001, jitter=0.0),
        text_hint=False,
        image_long_side=1536,
        image_jpeg_quality=85,
        ctx_limit=100000,
        request_timeout_seconds=60.0,
    )


# --- CLI parser surface -----------------------------------------------------


def test_no_cache_defaults_all_false() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["in.pdf", "-o", "out.md"])
    assert args.no_cache_all is False
    assert args.no_cache_render is False
    assert args.no_cache_text is False
    assert args.no_cache_resized is False
    assert args.no_cache_format is False


def test_no_cache_all_sets_every_flag() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["in.pdf", "-o", "out.md", "--no-cache-all"])
    assert args.no_cache_all is True
    assert args.no_cache_render is True
    assert args.no_cache_text is True
    assert args.no_cache_resized is True
    assert args.no_cache_format is True


def test_resume_and_reformat_flags_rejected() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["in.pdf", "-o", "out.md", "--resume"])
    with pytest.raises(SystemExit):
        parser.parse_args(["in.pdf", "-o", "out.md", "--reformat"])


def test_resolve_no_cache_flags_mirrors_args() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["in.pdf", "-o", "out.md", "--no-cache-format", "--no-cache-text"])
    flags = cli._resolve_no_cache_flags(args)
    assert flags == CacheNoCacheFlags(format=True, text=True)


@pytest.mark.parametrize(
    "flags",
    [
        CacheNoCacheFlags(),
        CacheNoCacheFlags(format=True),
        CacheNoCacheFlags(render=True, text=True, resized=True),
    ],
)
def test_cache_no_cache_flags_all_false_for_partial(flags: CacheNoCacheFlags) -> None:
    assert flags.all() is False


def test_cache_no_cache_flags_all_true_only_when_every_flag_set() -> None:
    assert CacheNoCacheFlags(render=True, text=True, resized=True, format=True).all() is True


# --- per-page priority matrix ----------------------------------------------


def _seed_complete_page(layout: CacheLayout, page_number: int) -> None:
    layout.page_format_path(page_number).write_text("final md", encoding="utf-8")
    layout.page_text_path(page_number).write_text("text hint", encoding="utf-8")


def test_no_cache_format_reruns_full_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    page = _page(1)
    layout = _layout(tmp_path, 1)
    _seed_complete_page(layout, 1)

    extract_t = _FakeTask(raw="fresh md")

    calls: list[str] = []

    def _track(*_args: object, **_kwargs: object) -> None:
        calls.append("kickoff")

    with (
        patch.object(runner, "make_extractor"),
        patch.object(runner, "make_extract_task", return_value=extract_t),
        patch.object(runner, "Crew") as crew_cls,
    ):
        crew_cls.return_value.kickoff = _track
        results = run_pipeline(
            pages=[page],
            config=_make_config(layout=layout, no_cache=CacheNoCacheFlags(format=True)),
            llm=object(),  # type: ignore[arg-type]
        )

    assert calls, "full pipeline must run when --no-cache-format is set"
    assert results[0].markdown == "fresh md"
    assert layout.page_format_path(1).read_text(encoding="utf-8") == "fresh md"


def test_trust_format_short_circuits_full_pipeline(tmp_path: Path) -> None:
    page = _page(1)
    layout = _layout(tmp_path, 1)
    _seed_complete_page(layout, 1)

    kickoff_calls: list[None] = []

    def _track() -> None:
        kickoff_calls.append(None)

    with (
        patch.object(runner, "make_extractor"),
        patch.object(runner, "make_extract_task"),
        patch.object(runner, "Crew") as crew_cls,
    ):
        crew_cls.return_value.kickoff = _track
        results = run_pipeline(
            pages=[page],
            config=_make_config(layout=layout),
            llm=object(),  # type: ignore[arg-type]
        )

    assert kickoff_calls == [], "trusting format.md must short-circuit the pipeline"
    assert results[0].markdown == "final md"

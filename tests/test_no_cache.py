"""Tests for the ``--no-cache-*`` family and ``CacheNoCacheFlags`` plumbing.

Maps the path-B contract:

* ``--no-cache-all`` flips every per-resource flag.
* Default = trust cache; all flags ``False``.
* Per-page priority: format short-circuit → extract short-circuit → full
  pipeline.
* ``has_cached_extract`` rejects empty extract.txt (H1 sentinel).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from pdf2md_agent import cli
from pdf2md_agent.cache import CacheLayout, CacheNoCacheFlags, has_cached_extract
from pdf2md_agent.crew import runner
from pdf2md_agent.crew.runner import PageImage, run_pipeline
from pdf2md_agent.llm_retry import RetryConfig


class _FakeOutput:
    def __init__(self, raw: str) -> None:
        self.raw = raw


class _FakeTask:
    def __init__(self, raw: str = "") -> None:
        self.output = _FakeOutput(raw)


def _page(page_number: int) -> PageImage:
    return PageImage(
        page_number=page_number,
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
        summary_path=tmp_path / "summary.json",
        meta_path=tmp_path / "meta.json",
    )


# --- CLI parser surface -----------------------------------------------------


def test_no_cache_defaults_all_false() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["in.pdf", "-o", "out.md"])
    assert args.no_cache_all is False
    assert args.no_cache_render is False
    assert args.no_cache_text is False
    assert args.no_cache_resized is False
    assert args.no_cache_extract is False
    assert args.no_cache_format is False
    assert args.no_cache_summary is False


def test_no_cache_extract_individual_flag() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["in.pdf", "-o", "out.md", "--no-cache-extract"])
    assert args.no_cache_extract is True
    assert args.no_cache_format is False


def test_no_cache_all_sets_every_flag() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["in.pdf", "-o", "out.md", "--no-cache-all"])
    assert args.no_cache_all is True
    assert args.no_cache_render is True
    assert args.no_cache_text is True
    assert args.no_cache_resized is True
    assert args.no_cache_extract is True
    assert args.no_cache_format is True
    assert args.no_cache_summary is True


def test_resume_and_reformat_flags_rejected() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["in.pdf", "-o", "out.md", "--resume"])
    with pytest.raises(SystemExit):
        parser.parse_args(["in.pdf", "-o", "out.md", "--reformat"])


def test_resolve_no_cache_flags_mirrors_args() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        ["in.pdf", "-o", "out.md", "--no-cache-format", "--no-cache-text"]
    )
    flags = cli._resolve_no_cache_flags(args)
    assert flags == CacheNoCacheFlags(format=True, text=True)


@pytest.mark.parametrize(
    "flags",
    [
        CacheNoCacheFlags(),
        CacheNoCacheFlags(format=True),
        CacheNoCacheFlags(extract=True, format=True),
        CacheNoCacheFlags(render=True, text=True, resized=True, extract=True),
    ],
)
def test_cache_no_cache_flags_all_false_for_partial(flags: CacheNoCacheFlags) -> None:
    assert flags.all() is False


def test_cache_no_cache_flags_all_true_only_when_every_flag_set() -> None:
    assert CacheNoCacheFlags(
        render=True, text=True, resized=True, extract=True, format=True, summary=True
    ).all() is True


# --- H1 sentinel: has_cached_extract rejects empty extract.txt -------------


def test_has_cached_extract_false_when_extract_empty(tmp_path: Path) -> None:
    layout = CacheLayout.for_pdf(tmp_path / "cache", tmp_path / "fake.pdf")
    layout.page_extract_path(1).write_text("", encoding="utf-8")
    assert has_cached_extract(layout, 1) is False


def test_has_cached_extract_true_when_extract_nonempty(tmp_path: Path) -> None:
    layout = CacheLayout.for_pdf(tmp_path / "cache", tmp_path / "fake.pdf")
    layout.page_extract_path(1).write_text("body", encoding="utf-8")
    assert has_cached_extract(layout, 1) is True


def test_has_cached_extract_false_when_extract_is_fallback_sentinel(
    tmp_path: Path,
) -> None:
    """Regression: the runner's text-layer fallback writes a NON-EMPTY
    sentinel line into ``extract.txt`` so the file isn't silently trusted
    as a real extractor payload. ``has_cached_extract`` must parse the
    prefix and reject the sentinel; otherwise ``--no-cache-extract``
    would feed the marker text into the formatter.
    """
    from pdf2md_agent.crew.runner import _FALLBACK_SENTINEL

    layout = CacheLayout.for_pdf(tmp_path / "cache", tmp_path / "fake.pdf")
    layout.page_extract_path(1).write_text(
        _FALLBACK_SENTINEL.format(page=1), encoding="utf-8"
    )
    assert has_cached_extract(layout, 1) is False


# --- per-page priority matrix ----------------------------------------------


def _seed_complete_page(layout: CacheLayout, page_number: int) -> None:
    layout.page_extract_path(page_number).write_text(
        "extracted body", encoding="utf-8"
    )
    layout.page_format_path(page_number).write_text(
        "final md", encoding="utf-8"
    )
    layout.page_text_path(page_number).write_text(
        "text hint", encoding="utf-8"
    )


def test_no_cache_format_reruns_full_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = _page(1)
    layout = _layout(tmp_path, 1)
    _seed_complete_page(layout, 1)

    extract_t = _FakeTask(raw="fresh extract")
    format_t = _FakeTask(raw="fresh md")

    calls: list[str] = []

    def _track(*_args: object, **_kwargs: object) -> None:
        calls.append("kickoff")

    with patch.object(runner, "make_extractor"), \
         patch.object(runner, "make_formatter"), \
         patch.object(runner, "make_extract_task", return_value=extract_t), \
         patch.object(runner, "make_format_task", return_value=format_t), \
         patch.object(runner, "make_summarize_task"), \
         patch.object(runner, "Crew") as crew_cls:
        crew_cls.return_value.kickoff = _track
        results = run_pipeline(
            pages=[page],
            layout=layout,
            with_summary=False,
            no_cache=CacheNoCacheFlags(format=True),
            text_hint=False,
            llm=object(),  # type: ignore[arg-type]
            retry_config=RetryConfig(
                max_attempts=1, initial_delay=0.0, backoff=2.0, jitter=0.0
            ),
            fallback_to_text=True,
        )

    assert calls, "full pipeline must run when --no-cache-format is set"
    assert results[0].markdown == "fresh md"
    assert layout.page_format_path(1).read_text(encoding="utf-8") == "fresh md"


def _seed_extract_only(layout: CacheLayout, page_number: int) -> None:
    layout.page_extract_path(page_number).write_text(
        "extracted body", encoding="utf-8"
    )
    layout.page_text_path(page_number).write_text(
        "text hint", encoding="utf-8"
    )


def test_no_cache_extract_runs_formatter_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = _page(1)
    layout = _layout(tmp_path, 1)
    _seed_extract_only(layout, 1)

    extract_t = _FakeTask(raw="unused")
    format_t = _FakeTask(raw="re-formatted md")
    summarize_t = _FakeTask(raw="new summary")

    extractor_calls: list[object] = []

    def _track_extractor(*_args: object, **_kwargs: object) -> object:
        extractor_calls.append(object())
        class _E: ...
        return _E()

    crew_obj = type("C", (), {})()
    crew_obj.kickoff = lambda: None

    with patch.object(runner, "make_extractor", side_effect=_track_extractor), \
         patch.object(runner, "make_formatter"), \
         patch.object(runner, "make_summarizer"), \
         patch.object(runner, "make_format_task_from_extract_file", return_value=format_t), \
         patch.object(runner, "make_summarize_task", return_value=summarize_t), \
         patch.object(runner, "_output", side_effect=lambda t: getattr(t.output, "raw", "")), \
         patch.object(runner, "Crew", return_value=crew_obj):
        results = run_pipeline(
            pages=[page],
            layout=layout,
            with_summary=True,
            no_cache=CacheNoCacheFlags(extract=True),
            text_hint=False,
            llm=object(),  # type: ignore[arg-type]
            retry_config=RetryConfig(
                max_attempts=1, initial_delay=0.0, backoff=2.0, jitter=0.0
            ),
            fallback_to_text=True,
        )

    assert extractor_calls == [], "make_extractor must NOT run with --no-cache-extract"
    assert results[0].markdown == "re-formatted md"
    assert layout.page_format_path(1).read_text(encoding="utf-8") == "re-formatted md"
    assert layout.summary_path.exists()


def test_no_cache_extract_falls_through_when_extract_missing(
    tmp_path: Path,
) -> None:
    page = _page(1)
    layout = _layout(tmp_path, 1)
    # No extract.txt on disk → must fall through to full pipeline.

    extract_t = _FakeTask(raw="fresh extract")
    format_t = _FakeTask(raw="fresh md")

    with patch.object(runner, "make_extractor"), \
         patch.object(runner, "make_formatter"), \
         patch.object(runner, "make_extract_task", return_value=extract_t), \
         patch.object(runner, "make_format_task", return_value=format_t), \
         patch.object(runner, "make_summarize_task"), \
         patch.object(runner, "Crew") as crew_cls:
        crew_cls.return_value.kickoff = lambda: None
        results = run_pipeline(
            pages=[page],
            layout=layout,
            with_summary=False,
            no_cache=CacheNoCacheFlags(extract=True),
            text_hint=False,
            llm=object(),  # type: ignore[arg-type]
            retry_config=RetryConfig(
                max_attempts=1, initial_delay=0.0, backoff=2.0, jitter=0.0
            ),
            fallback_to_text=True,
        )

    assert results[0].markdown == "fresh md"


def test_trust_format_short_circuits_full_pipeline(tmp_path: Path) -> None:
    page = _page(1)
    layout = _layout(tmp_path, 1)
    _seed_complete_page(layout, 1)

    kickoff_calls: list[None] = []

    def _track() -> None:
        kickoff_calls.append(None)

    with patch.object(runner, "make_extractor"), \
         patch.object(runner, "make_formatter"), \
         patch.object(runner, "make_extract_task"), \
         patch.object(runner, "make_format_task"), \
         patch.object(runner, "Crew") as crew_cls:
        crew_cls.return_value.kickoff = _track
        results = run_pipeline(
            pages=[page],
            layout=layout,
            with_summary=False,
            no_cache=CacheNoCacheFlags(),
            text_hint=False,
            llm=object(),  # type: ignore[arg-type]
            retry_config=RetryConfig(
                max_attempts=1, initial_delay=0.0, backoff=2.0, jitter=0.0
            ),
            fallback_to_text=True,
        )

    assert kickoff_calls == [], "trusting format.md must short-circuit the pipeline"
    assert results[0].markdown == "final md"


def test_no_cache_summary_does_not_seed(tmp_path: Path) -> None:
    page = _page(1)
    layout = _layout(tmp_path, 1)
    # Pre-existing summary.json must be ignored when --no-cache-summary is set.
    layout.summary_path.write_text(
        '{"summary": "stale carry-over"}', encoding="utf-8"
    )

    extract_t = _FakeTask(raw="new extract")
    format_t = _FakeTask(raw="new md")
    summarize_t = _FakeTask(raw="new summary text")

    written: list[Path] = []

    def _track_write(path: Path, _payload: str) -> None:
        written.append(path)

    with patch.object(runner, "write_summary", side_effect=_track_write), \
         patch.object(runner, "make_extractor"), \
         patch.object(runner, "make_formatter"), \
         patch.object(runner, "make_summarizer"), \
         patch.object(runner, "make_extract_task", return_value=extract_t), \
         patch.object(runner, "make_format_task", return_value=format_t), \
         patch.object(runner, "make_summarize_task", return_value=summarize_t), \
         patch.object(runner, "Crew") as crew_cls:
        crew_cls.return_value.kickoff = lambda: None
        run_pipeline(
            pages=[page],
            layout=layout,
            with_summary=True,
            no_cache=CacheNoCacheFlags(summary=True),
            text_hint=False,
            llm=object(),  # type: ignore[arg-type]
            retry_config=RetryConfig(
                max_attempts=1, initial_delay=0.0, backoff=2.0, jitter=0.0
            ),
            fallback_to_text=True,
        )

    assert written == [], "summary.json must NOT be written when --no-cache-summary is set"
    assert layout.summary_path.read_text(encoding="utf-8") == '{"summary": "stale carry-over"}'

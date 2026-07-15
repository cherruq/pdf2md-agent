"""Per-page CrewAI runner: maintains running summary state, writes cache."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from crewai import Crew, LLM, Process

from convertpdf.cache import (
    CacheLayout,
    is_page_complete,
    read_summary,
    write_summary,
)
from convertpdf.crew.agents import (
    make_extractor,
    make_formatter,
    make_summarizer,
)
from convertpdf.crew.tasks import (
    make_extract_task,
    make_format_task,
    make_summarize_task,
)
from convertpdf.pdf_renderer import PageImage

log = logging.getLogger("convertpdf.runner")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks from model output."""
    return _THINK_RE.sub("", text).strip()


def _output(task) -> str:
    """Extract clean text from a CrewAI task's output."""
    out = getattr(task, "output", None)
    if out is None:
        return ""
    raw = getattr(out, "raw", None)
    text = raw if isinstance(raw, str) else str(out)
    return _strip_think(text)


@dataclass(frozen=True, slots=True)
class PageResult:
    """One page's final markdown + the running summary after this page."""

    page_number: int
    markdown: str
    summary: str


def run_pipeline(
    *,
    pages: list[PageImage],
    layout: CacheLayout,
    with_summary: bool,
    resume: bool,
    text_hint: bool,
    llm: LLM,
) -> list[PageResult]:
    """Run the per-page CrewAI pipeline across ``pages`` and return page results.

    ``text_hint`` controls whether the native PDF text layer is fed to the
    extractor agent as a per-page hint. Disabled → pass empty string.
    """
    extractor = make_extractor(llm)
    formatter = make_formatter(llm)
    summarizer = make_summarizer(llm) if with_summary else None

    summary = read_summary(layout.summary_path)
    results: list[PageResult] = []
    pipeline_started = time.monotonic()
    total = len(pages)
    phases = "extract + format + summarize" if with_summary else "extract + format"

    for idx, page in enumerate(pages, start=1):
        artifacts = layout.artifacts_for(page)

        if resume and is_page_complete(layout, page.page_number):
            cached_md = artifacts.format_markdown.read_text(encoding="utf-8").strip()
            if with_summary:
                summary = read_summary(layout.summary_path)
            log.info("  [%d/%d] page %d: cached, skipping", idx, total, page.page_number)
            results.append(PageResult(page.page_number, cached_md, summary))
            continue

        text_hint_str = (
            artifacts.page_text.read_text(encoding="utf-8") if text_hint else ""
        )

        log.info("  [%d/%d] page %d: %s starting", idx, total, page.page_number, phases)
        page_started = time.monotonic()

        extract_t = make_extract_task(
            extractor, page.image_path, text_hint=text_hint_str, previous_summary=summary
        )
        format_t = make_format_task(formatter, extract_t)
        tasks = [extract_t, format_t]
        agents = [extractor, formatter]
        if summarizer is not None:
            summarize_t = make_summarize_task(summarizer, format_t, summary)
            tasks.append(summarize_t)
            agents.append(summarizer)
        else:
            summarize_t = None

        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
        )
        crew.kickoff()

        extract_text = _output(extract_t)
        format_md = _output(format_t)
        artifacts.extract_text.write_text(extract_text, encoding="utf-8")
        artifacts.format_markdown.write_text(format_md, encoding="utf-8")

        if summarize_t is not None:
            summary = _output(summarize_t)
            write_summary(layout.summary_path, summary)

        elapsed = time.monotonic() - page_started
        log.info(
            "  [%d/%d] page %d: done in %.1fs (%s chars)",
            idx,
            total,
            page.page_number,
            elapsed,
            f"{len(format_md):,}",
        )
        results.append(PageResult(page.page_number, format_md, summary))

    total_elapsed = time.monotonic() - pipeline_started
    log.info(
        "pipeline complete: %d page(s) in %.1fs (%.1fs avg)",
        total,
        total_elapsed,
        total_elapsed / max(total, 1),
    )
    return results
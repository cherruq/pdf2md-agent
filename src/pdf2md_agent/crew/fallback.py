"""Text-layer fallback helpers for the per-page pipeline.

When the vision model is unreachable after every retry, or when its output
fails CrewAI's task-output validation, the runner emits a fenced
text-layer Markdown stub instead of crashing the whole run.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from pdf2md_agent.cache import FALLBACK_SENTINEL, PageArtifacts
from pdf2md_agent.crew.results import PageResult

log = logging.getLogger("pdf2md_agent.runner")


def _text_layer_fallback(artifacts: PageArtifacts) -> str:
    """Build a best-effort markdown page from the PDF's native text layer."""
    text = artifacts.page_text.read_text(encoding="utf-8").strip()
    if not text:
        return (
            "*(vision model unavailable and PDF text layer is empty for this "
            "page — no content recovered)*"
        )
    return (
        "*(vision model unavailable — falling back to PDF text layer; "
        "tables, figures, and layout are NOT preserved)*\n\n"
        "```\n"
        f"{text}\n"
        "```\n"
    )


@dataclass(frozen=True, slots=True)
class FallbackRecord:
    """Arguments bundled for :func:`_record_text_layer_fallback`."""

    idx: int
    total: int
    page_number: int
    page_started: float
    artifacts: PageArtifacts
    completion_label: str


def _record_text_layer_fallback(
    *,
    idx: int,
    total: int,
    page_number: int,
    page_started: float,
    artifacts: PageArtifacts,
    completion_label: str,
    **_kwargs: object,
) -> PageResult:
    """Write the fallback artifacts for one page and return its :class:`PageResult`."""
    record = FallbackRecord(
        idx=idx,
        total=total,
        page_number=page_number,
        page_started=page_started,
        artifacts=artifacts,
        completion_label=completion_label,
    )
    format_md = _text_layer_fallback(record.artifacts)
    record.artifacts.extract_text.write_text(
        FALLBACK_SENTINEL.format(page=record.page_number),
        encoding="utf-8",
    )
    record.artifacts.format_markdown.write_text(format_md, encoding="utf-8")
    elapsed = time.monotonic() - record.page_started
    log.info(
        "  [%d/%d] page %d: done in %.1fs (%s, %s chars)",
        record.idx,
        record.total,
        record.page_number,
        elapsed,
        record.completion_label,
        f"{len(format_md):,}",
    )
    return PageResult(record.page_number, format_md)


__all__ = [
    "FallbackRecord",
    "_record_text_layer_fallback",
    "_text_layer_fallback",
]
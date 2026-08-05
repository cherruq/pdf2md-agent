"""Text-layer fallback helpers for the per-page pipeline.

When the vision model is unreachable after every retry, or when its output
fails CrewAI's task-output validation, the runner emits a fenced
text-layer Markdown stub instead of crashing the whole run. This module
isolates:

* :func:`_text_layer_fallback` — build the stub markdown from the PDF's
  native text layer for a single page.
* :func:`_record_text_layer_fallback` — write the stub into
  the page's format artifacts and return a :class:`PageResult`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from pdf2md_agent.cache import PageArtifacts
from pdf2md_agent.crew.types import FallbackRecord, PageRunContext
from pdf2md_agent.pdf_renderer import read_page_text

log = logging.getLogger("pdf2md_agent.runner")


def _text_layer_fallback(artifacts: PageArtifacts) -> str:
    """Build a best-effort markdown page from the PDF's native text layer.

    Used when the vision model is unreachable after all retries. The page's
    PNG is dropped from the output (we can't describe it) and the text is
    emitted verbatim in a fenced block so reviewers can spot drift.
    """
    text = read_page_text(artifacts.page_text).strip()
    if not text:
        return "*(vision model unavailable and PDF text layer is empty for this page — no content recovered)*"
    return (
        "*(vision model unavailable — falling back to PDF text layer; "
        "tables, figures, and layout are NOT preserved)*\n\n"
        "```\n"
        f"{text}\n"
        "```\n"
    )


def _record_text_layer_fallback(
    *,
    ctx: PageRunContext,
    artifacts: PageArtifacts,
    completion_label: str,
) -> str:
    """Write the fallback artifacts for one page and return the fallback markdown.

    Writes the fallback markdown to ``format.md``.
    """
    record = FallbackRecord(
        artifacts=artifacts,
        completion_label=completion_label,
        ctx=ctx,
    )
    format_md = _text_layer_fallback(record.artifacts)
    record.artifacts.format_markdown.write_text(format_md, encoding="utf-8")

    elapsed = time.monotonic() - record.ctx.page_started
    log.info(
        "  [%d/%d] page %d: done in %.1fs (%s, %s chars)",
        record.ctx.idx,
        record.ctx.total,
        record.ctx.page_number,
        elapsed,
        record.completion_label,
        f"{len(format_md):,}",
    )
    return format_md


__all__ = [
    "FallbackRecord",
    "_record_text_layer_fallback",
    "_text_layer_fallback",
]

"""Per-page result type shared across the crew package.

:class:`PageResult` is the single output unit of
:func:`pdf2md_agent.crew.runner.run_pipeline` and the single input unit of
:func:`pdf2md_agent.post_stream.stitch_pages`. Defining it here (rather
than inside the runner) breaks an import cycle between the runner and
the text-layer fallback helpers, both of which need to construct
``PageResult`` instances.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PageResult:
    """One page's final markdown + the running summary after this page."""

    page_number: int
    markdown: str
    summary: str


__all__ = ["PageResult"]
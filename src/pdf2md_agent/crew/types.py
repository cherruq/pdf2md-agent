"""Centralized data structures for the per-page extraction pipeline.

These structures represent the inputs, context, and outputs of the various
steps in the pipeline. They are ordered chronologically based on when they
are created and used in the processing lifecycle of a single page.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pdf2md_agent.cache import PageArtifacts


# 1. Pipeline Context
@dataclass(frozen=True, slots=True)
class PageRunContext:
    """Read-only global metadata for the current page being processed.
    
    This struct is threaded through the pipeline to provide logging context
    and progress tracking metadata without polluting individual function signatures.
    """

    page_number: int
    """The physical page number of the page."""
    idx: int
    """The 1-based index of the current page being processed in the pipeline batch."""
    total: int
    """The total number of pages in the current pipeline batch."""
    page_started: float
    """``time.monotonic()`` snapshot taken at the start of the page pipeline —
    used by the caller to log per-page elapsed time."""


# 2. Rendered PDF Page
@dataclass(frozen=True, slots=True)
class RenderedPage:
    """One rendered PDF page: raster image artifact and native text layer.
    
    This is the output of the PDF rendering stage and serves
    as the primary data input for the rest of the extraction pipeline.
    """

    width: int
    height: int
    image_path: Path
    ctx: PageRunContext
    text_path: Path | None = None
    text: str = ""


# 3. Image Preparation Output
@dataclass(frozen=True, slots=True)
class PreparedPage:
    """Inputs prepared for the extraction loop (Output of Step 1).

    This structure carries the original rendered page and the image that
    should actually be attached to the LLM (which may be downscaled or tiled).
    """

    page: RenderedPage
    """The original rendered PDF page."""
    text_hint_str: str
    """Text layer content or empty string."""
    attach_image_path: Path
    """Path the task should reference. Equal to ``page.image_path`` for
    pages that already fit the budget at their original size, or a downscaled
    cache path if resized."""
    is_tiled: bool
    """True if the page was split into ``tile_paths`` (extreme budget case)."""
    tile_paths: list[Path]
    """Half-overlap JPEG tiles; empty when ``is_tiled`` is False."""
    ctx: PageRunContext
    """The pipeline context for the current page."""


# 4. Extraction Step Output
@dataclass(frozen=True, slots=True)
class ExtractionOutcome:
    """Result of one page's extraction loop (Output of Step 2).

    This struct captures whether the extraction succeeded via the vision model,
    or if it fell back to the text-layer stub.
    """

    format_md: str
    """The final markdown content generated (either vision output or fallback stub)."""
    succeeded: bool
    """True if the vision extraction succeeded cleanly."""
    fell_back: bool
    """True if the vision model failed and we fell back to the text layer."""
    ctx: PageRunContext
    """The pipeline context for the current page."""


# 5. Fallback Execution Input
@dataclass(frozen=True, slots=True)
class FallbackRecord:
    """Arguments bundled for the text layer fallback handler.

    Bundling the fields keeps the call sites readable; the runner
    threads one of these through to the helper on every fallback path.
    """

    artifacts: PageArtifacts
    """The file paths for the cache artifacts of the current page."""
    completion_label: str
    """Label for logging (e.g., 'fallback', 'validation-fallback')."""
    ctx: PageRunContext
    """The pipeline context for the current page."""


# 6. Pipeline Output
@dataclass(frozen=True, slots=True)
class PageResult:
    """One page's final output (Output of Step 3).

    The ultimate result yielded by the runner back to the CLI layer.
    """

    page_number: int
    """The physical page number."""
    markdown: str
    """The final processed markdown content."""


__all__ = [
    "RenderedPage",
    "PageRunContext",
    "PreparedPage",
    "ExtractionOutcome",
    "FallbackRecord",
    "PageResult",
]

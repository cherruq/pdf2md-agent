"""Cross-page stitching post-processor.

The per-page extractor sees only one page at a time, so a sentence,
list item, or table row that gets split across a page boundary is
emitted as two fragments separated by an explicit ``\\n\\n---\\n\\n``
delimiter. That hard delimiter makes the split visible in the output
and is almost always wrong: the original document was one continuous
block.

``StreamingStitcher`` runs **after** the per-page pipeline and
re-stitches these fragments into a single continuous Markdown document.
It is a pure post-processor with no LLM dependency — the decision of
whether two fragments belong together is made by a small set of cheap
text heuristics (see :mod:`pdf2md_agent.post_stream_decision`):

* If the previous fragment ends without a sentence-terminating
  punctuation and the next fragment does not start a new block
  (heading, list, blockquote, table, code fence), they are joined.
* If the previous fragment ends with an unclosed markdown table row
  (odd number of ``|``, no trailing ``|``), the row is closed and the
  next fragment is appended; if the next fragment repeats the table
  header, that header is dropped.
* CJK characters are joined without an intervening space; Latin
  fragments get a single space.

The class is called "Streaming" because it buffers only the last
fragment of each page — the same idea as a streaming parser holding back
a token until it sees the next one to decide whether to emit it.
``feed()`` yields confirmed fragments and ``finalize()`` flushes any
remaining buffer at end of document. The top-level helper
:func:`stitch_pages` wraps the class for the common case.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from enum import Enum

from pdf2md_agent.crew.runner import PageResult
from pdf2md_agent.post_stream_decision import (
    _BLOCK_SEPARATOR,
    _decide_continuation,
    _is_unclosed_table_row,
    _last_meaningful_line,
    _smart_join,
    _split_into_blocks,
)
from pdf2md_agent.post_stream_table import _join_table_continuation


# --- Public API --------------------------------------------------------------


class StitchMode(str, Enum):
    """How aggressively to stitch per-page fragments together."""

    OFF = "off"
    """Pre-stitcher behavior: hard ``\\n\\n---\\n\\n`` between every page."""

    HEURISTIC = "heuristic"
    """Default. Pure-text heuristic, no extra LLM calls, no extra latency."""


def stitch_pages(
    pages: Iterable[PageResult],
    *,
    mode: StitchMode = StitchMode.HEURISTIC,
) -> str:
    """Concatenate per-page Markdown with optional cross-page stitching.

    Parameters
    ----------
    pages
        Per-page output from :func:`pdf2md_agent.crew.runner.run_pipeline`.
        Any iterable; only iterated once.
    mode
        See :class:`StitchMode`. ``HEURISTIC`` is the default and is
        sufficient for the vast majority of prose, list, and simple
        table splits.

    Returns
    -------
    str
        The full document as one Markdown string. In ``HEURISTIC`` mode
        there is no ``---`` separator between pages; in ``OFF`` mode the
        legacy ``\\n\\n---\\n\\n`` separator is preserved verbatim.
    """
    _LEGACY_SEPARATOR = "\n\n---\n\n"

    if mode == StitchMode.OFF:
        return _LEGACY_SEPARATOR.join(r.markdown for r in pages)

    page_list = list(pages)
    cleaned_mds = _clean_page_markdown(page_list)

    stitcher = StreamingStitcher()
    chunks: list[str] = []
    for md in cleaned_mds:
        chunks.extend(stitcher.feed(md))
    chunks.extend(stitcher.finalize())
    return _BLOCK_SEPARATOR.join(chunks)


def _strip_repeating_header_footer(text: str, compare_text: str) -> str:
    """Strip repeating running header or footer lines across adjacent pages."""
    if not text or not compare_text:
        return text

    lines = [ln.rstrip() for ln in text.splitlines()]
    comp_lines = [ln.strip() for ln in compare_text.splitlines() if ln.strip()]
    if not lines or not comp_lines:
        return text

    first_idx = next((k for k, ln in enumerate(lines) if ln.strip()), None)
    if first_idx is not None and len(lines) > 1:
        first_line = lines[first_idx].strip()
        if (
            len(first_line) >= 4
            and len(first_line) <= 120
            and not first_line.startswith(("|", "```", "~~~"))
            and first_line == comp_lines[0]
        ):
            lines[first_idx] = ""

    last_idx = next(
        (k for k in range(len(lines) - 1, -1, -1) if lines[k].strip()), None
    )
    if last_idx is not None and last_idx != first_idx and len(lines) > 1:
        last_line = lines[last_idx].strip()
        if (
            len(last_line) >= 4
            and len(last_line) <= 120
            and not last_line.startswith(("|", "```", "~~~"))
            and last_line == comp_lines[-1]
        ):
            lines[last_idx] = ""

    return "\n".join(lines).strip()


def _strip_standalone_page_numbers(text: str) -> str:
    """Strip leading and trailing standalone page number lines."""
    if not text:
        return text
    lines = text.splitlines()
    no_pat = re.compile(
        r"^\s*(?:[-–—\s]*\d+[-–—\s]*|page\s*\d+|第\s*\d+\s*[页张])\s*$",
        re.IGNORECASE,
    )
    while lines and no_pat.match(lines[0]):
        lines.pop(0)
    while lines and no_pat.match(lines[-1]):
        lines.pop(-1)
    return "\n".join(lines).strip()



def _clean_page_markdown(pages: list[PageResult]) -> list[str]:
    """Strip repeating running headers/footers and standalone page numbers across pages."""
    cleaned: list[str] = []
    n = len(pages)
    for i, r in enumerate(pages):
        text = r.markdown.strip()
        if not text:
            cleaned.append("")
            continue

        if n > 1:
            neighbors: list[str] = []
            if i > 0:
                neighbors.append(pages[i - 1].markdown)
            if i < n - 1:
                neighbors.append(pages[i + 1].markdown)
            for compare_text in neighbors:
                text = _strip_repeating_header_footer(text, compare_text)

        text = _strip_standalone_page_numbers(text)
        cleaned.append(text)
    return cleaned


class StreamingStitcher:
    """Buffer-and-flush stitcher with last-fragment lookahead.

    Usage::

        stitcher = StreamingStitcher()
        for page_md in pages:
            for chunk in stitcher.feed(page_md):
                write(chunk)
        for chunk in stitcher.finalize():
            write(chunk)
    """

    def __init__(self) -> None:
        self._buffer: str = ""

    def feed(self, page_md: str) -> Iterator[str]:
        """Yield confirmed fragments from one page; hold the last one back."""
        page_md = page_md.strip()
        if not page_md:
            return

        fragments = _split_into_blocks(page_md)
        if not fragments:
            return

        if self._buffer:
            decision = _decide_continuation(self._buffer, fragments[0])
            if decision.name == "CONTINUES":
                if _is_unclosed_table_row(_last_meaningful_line(self._buffer)):
                    self._buffer = _join_table_continuation(self._buffer, fragments[0])
                else:
                    self._buffer = _smart_join(self._buffer, fragments[0])
                fragments = fragments[1:]
                if not fragments:
                    # The joined content might continue on the next page;
                    # keep it buffered until the next feed() call.
                    return
                # More blocks follow on the same page — the join is
                # confirmed complete.  Yield it now so it isn't silently
                # overwritten by the last-fragment buffer below.
                yield self._buffer
                self._buffer = ""
            else:
                yield self._buffer
                self._buffer = ""

        if not fragments:
            return

        for frag in fragments[:-1]:
            yield frag

        self._buffer = fragments[-1]

    def finalize(self) -> Iterator[str]:
        """Flush any held fragment. Idempotent — second call yields nothing."""
        if self._buffer:
            yield self._buffer
            self._buffer = ""


# Step 3 entry point alias for the unified conversion pipeline
step3_stitch_and_clean = stitch_pages

__all__ = [
    "StitchMode",
    "StreamingStitcher",
    "step3_stitch_and_clean",
    "stitch_pages",
]
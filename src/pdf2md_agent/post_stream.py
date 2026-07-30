"""Cross-page stitching post-processor.

The per-page formatter sees only one page at a time, so a sentence,
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

    stitcher = StreamingStitcher()
    chunks: list[str] = []
    for r in pages:
        chunks.extend(stitcher.feed(r.markdown))
    chunks.extend(stitcher.finalize())
    return _BLOCK_SEPARATOR.join(chunks)


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


__all__ = ["StitchMode", "StreamingStitcher", "stitch_pages"]
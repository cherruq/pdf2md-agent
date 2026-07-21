"""Cross-page stitching post-processor.

The per-page Formatter (CrewAI agent) sees only one page at a time, so a
sentence or list item or table row that gets split across a page boundary
is emitted as two fragments separated by an explicit ``\\n\\n---\\n\\n``
delimiter. That hard delimiter makes the split visible in the output and
is almost always wrong: the original document was one continuous block.

``StreamingStitcher`` runs **after** the per-page pipeline and re-stitches
these fragments into a single continuous Markdown document. It is a pure
post-processor with no LLM dependency — the decision of whether two
fragments belong together is made by a small set of cheap text heuristics:

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
from dataclasses import dataclass
from enum import Enum

from pdf2md_agent.crew.runner import PageResult


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
    if mode == StitchMode.OFF:
        return _LEGACY_SEPARATOR.join(r.markdown for r in pages)

    stitcher = StreamingStitcher()
    chunks: list[str] = []
    for r in pages:
        chunks.extend(stitcher.feed(r.markdown))
    chunks.extend(stitcher.finalize())
    return _BLOCK_SEPARATOR.join(chunks)


# --- Streaming core ----------------------------------------------------------


_LEGACY_SEPARATOR = "\n\n---\n\n"
"""Pre-stitcher inter-page separator, kept for ``StitchMode.OFF`` parity."""
_BLOCK_SEPARATOR = "\n\n"
"""Blank-line separator between confirmed blocks in stitched output."""


@dataclass
class _Fragment:
    """One chunk the stitcher is holding or about to yield."""

    text: str
    is_table_row: bool = False
    """True if this fragment is *part of* a markdown table (not a new table)."""


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

    # ---- public ------------------------------------------------------------

    def feed(self, page_md: str) -> Iterator[str]:
        """Yield confirmed fragments from one page; hold the last one back."""
        page_md = page_md.strip()
        if not page_md:
            # Empty / whitespace-only page: nothing to yield, buffer unchanged.
            return

        fragments = _split_into_blocks(page_md)
        if not fragments:
            return

        # 1) Resolve buffer vs first fragment: merge or flush.
        if self._buffer:
            decision = _decide_continuation(self._buffer, fragments[0])
            if decision is _Decision.CONTINUES:
                if _is_unclosed_table_row(_last_meaningful_line(self._buffer)):
                    self._buffer = _join_table_continuation(self._buffer, fragments[0])
                else:
                    self._buffer = _smart_join(self._buffer, fragments[0])
                fragments = fragments[1:]
            else:
                yield self._buffer
                self._buffer = ""

        if not fragments:
            # The merge absorbed the only fragment — buffer already updated.
            return

        # 2) Yield every fragment except the last (held for next page).
        for frag in fragments[:-1]:
            yield frag

        # 3) Hold the last fragment.
        self._buffer = fragments[-1]

    def finalize(self) -> Iterator[str]:
        """Flush any held fragment. Idempotent — second call yields nothing."""
        if self._buffer:
            yield self._buffer
            self._buffer = ""


# --- Decision engine ---------------------------------------------------------


class _Decision(Enum):
    CONTINUES = "continues"
    NEW_BLOCK = "new_block"


_SENTENCE_END = re.compile(r"[。！？；?!.．)\]】」』\"'`]+\s*$")
"""Suffix characters that mark the end of a sentence / block in CJK + Latin."""

_BLOCK_START = re.compile(
    r"""^\s*(?:
        \#{1,6}\s           |  # ATX heading
        [-*+]\s            |  # bullet list (no indent or list-indent)
        \d+\.\s            |  # ordered list
        >\s                |  # blockquote
        ```                |  # fenced code block
        \|\s               |  # table row
        \s{4,}                 # 4+ leading spaces = indented code block
    )""",
    re.VERBOSE,
)
"""Prefix patterns that mark the start of a new block (must NOT merge)."""

_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
"""A markdown table separator row (e.g. ``|---|---|``)."""


def _split_into_blocks(page_md: str) -> list[str]:
    """Split a page's markdown into top-level blocks.

    A block is either:

    * A heading + its following paragraph (until a blank line).
    * A list (contiguous ``-``/``*``/``1.`` lines).
    * A table (contiguous lines starting with ``|``).
    * A standalone paragraph.
    * A code fence.

    Blank lines separate blocks.
    """
    # First normalize: split on blank-line boundaries.
    chunks = re.split(r"\n\s*\n", page_md)
    blocks: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip("\n")
        if chunk.strip():
            blocks.append(chunk)
    return blocks


def _last_meaningful_line(text: str) -> str:
    """Return the last non-empty line of ``text`` (used to inspect prev tail)."""
    lines = [ln for ln in text.rstrip().split("\n") if ln.strip()]
    return lines[-1] if lines else ""


def _first_meaningful_line(text: str) -> str:
    """Return the first non-empty line of ``text`` (used to inspect curr head)."""
    for ln in text.lstrip().split("\n"):
        if ln.strip():
            return ln
    return ""


def _decide_continuation(prev: str, curr: str) -> _Decision:
    """Decide whether ``curr`` continues the paragraph ``prev`` started.

    Heuristic table:

    ===========  =============  ============
    prev ends    curr starts    result
    ===========  =============  ============
    unfinished   continuation   CONTINUES
    complete     new_block      NEW_BLOCK
    unfinished   new_block      NEW_BLOCK (e.g. prev is a list item, curr is heading)
    complete     continuation   NEW_BLOCK (don't risk false merge)
    ===========  =============  ============

    Special cases override the table:

    * prev is a heading → always NEW_BLOCK
    * prev's last line is an unclosed table row → CONTINUES
      (with table-row semantics applied at join time)
    """
    prev_last = _last_meaningful_line(prev)
    curr_first = _first_meaningful_line(curr)

    # Headings always end the previous block — no merge.
    if prev_last.lstrip().startswith("#"):
        return _Decision.NEW_BLOCK

    # Unclosed markdown table row → always merge.
    if _is_unclosed_table_row(prev_last):
        return _Decision.CONTINUES

    prev_complete = bool(_SENTENCE_END.search(prev_last.rstrip()))
    curr_new_block = bool(_BLOCK_START.match(curr_first))

    if not prev_complete and not curr_new_block:
        return _Decision.CONTINUES
    return _Decision.NEW_BLOCK


def _is_unclosed_table_row(line: str) -> bool:
    """True if ``line`` starts a markdown table row but has no closing ``|``.

    Example unclosed: ``| col1 | col2 | col3``
    Example closed:   ``| col1 | col2 | col3 |``

    Pipe parity is not a reliable signal (it alternates with cell count);
    we rely on the trailing-``|`` check alone.
    """
    s = line.lstrip()
    if not s.startswith("|"):
        return False
    if s.rstrip().endswith("|"):
        return False
    return True


def _smart_join(prev: str, curr: str) -> str:
    """Join two fragments with appropriate separator.

    * CJK characters on either side: no separator (no word boundary).
    * Opening bracket ``( [ {`` at end of prev: no separator.
    * Otherwise: single space.
    """
    prev = prev.rstrip()
    curr = curr.lstrip()
    if not prev:
        return curr
    if not curr:
        return prev
    last = prev[-1]
    if last in "([{「『【《":
        return prev + curr
    if _is_cjk(last) or _is_cjk(curr[0]):
        return prev + curr
    return prev + " " + curr


def _is_cjk(ch: str) -> bool:
    """True if ``ch`` is a CJK ideograph."""
    if not ch:
        return False
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF          # CJK Unified Ideographs
        or 0x3400 <= o <= 0x4DBF       # CJK Extension A
        or 0x20000 <= o <= 0x2A6DF     # CJK Extension B
        or 0x2A700 <= o <= 0x2B73F     # CJK Extension C
        or 0x2B740 <= o <= 0x2B81F     # CJK Extension D
        or 0xF900 <= o <= 0xFAFF       # CJK Compatibility Ideographs
        or 0x2F800 <= o <= 0x2FA1F     # CJK Compatibility Supplement
    )


# --- Table-row helpers (for the dedup case) ---------------------------------


def _join_table_continuation(prev: str, curr: str) -> str:
    """Append ``curr`` to ``prev`` when ``prev`` ends mid-table-row.

    * Closes prev's unclosed final row by appending ``" |"``.
    * Strips a duplicate header + separator from ``curr`` if present.
    * Joins with a newline so each row stays on its own line.
    """
    prev_lines = prev.rstrip().split("\n")
    prev_lines[-1] = prev_lines[-1].rstrip() + " |"
    closed = "\n".join(prev_lines)
    deduped = _strip_redundant_table_header(curr)
    return closed + "\n" + deduped.lstrip()


def _strip_redundant_table_header(curr: str) -> str:
    """If ``curr`` begins with ``header / separator / rows...``, drop the
    redundant header + separator so it appends cleanly to the previous
    table.
    """
    lines = curr.split("\n")
    if len(lines) < 3:
        return curr
    if _TABLE_SEPARATOR.match(lines[1].strip()):
        return "\n".join(lines[2:])
    return curr


__all__ = [
    "StitchMode",
    "StreamingStitcher",
    "stitch_pages",
]
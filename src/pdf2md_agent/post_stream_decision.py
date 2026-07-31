"""Decision engine for the cross-page stitcher.

Given the buffered "previous fragment" and the next page's first block,
decide whether they belong together (and how to join them). Two
sub-concerns live here:

* :func:`_split_into_blocks` — chop a page's Markdown into top-level
  blocks separated by blank lines.
* :func:`_decide_continuation` — the CONTINUES-vs-NEW_BLOCK verdict.
* :func:`_smart_join` — the actual join, with CJK-aware spacing.
* :func:`_is_cjk` — the CJK Unified Ideographs + extensions predicate.

Table-row continuation lives in
:mod:`pdf2md_agent.post_stream_table` because it is a different
heuristic (close + dedup) than the prose smart-join.
"""

from __future__ import annotations

import re
from enum import Enum


# --- Constants shared with the table helpers ------------------------------

_BLOCK_SEPARATOR = "\n\n"
"""Blank-line separator between confirmed blocks in stitched output."""

_BLANK_LINE_RE = re.compile(r"\n\s*\n")
"""Blank-line boundary used by :func:`_split_into_blocks`; hoisted to
module scope so the pattern is compiled exactly once per process instead
of on every per-page call."""


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


# --- Block splitting --------------------------------------------------------


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
    chunks = _BLANK_LINE_RE.split(page_md)
    return [chunk.strip("\n") for chunk in chunks if chunk.strip()]


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


# --- Continuation decision -------------------------------------------------


class _Decision(Enum):
    CONTINUES = "continues"
    NEW_BLOCK = "new_block"


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

    if prev_last.lstrip().startswith("#"):
        return _Decision.NEW_BLOCK

    if _is_unclosed_table_row(prev_last):
        return _Decision.CONTINUES

    prev_complete = bool(_SENTENCE_END.search(prev_last.rstrip()))
    curr_new_block = bool(_BLOCK_START.match(curr_first))

    if not prev_complete and not curr_new_block:
        return _Decision.CONTINUES
    return _Decision.NEW_BLOCK


def _is_unclosed_table_row(line: str) -> bool:
    """True if ``line`` starts a markdown table row but has no closing ``|``.

    Pipe parity is not a reliable signal (it alternates with cell count);
    we rely on the trailing-``|`` check alone.
    """
    s = line.lstrip()
    if not s.startswith("|"):
        return False
    if s.rstrip().endswith("|"):
        return False
    return True


# --- Smart join -------------------------------------------------------------


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
        0x4E00 <= o <= 0x9FFF
        or 0x3400 <= o <= 0x4DBF
        or 0x20000 <= o <= 0x2A6DF
        or 0x2A700 <= o <= 0x2B73F
        or 0x2B740 <= o <= 0x2B81F
        or 0xF900 <= o <= 0xFAFF
        or 0x2F800 <= o <= 0x2FA1F
    )


__all__ = [
    "_BLOCK_SEPARATOR",
    "_Decision",
    "_decide_continuation",
    "_first_meaningful_line",
    "_is_cjk",
    "_is_unclosed_table_row",
    "_last_meaningful_line",
    "_smart_join",
    "_split_into_blocks",
]

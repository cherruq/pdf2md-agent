"""Table-row continuation helpers for the cross-page stitcher.

When a page boundary cuts a markdown table mid-row, the extractor for
the next page usually re-emits the table header + separator before the
continuing rows. Stitching those two fragments naively duplicates the
header. This module:

* :func:`_join_table_continuation` — closes the unclosed trailing row
  of ``prev`` and appends ``curr`` to it (deduping the header).
* :func:`_strip_redundant_table_header` — drops the redundant
  header + separator from ``curr`` so it appends cleanly.
"""

from __future__ import annotations

import re


_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
"""A markdown table separator row (e.g. ``|---|---|``)."""


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


__all__ = ["_join_table_continuation", "_strip_redundant_table_header"]

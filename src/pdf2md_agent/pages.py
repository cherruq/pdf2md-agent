"""Page-spec parsing and resolution for the --pages CLI flag.

Two pure functions, no I/O:

- :func:`parse_page_spec` is the argparse ``type=`` callable; it validates
  syntax and raises :class:`argparse.ArgumentTypeError` on bad input so
  the CLI rejects malformed specs before opening the PDF.

- :func:`resolve_pages` dedupes, sorts, and validates a parsed page list
  against the PDF's actual page count; raises :class:`ValueError` with a
  user-facing message on out-of-range pages.
"""
from __future__ import annotations

import argparse
import re

# Regexes (anchored, whitespace-tolerant). The parse step only enforces
# "positive integer" for tokens; the upper bound is checked later by
# resolve_pages against the actual PDF page count.
_TOKEN_RE = re.compile(r"^\s*(\d+)\s*$")
_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")

# DoS guard: cap the width of any single "N-M" range. Without this,
# ``--pages 1-999999999999`` would materialize a list of that many
# integers before we ever reach ``resolve_pages`` (which only sees a
# positive-integer list and can't tell it came from a huge range).
# 10_000 pages is roughly the largest real-world PDF a single user would
# process; beyond that, the spec is almost certainly adversarial or
# a typo (e.g. meant 1-999 and the trailing 9 was lost).
_MAX_RANGE_SPAN = 10_000


def _err(msg: str) -> argparse.ArgumentTypeError:
    return argparse.ArgumentTypeError(msg)


def parse_page_spec(spec: str) -> list[int]:
    """Parse a --pages value like ``'1-5,8,11-13'`` into ``[1,2,3,4,5,8,11,12,13]``.

    Comma-separated items, each either a single page number (``8``) or a
    range (``1-5``). Whitespace around numbers and around the ``-`` is
    tolerated.

    Page numbers must be positive integers (``>= 1``); the upper bound is
    not enforced here (that's :func:`resolve_pages`'s job, since it
    depends on the actual PDF).

    Raises :class:`argparse.ArgumentTypeError` on any malformed input.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise _err(f"expected integer or N-M, got {spec!r}")

    pages: list[int] = []
    seen: set[int] = set()
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            raise _err(f"expected integer or N-M, got {raw_item!r}")

        m_range = _RANGE_RE.match(item)
        if m_range:
            start = int(m_range.group(1))
            end = int(m_range.group(2))
            if start == 0 or end == 0:
                raise _err(f"page numbers must be >= 1, got {item!r}")
            if start > end:
                raise _err(f"range start must be <= end, got {item!r}")
            if end - start + 1 > _MAX_RANGE_SPAN:
                raise _err(
                    f"range {item!r} exceeds {_MAX_RANGE_SPAN} pages "
                    "(spec too wide; split into smaller ranges)"
                )
            for p in range(start, end + 1):
                if p not in seen:
                    seen.add(p)
                    pages.append(p)
            continue

        m_token = _TOKEN_RE.match(item)
        if m_token:
            n = int(m_token.group(1))
            if n == 0:
                raise _err(f"page numbers must be >= 1, got {item!r}")
            if n not in seen:
                seen.add(n)
                pages.append(n)
            continue

        raise _err(f"expected integer or N-M, got {item!r}")

    return pages


def resolve_pages(spec: list[int], total: int) -> list[int]:
    """Dedupe, sort, and validate a parsed page list against ``total``.

    Returns a new list of unique page numbers in ascending order, all
    within ``[1, total]``.

    Raises :class:`ValueError` on the first out-of-range page
    encountered, with a message of the form ``"page N out of range (PDF
    has M pages)"`` so the CLI can surface it directly.

    An empty result cannot arise: :func:`parse_page_spec` guarantees
    each item is a positive integer, and dedupe of a non-empty list is
    non-empty. No defensive empty-list check is needed.
    """
    if total < 1:
        raise ValueError(f"PDF has {total} pages; nothing to convert")

    out = sorted(set(spec))
    for n in out:
        if n < 1 or n > total:
            raise ValueError(f"page {n} out of range (PDF has {total} pages)")
    return out
"""Page-spec parsing for the --pages CLI flag.

Currently exposes :func:`parse_page_spec`, the ``argparse`` ``type=`` callable
that validates user input and rejects malformed specs before the PDF is opened.

A companion :func:`resolve_pages` will validate the parsed list against the
PDF's actual page count; that lives in the next task.
"""
from __future__ import annotations

import argparse
import re

# Regexes (anchored, whitespace-tolerant). The parse step only enforces
# "positive integer" for tokens; the upper bound is checked later by
# resolve_pages against the actual PDF page count.
_TOKEN_RE = re.compile(r"^\s*(\d+)\s*$")
_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")


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


def resolve_pages(pages: list[int], total: int) -> list[int]:
    """Validate parsed page list against the PDF page count. Pending implementation."""
    raise NotImplementedError(
        "resolve_pages() is not yet implemented; will be added in a follow-up"
    )
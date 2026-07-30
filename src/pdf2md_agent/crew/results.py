"""Per-page result type shared across the crew package."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PageResult:
    """One page's final markdown."""

    page_number: int
    markdown: str


__all__ = ["PageResult"]
"""Per-PDF intermediate-file cache: PNG pages, native text, and per-page outputs.

The cache layer provides:
* **Filesystem primitives** (:func:`atomic_write_text`) — sibling-tempfile
  + ``os.replace`` so a crash mid-write leaves the original file intact.
* **Layout** (:class:`CacheLayout`, :class:`PageArtifacts`,
  :func:`is_page_complete`, :func:`has_cached_extract`) — the on-disk
  paths and their trust-cache gates.
* **JSON state** (:func:`read_meta`, :func:`write_meta`,
  :func:`check_meta_matches`) — lightweight validation of the PDF file path.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TYPE_CHECKING

if TYPE_CHECKING:
    from pdf2md_agent.pdf_renderer import PageImage

log = logging.getLogger("pdf2md_agent.cache")

_ATOMIC_TMP_MODE: Final[int] = 0o600

FALLBACK_SENTINEL: Final[str] = (
    "(vision model unavailable for page {page}; text-layer fallback emitted; "
    "treat as sentinel — no extractor payload available)\n"
)
"""Marker written into ``extract.txt`` by the text-layer fallback."""

class CacheCorruptedError(Exception):
    """Raised when a cached file cannot be read or verified."""


# --- Filesystem primitives ---------------------------------------------------


def atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` via a sibling temp file + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _fd_unused, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(_fd_unused)
    tmp_path = Path(tmp_name)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(
        tmp_name,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | nofollow,
        _ATOMIC_TMP_MODE,
    )
    try:
        try:
            os.write(fd, content.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    try:
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# --- Layout ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PageArtifacts:
    """Files written for one page: source PNG, native text, agent outputs."""

    page_number: int
    page_png: Path
    page_text: Path
    extract_text: Path
    format_markdown: Path


@dataclass(frozen=True, slots=True)
class CacheLayout:
    """Directory layout for a PDF's intermediate cache."""

    root: Path
    pages_dir: Path
    meta_path: Path

    @classmethod
    def for_pdf(cls, root: Path, pdf_path: Path) -> "CacheLayout":
        root.mkdir(parents=True, exist_ok=True)
        pages = root / "pages"
        pages.mkdir(exist_ok=True)
        return cls(
            root=root,
            pages_dir=pages,
            meta_path=root / "meta.json",
        )

    def page_png_path(self, page_number: int) -> Path:
        return self.pages_dir / f"page_{page_number:04d}.png"

    def page_text_path(self, page_number: int) -> Path:
        return self.pages_dir / f"page_{page_number:04d}_text.txt"

    def page_extract_path(self, page_number: int) -> Path:
        return self.pages_dir / f"page_{page_number:04d}_extract.txt"

    def page_format_path(self, page_number: int) -> Path:
        return self.pages_dir / f"page_{page_number:04d}_format.md"

    def artifacts_for(self, page: PageImage) -> PageArtifacts:
        return PageArtifacts(
            page_number=page.page_number,
            page_png=self.page_png_path(page.page_number),
            page_text=self.page_text_path(page.page_number),
            extract_text=self.page_extract_path(page.page_number),
            format_markdown=self.page_format_path(page.page_number),
        )


# --- JSON state: meta fingerprint ------------------------------------------


def write_meta(
    meta_path: Path,
    *,
    pdf: Path,
    **_kwargs: Any,
) -> None:
    """Serialize run metadata (PDF path) to ``meta_path`` atomically."""
    canonical_pdf = pdf.resolve()
    atomic_write_text(
        meta_path,
        json.dumps(
            {"pdf": str(canonical_pdf)},
            indent=2,
            ensure_ascii=False,
        ),
    )


@dataclass(frozen=True, slots=True)
class MetaInfo:
    """The on-disk ``meta.json`` payload, parsed and frozen."""

    pdf: str


_META_REQUIRED_FIELDS: Final[tuple[str, ...]] = ("pdf",)


def read_meta(meta_path: Path) -> MetaInfo | None:
    """Return the parsed ``MetaInfo`` or ``None`` for missing/malformed input."""
    if not meta_path.exists():
        return None
    try:
        payload: Any = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if any(field not in payload for field in _META_REQUIRED_FIELDS):
        return None
    if not isinstance(payload["pdf"], str):
        return None
    return MetaInfo(pdf=payload["pdf"])


def check_meta_matches(
    stored: MetaInfo,
    *,
    pdf: str,
    **_kwargs: Any,
) -> list[str]:
    """Return a list of human-readable mismatch reasons; empty list == match."""
    reasons: list[str] = []
    if stored.pdf != pdf:
        reasons.append(
            f"pdf changed: cached={stored.pdf!r}, current={pdf!r}"
        )
    return reasons


# --- Trust-cache gates ------------------------------------------------------


def is_page_complete(layout: CacheLayout, page_number: int) -> bool:
    """True if the cached extract + format outputs already exist for this page."""
    return (
        layout.page_extract_path(page_number).exists()
        and layout.page_format_path(page_number).exists()
    )


@dataclass(frozen=True, slots=True)
class CacheNoCacheFlags:
    """Per-resource opt-out switches for the no-cache flag family."""

    render: bool = False
    text: bool = False
    resized: bool = False
    extract: bool = False
    format: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {
            "render": self.render,
            "text": self.text,
            "resized": self.resized,
            "extract": self.extract,
            "format": self.format,
        }

    def all(self) -> bool:
        return all(self.as_dict().values())


_FALLBACK_SENTINEL_PREFIX: Final[str] = "(vision model unavailable for page"


def has_cached_extract(layout: CacheLayout, page_number: int) -> bool:
    path = layout.page_extract_path(page_number)
    if not (path.is_file() and path.stat().st_size > 0):
        return False
    head = path.read_text(encoding="utf-8", errors="replace")[:128]
    if head.lstrip().startswith(_FALLBACK_SENTINEL_PREFIX):
        return False
    return True


__all__ = [
    "CacheCorruptedError",
    "CacheLayout",
    "CacheNoCacheFlags",
    "FALLBACK_SENTINEL",
    "MetaInfo",
    "PageArtifacts",
    "atomic_write_text",
    "check_meta_matches",
    "has_cached_extract",
    "is_page_complete",
    "read_meta",
    "write_meta",
]
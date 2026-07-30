"""Per-PDF intermediate-file cache: PNG pages, per-page agent outputs.

The cache layer is split into three cohesive concerns:

* **Filesystem primitives** (:func:`atomic_write_text`) — sibling-tempfile
  + ``os.replace`` so a crash mid-write leaves the original file intact.
  Both ``meta.json`` writes and the CLI's final output write go through
  this seam.
* **Layout** (:class:`CacheLayout`, :class:`PageArtifacts`,
  :func:`is_page_complete`, :func:`has_cached_extract`) — the on-disk
  paths and their trust-cache gates. ``has_cached_extract`` rejects both
  empty extracts (H1 sentinel) and the ``FALLBACK_SENTINEL`` marker
  written by the text-layer fallback.
* **JSON state** (:func:`read_meta`, :func:`write_meta`,
  :func:`check_meta_matches`) — the persisted state file (``meta.json``
  fingerprint). The reader is forgiving on missing input (returns a
  safe default) but loud on malformed input (returns mismatch reasons).
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
"""Marker written into ``extract.txt`` by the text-layer fallback.

Two readers depend on the exact prefix string:

* :func:`has_cached_extract` — refuses to trust an extract whose first
  characters match ``"(vision model unavailable for page"`` so a
  ``--no-cache-extract`` pass does not feed the marker text into the
  formatter.
* The fallback itself (:func:`pdf2md_agent.crew.fallback._record_text_layer_fallback`)
  which writes ``FALLBACK_SENTINEL.format(page=N)`` to the extract
  artifact so a follow-up run sees the sentinel and refuses to trust
  the file.

The prefix substring ``"(vision model unavailable for page"`` is also
hard-coded in :func:`has_cached_extract` — they must stay in sync.
"""

class CacheCorruptedError(Exception):
    """Raised when a cached file cannot be read or verified."""


# --- Filesystem primitives ---------------------------------------------------


def atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` via a sibling temp file + ``os.replace``.

    A crash mid-write leaves the original file (if any) intact instead of
    producing a truncated output. The temp file uses a randomized suffix and
    lives in the same directory as ``path`` so ``os.replace`` is atomic on
    POSIX and Windows alike.

    The temp file is opened with ``O_NOFOLLOW`` (when available) and mode
    ``0o600`` so a pre-existing symlink at the temp path cannot redirect the
    write to an attacker-controlled location.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a unique name; the fd from mkstemp is closed immediately
    # and we re-open with O_NOFOLLOW below so a symlink at tmp_name
    # cannot redirect the write.
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
    """Serialize run metadata to ``meta_path`` atomically.

    The schema is the fingerprint a follow-up run validates against
    via :func:`read_meta` + :func:`check_meta_matches`. Drift in any field
    means the cached outputs no longer correspond to the current pipeline
    configuration, so the runner fails loud instead of silently re-using
    stale data.

    ``pdf`` is canonicalized via :meth:`Path.resolve` before serialization
    so the on-disk fingerprint is always a stable realpath. Otherwise a
    follow-up run invoked from a different working directory (or with a
    different relative-path spelling of the same file) would see the cached
    value drift away from the current run's value purely due to path
    formatting, even though the underlying file is identical.

    Symlinks are followed by :meth:`Path.resolve`; a symlink PDF and its
    real-path target therefore canonicalize to the same stored value.
    """
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
    """The on-disk ``meta.json`` payload, parsed and frozen.

    Holding the fingerprint in a typed record keeps the match-check pure:
    the runner never re-parses JSON inside the hot loop, and tests can
    construct expected ``MetaInfo`` values without touching disk.
    """

    pdf: str


_META_REQUIRED_FIELDS: Final[tuple[str, ...]] = ("pdf",)


def read_meta(meta_path: Path) -> MetaInfo | None:
    """Return the parsed ``MetaInfo`` or ``None`` for missing/malformed input.

    Missing files, unreadable files, non-object JSON, or missing required
    fields all return ``None`` — the caller decides whether to fail loud
    (a follow-up run) or rebuild silently (the initial run). The fingerprint
    either matches or it doesn't, and a missing/malformed ``meta.json`` is
    a safe signal to rebuild from scratch.
    """
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
    """Return a list of human-readable mismatch reasons; empty list == match.

    The runner surfaces each reason in the validation error so a user
    knows exactly which fingerprint field drifted.
    """
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
    """Per-resource opt-out switches for the no-cache flag family.

    Default semantics: trust cached output (every flag ``False``). Setting a
    flag to ``True`` invalidates the corresponding cache resource for the
    duration of the run — the runner treats the resource as missing and
    rebuilds it from scratch.

    * ``render`` — skip the per-page PNG render when one already exists at
      the configured DPI. Setting it forces a re-render.
    * ``text`` — skip the per-page ``_text.txt`` re-emit when the cache
      file exists.
    * ``resized`` — skip the downscaled JPEG re-resize when the cache file
      already matches the budgeted ``long_side``.
    * ``extract`` — don't trust the cached ``extract.txt``; re-run the
      extractor (downstream format still trusts its own cache unless
      that flag is set too).
    * ``format`` — don't trust the cached ``format.md``; re-run the
      formatter. Short-circuits the entire per-page pipeline if the
      cached markdown is available and valid.
    """

    render: bool = False
    text: bool = False
    resized: bool = False
    extract: bool = False
    format: bool = False

    def as_dict(self) -> dict[str, bool]:
        """Return a dict of the underlying flags."""
        return {
            "render": self.render,
            "text": self.text,
            "resized": self.resized,
            "extract": self.extract,
            "format": self.format,
        }

    def all(self) -> bool:
        """True iff every per-resource flag is True (i.e. ``--no-cache-all``).

        ``--no-cache-all`` opts every cache resource out, which means
        downstream code can also discard stale derived state (such as
        ``meta.json``'s recorded fingerprint) rather than refusing the
        run on a drift that's about to be regenerated anyway.
        """
        return all(self.as_dict().values())



_FALLBACK_SENTINEL_PREFIX: Final[str] = "(vision model unavailable for page"


def has_cached_extract(layout: CacheLayout, page_number: int) -> bool:
    """True if a cached ``page_NNNN_extract.txt`` exists for this page
    AND its content is a real extractor payload (not a fallback sentinel).

    Two sentinel shapes must be rejected:

    * Zero-byte file — legacy H1 sentinel: vision model failed and the
      runner wrote an empty placeholder. Trusting it would propagate
      empty markdown into the formatter.
    * Non-empty fallback marker — the runner writes
      :data:`FALLBACK_SENTINEL` on text-layer fallback; trusting that as
      a real extract would feed the sentinel text into the
      ``--no-cache-extract`` formatter pass.
    """
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
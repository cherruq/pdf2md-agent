"""Per-PDF intermediate-file cache: PNG pages, per-page agent outputs, running summary."""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pdf2md_agent.pdf_renderer import PageImage


log = logging.getLogger("pdf2md_agent.cache")

_ATOMIC_TMP_MODE: Final[int] = 0o600


class CacheCorruptedError(RuntimeError):
    """Raised when a cache JSON file cannot be parsed or has the wrong shape.

    The offending file is preserved alongside its original location as
    ``<path>.corrupt-<unix-ts>`` so a human (or a follow-up re-run) can
    inspect what was on disk before the cache rebuilds from scratch.
    """


def atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` via a sibling temp file + ``os.replace``.

    A crash mid-write leaves the original file (if any) intact instead of
    producing a truncated output. The temp file uses a randomized suffix and
    lives in the same directory as ``path`` so ``os.replace`` is atomic on
    POSIX and Windows alike.

    The temp file is opened with ``O_NOFOLLOW`` (when available) and mode
    ``0o600`` so a pre-existing symlink at the temp path cannot redirect the
    write to an attacker-controlled location.

    Factored out of ``cli.py`` (D15-004/005) so both ``cache.py`` writes
    and the CLI's final output write share the same security +
    atomicity guarantees.
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
    summary_path: Path
    meta_path: Path

    @classmethod
    def for_pdf(cls, root: Path, pdf_path: Path) -> "CacheLayout":
        root.mkdir(parents=True, exist_ok=True)
        pages = root / "pages"
        pages.mkdir(exist_ok=True)
        return cls(
            root=root,
            pages_dir=pages,
            summary_path=root / "summary.json",
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


def _backup_corrupt_file(path: Path) -> Path | None:
    """Move ``path`` aside as ``<path>.corrupt-<unix-ts>`` and return the new path.

    Returns ``None`` if the source path doesn't exist (no-op) or if the
    rename itself failed — a backup is best-effort; the caller still
    raises :class:`CacheCorruptedError` so the rest of the recovery can
    proceed.
    """
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
    try:
        os.replace(path, backup)
    except OSError as exc:
        log.warning(
            "could not back up corrupt cache file %s -> %s: %s",
            path, backup, exc,
        )
        return None
    return backup


def write_meta(
    meta_path: Path,
    *,
    pdf: Path,
    dpi: int,
    with_summary: bool,
    pages: list[int] | None = None,
) -> None:
    """Serialize run metadata to ``meta_path`` atomically."""
    atomic_write_text(
        meta_path,
        json.dumps(
            {
                "pdf": str(pdf),
                "dpi": dpi,
                "with_summary": with_summary,
                "pages": pages,
            },
            indent=2,
            ensure_ascii=False,
        ),
    )


def read_summary(path: Path) -> str:
    """Read the running-summary payload from ``path``.

    Raises :class:`CacheCorruptedError` if the file exists but cannot be
    parsed as a JSON object — losing the running summary would silently
    drop cross-page context, so the corruption must surface (D6-008).
    On corruption the offending file is moved aside as
    ``<path>.corrupt-<unix-ts>`` so a follow-up re-run (or a human) can
    inspect it.

    A missing file still returns ``""`` — that is the legitimate "first
    run" state, not corruption.
    """
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        backup = _backup_corrupt_file(path)
        log.warning(
            "read_summary: %s is unreadable (%s); backed up to %s and raising",
            path, exc, backup if backup else "<backup failed>",
        )
        raise CacheCorruptedError(
            f"{path} is unreadable ({exc}); backed up to {backup}"
        ) from exc
    if not isinstance(payload, dict):
        backup = _backup_corrupt_file(path)
        log.warning(
            "read_summary: %s is not a JSON object; backed up to %s and raising",
            path, backup if backup else "<backup failed>",
        )
        raise CacheCorruptedError(
            f"{path} is not a JSON object; backed up to {backup}"
        )
    return str(payload.get("summary", ""))


def write_summary(path: Path, summary: str) -> None:
    """Serialize the running summary to ``path`` atomically.

    Logs + re-raises any failure so the caller surfaces the error
    instead of silently dropping the cross-page context. The temp file
    is cleaned up by :func:`atomic_write_text` itself on the error
    path.
    """
    payload = json.dumps({"summary": summary}, indent=2, ensure_ascii=False)
    try:
        atomic_write_text(path, payload)
    except OSError as exc:
        log.error("write_summary: failed to write %s: %s", path, exc)
        raise


def is_page_complete(layout: CacheLayout, page_number: int) -> bool:
    """True if the cached extract + format outputs already exist for this page."""
    return (
        layout.page_extract_path(page_number).exists()
        and layout.page_format_path(page_number).exists()
    )


def has_cached_extract(layout: CacheLayout, page_number: int) -> bool:
    """True if a cached ``page_NNNN_extract.txt`` exists for this page.

    Independent of ``format.md``: ``--reformat`` mode uses this to decide
    whether to skip the extractor for a given page.
    """
    return layout.page_extract_path(page_number).is_file()


__all__ = [
    "CacheCorruptedError",
    "CacheLayout",
    "PageArtifacts",
    "atomic_write_text",
    "has_cached_extract",
    "is_page_complete",
    "read_summary",
    "write_meta",
    "write_summary",
]
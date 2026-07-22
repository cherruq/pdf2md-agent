"""Per-PDF intermediate-file cache: PNG pages, per-page agent outputs, running summary."""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

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
    model: str,
    persona_version: str,
) -> None:
    """Serialize run metadata to ``meta_path`` atomically.

    The 6-field schema is the fingerprint a follow-up run validates against
    via :func:`read_meta` + :func:`check_meta_matches`. Drift in any field
    means the cached outputs no longer correspond to the current pipeline
    configuration, so the runner fails loud instead of silently re-using
    stale data.
    """
    atomic_write_text(
        meta_path,
        json.dumps(
            {
                "pdf": str(pdf),
                "dpi": dpi,
                "with_summary": with_summary,
                "pages": pages,
                "model": model,
                "persona_version": persona_version,
            },
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
    dpi: int
    with_summary: bool
    pages: tuple[int, ...] | None
    model: str
    persona_version: str


_META_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "pdf",
    "dpi",
    "with_summary",
    "pages",
    "model",
    "persona_version",
)


def read_meta(meta_path: Path) -> MetaInfo | None:
    """Return the parsed ``MetaInfo`` or ``None`` for missing/malformed input.

    Missing files, unreadable files, non-object JSON, or missing required
    fields all return ``None`` — the caller decides whether to fail loud
    (a follow-up run) or rebuild silently (the initial run). The corruption
    that ``read_summary`` guards against (silent loss of cross-page context)
    does not apply here: the fingerprint either matches or it doesn't, and
    a missing/malformed ``meta.json`` is a safe signal to rebuild from
    scratch.
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
    if not isinstance(payload["dpi"], int):
        return None
    if not isinstance(payload["with_summary"], bool):
        return None
    pages_raw = payload["pages"]
    if pages_raw is not None:
        if not isinstance(pages_raw, list) or not all(
            isinstance(p, int) for p in pages_raw
        ):
            return None
    if not isinstance(payload["model"], str):
        return None
    if not isinstance(payload["persona_version"], str):
        return None
    return MetaInfo(
        pdf=payload["pdf"],
        dpi=payload["dpi"],
        with_summary=payload["with_summary"],
        pages=tuple(pages_raw) if pages_raw is not None else None,
        model=payload["model"],
        persona_version=payload["persona_version"],
    )


def check_meta_matches(
    stored: MetaInfo,
    *,
    pdf: str,
    dpi: int,
    with_summary: bool,
    pages: list[int] | None,
    model: str,
    persona_version: str,
) -> list[str]:
    """Return a list of human-readable mismatch reasons; empty list == match.

    The runner surfaces each reason in the validation error so a user
    knows exactly which fingerprint field drifted. Page lists are compared
    as sets (order-invariant) — :func:`resolve_pages` may emit pages in
    user-supplied order, but the on-disk schema records the sorted, deduped
    set.
    """
    reasons: list[str] = []
    if stored.pdf != pdf:
        reasons.append(
            f"pdf changed: cached={stored.pdf!r}, current={pdf!r}"
        )
    if stored.dpi != dpi:
        reasons.append(
            f"dpi changed: cached={stored.dpi}, current={dpi}"
        )
    if stored.with_summary != with_summary:
        reasons.append(
            f"with_summary changed: cached={stored.with_summary}, "
            f"current={with_summary}"
        )
    stored_pages = set(stored.pages) if stored.pages is not None else None
    current_pages = set(pages) if pages is not None else None
    if stored_pages != current_pages:
        reasons.append(
            f"pages changed: cached={sorted(stored_pages) if stored_pages is not None else None}, "
            f"current={sorted(current_pages) if current_pages is not None else None}"
        )
    if stored.model != model:
        reasons.append(
            f"model changed: cached={stored.model!r}, current={model!r}"
        )
    if stored.persona_version != persona_version:
        reasons.append(
            f"persona_version changed: cached={stored.persona_version!r}, "
            f"current={persona_version!r}"
        )
    return reasons


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
    "MetaInfo",
    "PageArtifacts",
    "atomic_write_text",
    "check_meta_matches",
    "has_cached_extract",
    "is_page_complete",
    "read_meta",
    "read_summary",
    "write_meta",
    "write_summary",
]
"""Per-PDF intermediate-file cache: PNG pages, per-page agent outputs, running summary."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from convertpdf.pdf_renderer import PageImage


log = logging.getLogger("convertpdf.cache")


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


def write_meta(
    meta_path: Path,
    *,
    pdf: Path,
    dpi: int,
    with_summary: bool,
    pages: list[int] | None = None,
) -> None:
    meta_path.write_text(
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
        encoding="utf-8",
    )


def read_summary(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("read_summary: %s is unreadable (%s); treating as empty", path, exc)
        return ""
    if not isinstance(payload, dict):
        log.warning("read_summary: %s is not a JSON object; treating as empty", path)
        return ""
    return str(payload.get("summary", ""))


def write_summary(path: Path, summary: str) -> None:
    path.write_text(
        json.dumps({"summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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
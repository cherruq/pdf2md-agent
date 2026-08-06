"""每个 PDF 的中间文件缓存：PNG 页面、每页的智能体输出。"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

log = logging.getLogger("pdf2md_agent.cache")

_ATOMIC_TMP_MODE: Final[int] = 0o600


# --- Filesystem primitives ---------------------------------------------------


def atomic_write_text(path: Path, content: str) -> None:
    """通过同目录的临时文件加上 ``os.replace`` 原子的将 ``content`` 写入到 ``path``。"""
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
    """为单一页面写入的文件：源 PNG、原生文本和格式化 markdown。"""

    page_number: int
    page_png: Path
    page_text: Path
    format_markdown: Path


@dataclass(frozen=True, slots=True)
class CacheLayout:
    """PDF 的中间缓存目录布局。"""

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

    def page_format_path(self, page_number: int) -> Path:
        return self.pages_dir / f"page_{page_number:04d}_format.md"

    def artifacts_for(self, page_number: int) -> PageArtifacts:
        return PageArtifacts(
            page_number=page_number,
            page_png=self.page_png_path(page_number),
            page_text=self.page_text_path(page_number),
            format_markdown=self.page_format_path(page_number),
        )


# --- JSON state: meta fingerprint ------------------------------------------


def write_meta(
    meta_path: Path,
    *,
    pdf: Path,
    **_kwargs: Any,
) -> None:
    """原子性地将运行元数据序列化到 ``meta_path``。"""
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
    """磁盘上的 ``meta.json`` 负载，已解析并冻结。"""

    pdf: str


_META_REQUIRED_FIELDS: Final[tuple[str, ...]] = ("pdf",)


def read_meta(meta_path: Path) -> MetaInfo | None:
    """返回解析后的 ``MetaInfo``，若输入缺失或格式不正确则返回 ``None``。"""
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
    """返回一个不匹配原因的列表；空列表表示匹配。"""
    reasons: list[str] = []
    if stored.pdf != pdf:
        reasons.append(f"pdf changed: cached={stored.pdf!r}, current={pdf!r}")
    return reasons


# --- Trust-cache gates ------------------------------------------------------


def is_page_complete(layout: CacheLayout, page_number: int) -> bool:
    """如果此页面的已缓存格式化输出已存在，则为 True。"""
    return layout.page_format_path(page_number).exists()


@dataclass(frozen=True, slots=True)
class CacheNoCacheFlags:
    """用于无缓存（no-cache）标志系列的按资源选择退出（opt-out）开关。"""

    render: bool = False
    text: bool = False
    resized: bool = False
    format: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {
            "render": self.render,
            "text": self.text,
            "resized": self.resized,
            "format": self.format,
        }

    def all(self) -> bool:
        """当且仅当每个资源对应的标志都为 True 时，返回 True (即 ``--no-cache-all``)。"""
        return all(self.as_dict().values())


__all__ = [
    "CacheLayout",
    "CacheNoCacheFlags",
    "PageArtifacts",
    "atomic_write_text",
    "check_meta_matches",
    "is_page_complete",
    "read_meta",
    "write_meta",
]

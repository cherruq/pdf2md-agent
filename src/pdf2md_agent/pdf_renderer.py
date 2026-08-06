"""使用 PyMuPDF 将 PDF 渲染为按页的 PNG 图片 + 原生文本层。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image
import pymupdf

from pdf2md_agent.crew.types import PageRunContext, RenderedPage

if TYPE_CHECKING:
    from pdf2md_agent.config import ConversionConfig


def render_pdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = 144,
    prefix: str = "page",
    pages: list[int] | None = None,
) -> list[RenderedPage]:
    """在 ``output_dir`` 下将 ``pdf_path`` 渲染为每页对应的 PNG。

    如果 ``pages`` 为 ``None`` (默认)，会按文档顺序渲染每一页。如果 ``pages``
    是由从 1 开始的页码组成的列表，则仅渲染那些页面 (依然按文档顺序 —— 在内部
    会对列表排序) 并跳过其他页。输出的文件名总是采用 **原本** 从 1 开始
    的页码，因此在有着不同 ``pages`` 选项的跨次调用中缓存目录可以保持稳定。

    对于每一个已渲染的页面，也会同时写入位于同目录且带有该页 PDF
    原生文本层内容的 ``{prefix}_{NNNN}_text.txt`` (对于扫描版页面内容为空)。

    按文档顺序返回这些页面。由调用者负责保证 ``output_dir`` 的存在；
    函数负责往该目录内写入而不会进行创建。
    """
    doc = pymupdf.open(pdf_path)
    try:
        zoom = dpi / 72
        matrix = pymupdf.Matrix(zoom, zoom)
        page_numbers = list(range(1, doc.page_count + 1)) if pages is None else sorted(set(pages))
        pages_out: list[RenderedPage] = []
        total = len(page_numbers)
        for idx, page_number in enumerate(page_numbers, 1):
            ctx = PageRunContext(
                page_number=page_number,
                idx=idx,
                total=total,
                page_started=0.0,
            )
            png, text = _page_artifact_paths(output_dir, prefix, page_number)
            page = doc.load_page(page_number - 1)
            pages_out.append(_render_single_page(page, ctx, png, text, matrix))
        return pages_out
    finally:
        doc.close()


def _render_single_page(
    page: pymupdf.Page,
    ctx: PageRunContext,
    png_path: Path,
    txt_path: Path,
    matrix: pymupdf.Matrix,
    *,
    extracted_text: str | None = None,
) -> RenderedPage:
    """将单个 PyMuPDF 页面渲染为 PNG 并写入其原生文本层。"""
    if extracted_text is None:
        extracted_text = page.get_text("text", sort=True)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    pix.save(png_path)
    txt_path.write_text(extracted_text, encoding="utf-8")
    return RenderedPage(
        width=int(pix.width),
        height=int(pix.height),
        image_path=png_path,
        ctx=ctx,
        text_path=txt_path,
        text=extracted_text,
    )


def _page_artifact_paths(output_dir: Path, prefix: str, page_number: int) -> tuple[Path, Path]:
    """返回渲染单页面所需的 ``(png_path, text_path)``。

    单页面文件名中内嵌着从 1 开始的 ``page_number``，因此每一处调用
    都能生成全新的一对 ``Path``；设置该辅助函数是为了合并构建过程
    (用来匹配 :mod:`pdf2md_agent.cache` 中的缓存目录布局) 并且保证
    渲染循环的可读性。
    """
    stem = f"{prefix}_{page_number:04d}"
    return output_dir / f"{stem}.png", output_dir / f"{stem}_text.txt"


def read_page_text(text_path: Path) -> str:
    """读取由 :func:`render_pdf` 写入的单页面文本文件，并且能够安全的忽略 I/O 错误。"""
    try:
        if not text_path.exists():
            return ""
        return text_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _is_cached_png_valid(png_path: Path) -> bool:
    """返回 ``png_path`` 是否存在且非空，而不会抛出 I/O 错误。"""
    try:
        return png_path.is_file() and png_path.stat().st_size > 0
    except OSError:
        return False


def _is_cached_text_valid(txt_path: Path, expected_text: str) -> bool:
    """返回 ``txt_path`` 是否存在且与 ``expected_text`` 相匹配，而不会抛出 I/O 错误。"""
    try:
        if not txt_path.is_file():
            return False
        return txt_path.read_text(encoding="utf-8") == expected_text
    except (OSError, UnicodeDecodeError):
        return False


def pdf_page_count(pdf: Path) -> int:
    """通过 PyMuPDF 返回一个 PDF 文件的总页数。"""
    doc = pymupdf.open(pdf)
    try:
        return doc.page_count
    finally:
        doc.close()


def render_pages(config: ConversionConfig) -> list[RenderedPage]:
    """渲染该 PDF，在遭遇实时的文本内容漂移或启用无缓存标志时让步骤 2 (Step 2) 缓存失效。"""
    if config.no_cache.render or config.no_cache.text:
        return render_pdf(config.pdf, config.render_target, dpi=config.dpi, pages=config.resolved_pages)

    layout = config.layout
    with pymupdf.open(config.pdf) as doc:
        target_pages = (
            list(config.resolved_pages) if config.resolved_pages is not None else list(range(1, doc.page_count + 1))
        )
        total = len(target_pages)
        zoom = config.dpi / 72
        matrix = pymupdf.Matrix(zoom, zoom)
        pages: list[RenderedPage] = []

        for idx, n in enumerate(target_pages, 1):
            ctx = PageRunContext(
                page_number=n,
                idx=idx,
                total=total,
                page_started=0.0,
            )
            png = layout.page_png_path(n)
            txt = layout.page_text_path(n)
            page = doc.load_page(n - 1)

            need_png = not _is_cached_png_valid(png)
            new_text = page.get_text("text", sort=True)
            text_valid = _is_cached_text_valid(txt, new_text)

            if not text_valid:
                # 无效分支：尝试删后续资源（Step 2 产物与缩放 JPEG）
                layout.page_format_path(n).unlink(missing_ok=True)
                for jpg_path in config.render_target.glob(f"page_{n:04d}_*.jpg"):
                    jpg_path.unlink(missing_ok=True)

            if need_png or not text_valid:
                pages.append(_render_single_page(page, ctx, png, txt, matrix, extracted_text=new_text))
            else:
                # 有效分支：缓存直接可用
                with Image.open(png) as img:
                    pages.append(
                        RenderedPage(
                            width=img.width,
                            height=img.height,
                            image_path=png,
                            ctx=ctx,
                            text_path=txt,
                            text=new_text,
                        )
                    )
        return pages


__all__ = [
    "RenderedPage",
    "read_page_text",
    "render_pages",
    "render_pdf",
]

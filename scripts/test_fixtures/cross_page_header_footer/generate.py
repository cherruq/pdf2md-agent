#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pymupdf>=1.24,<2",
# ]
# ///
"""
Generate the cross-page header/footer test fixture.

Produces a paired Markdown source and a 3-page A4 PDF rendered by PyMuPDF.
The Markdown is the canonical source; the PDF is laid out from the same
string so the two artifacts can never drift. One unsplit Markdown
paragraph (the one labelled ``【P04-跨页验证】``) is intended to cross a
PDF page boundary naturally, providing a real cross-page test fixture
without any manual page breaks in the Markdown.

Required system font: AR PL UMing CN (or any CJK font that round-trips
Chinese + Latin through PyMuPDF text extraction). The font path is
auto-detected from a small allowlist; override with the
``FIXTURE_FONT_PATH`` env var if the font lives elsewhere.

Output: writes ``test_cross_page_header_footer.md`` and
``test_cross_page_header_footer.pdf`` next to this script. Re-running
this script deterministically produces the same two files.

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run tests/fixtures/cross_page_header_footer/generate.py
# 3. Or make executable and run:
#      chmod +x tests/fixtures/cross_page_header_footer/generate.py \
#        && ./tests/fixtures/cross_page_header_footer/generate.py
# 4. Override the font path if the default lookup fails:
#      FIXTURE_FONT_PATH=/path/to/uming.ttc \
#        uv run tests/fixtures/cross_page_header_footer/generate.py
# ──────────────────
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pymupdf

HERE: Final[Path] = Path(__file__).resolve().parent
DEFAULT_FONT_CANDIDATES: Final[tuple[str, ...]] = (
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
)
TITLE: Final[str] = "跨页连续段落页眉页脚测试"
HEADER: Final[str] = "跨页连续段落测试 · 固定页眉"
FOOTER: Final[str] = "pdf2md-agent 测试夹具"
TOTAL_PAGES: Final[int] = 3
PAGE_W_PT: Final[float] = 595.0
PAGE_H_PT: Final[float] = 842.0
BODY_X0: Final[float] = 60.0
BODY_X1: Final[float] = 535.0
BODY_Y0: Final[float] = 110.0
BODY_Y1: Final[float] = 760.0
BODY_FONT_SIZE: Final[float] = 16.0
TITLE_FONT_SIZE: Final[float] = 24.0
CHROME_FONT_SIZE: Final[float] = 9.0
LINE_HEIGHT: Final[float] = 28.0
PARAGRAPH_GAP: Final[float] = 18.0
CJK_FONT_NAME: Final[str] = "FixtureCJK"


@dataclass(frozen=True, slots=True)
class Paragraph:
    """A single Markdown paragraph with a stable serial marker."""

    marker: str
    body: str

    def render(self) -> str:
        return f"{self.marker}{self.body}"


PARAGRAPHS: Final[tuple[Paragraph, ...]] = (
    Paragraph(
        "【P01】",
        "清晨的阅览室刚开门，窗边长桌还留着夜间的凉意。"
        "管理员拉开百叶帘，让柔和天光落在纸面；"
        "读者翻到目录，先确认章节，再用铅笔记下今天要核对的问题。",
    ),
    Paragraph(
        "【P02】",
        "这份夹具不追求复杂版式，而是把可观察的线索放在明处。"
        "段首编号用于确认顺序，空行用于确认边界，"
        "固定的页眉和页脚提供参照，因此人工检查时不必猜测排版意图。",
    ),
    Paragraph(
        "【P03】",
        "午后的光线逐渐偏暖，纸张颜色却应保持稳定。"
        "提取时，正文中的逗号、分号和书名号都要原样保留，"
        "英文名称 pdf2md-agent 也不能被拆散；"
        "只有换页可以改变视觉位置，不能改变阅读次序。",
    ),
    Paragraph(
        "【P04-跨页验证】",
        "这一段专门承担跨页连续性测试。"
        "它从当前页面剩余的正文区域自然开始，"
        "随着宽松行距向下排列，"
        "不插入分页符，也不另设段落标题。"
        "读者在页底会看到叙述尚未结束："
        "记录员先核对段首标记、标点和相邻词语，"
        "再把视线移到下一页。"
        "翻页以后，句子应当无缝接续，"
        "既不重复页底已经出现的内容，"
        "也不遗漏边界附近的文字。"
        "为了让结果明确，段落继续描述同一条操作链："
        "先比较前后片段，再移除固定页眉、页脚和页码，"
        "随后合并换行并检查完整文本。"
        "若提取顺序、字符编码和段落归属都正确，"
        "最终应在下一页读到唯一收束句——"
        "跨页段落到此完整结束，蓝色书签仍夹在最后一行旁边。",
    ),
    Paragraph(
        "【P05】",
        "完成跨页核对后，"
        "可以观察段落之间的留白。"
        "每个编号只出现一次，"
        "相邻段落保持清楚间距；"
        "如果跨页条件被破坏，"
        "整段夹具就需要重新生成。"
        "为了让 P05 跨越 P2 与 P3 的边界，"
        "本段需要再补足一句过渡描述，"
        "让 P05 末尾出现在 P3 的开头位置。"
        "再补几句常规说明，"
        "确保 P05 真正跨越 P2 与 P3 的边界。"
        "还要讨论 P05 跨页带来的影响："
        "跨页行可能正好落在句中，"
        "需要确认 P05 的首尾字符连续。",
    ),
    Paragraph(
        "【P06】",
        "傍晚时，测试者把三页并排查看，"
        "再次确认页眉与页脚完全一致。",
    ),
    Paragraph(
        "【P07】",
        "最后再核对一次：标题 段首标记 页码 各自唯一。",
    ),
    Paragraph(
        "【P08】",
        "核对完成后，读者合上文档，"
        "把铅笔和蓝色书签放回桌角。"
        "这份简短夹具到此结束。",
    ),
)


def render_markdown() -> str:
    parts: list[str] = [f"# {TITLE}", ""]
    for index, paragraph in enumerate(PARAGRAPHS):
        if index > 0:
            parts.append("")
        parts.append(paragraph.render())
    return "\n".join(parts) + "\n"


def resolve_font_path() -> Path:
    override = os.environ.get("FIXTURE_FONT_PATH")
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(
                f"FIXTURE_FONT_PATH={override} does not exist or is not a file"
            )
        return path
    for candidate in DEFAULT_FONT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No CJK font found. Tried: "
        + ", ".join(DEFAULT_FONT_CANDIDATES)
        + ". Set FIXTURE_FONT_PATH to a CJK-capable TTF/TTC file."
    )


def draw_header(page: pymupdf.Page) -> None:
    page.insert_textbox(
        pymupdf.Rect(BODY_X0, 60, BODY_X1, 80),
        HEADER,
        fontname=CJK_FONT_NAME,
        fontsize=CHROME_FONT_SIZE,
        align=pymupdf.TEXT_ALIGN_CENTER,
        color=(0.27, 0.40, 0.70),
    )
    page.draw_line(
        pymupdf.Point(BODY_X0, 85),
        pymupdf.Point(BODY_X1, 85),
        color=(0.27, 0.40, 0.70),
        width=0.6,
    )


def draw_footer(page: pymupdf.Page, page_number: int) -> None:
    page.insert_textbox(
        pymupdf.Rect(BODY_X0, 805, BODY_X0 + 200, 825),
        FOOTER,
        fontname=CJK_FONT_NAME,
        fontsize=CHROME_FONT_SIZE,
        align=pymupdf.TEXT_ALIGN_LEFT,
        color=(0.40, 0.40, 0.40),
    )
    label = f"第 {page_number} / {TOTAL_PAGES} 页"
    page.insert_textbox(
        pymupdf.Rect(BODY_X1 - 200, 805, BODY_X1, 825),
        label,
        fontname=CJK_FONT_NAME,
        fontsize=CHROME_FONT_SIZE,
        align=pymupdf.TEXT_ALIGN_RIGHT,
        color=(0.40, 0.40, 0.40),
    )


def render_pdf() -> bytes:
    font_path = resolve_font_path()
    document = pymupdf.open()
    document.set_metadata(
        {
            "title": TITLE,
            "author": "pdf2md-agent test fixture",
            "subject": "Cross-page paragraph + header + footer fixture",
            "creator": "scripts/test_fixtures/cross_page_header_footer/generate.py",
        }
    )

    page = document.new_page(width=PAGE_W_PT, height=PAGE_H_PT)
    page.insert_font(fontname=CJK_FONT_NAME, fontfile=str(font_path))
    draw_header(page)
    draw_footer(page, 1)
    y_cursor = BODY_Y0
    page_index = 1

    def new_page() -> pymupdf.Page:
        nonlocal page, y_cursor, page_index
        page = document.new_page(width=PAGE_W_PT, height=PAGE_H_PT)
        page.insert_font(fontname=CJK_FONT_NAME, fontfile=str(font_path))
        page_index += 1
        draw_header(page)
        draw_footer(page, page_index)
        y_cursor = BODY_Y0
        return page

    def draw_title(text: str, size: float) -> None:
        nonlocal y_cursor
        page.insert_textbox(
            pymupdf.Rect(BODY_X0, y_cursor, BODY_X1, y_cursor + size * 1.5),
            text,
            fontname=CJK_FONT_NAME,
            fontsize=size,
            color=(0.12, 0.23, 0.47),
        )
        y_cursor += size * 1.6 + 8

    def draw_paragraph(text: str, size: float) -> None:
        nonlocal y_cursor
        chars_per_line = 17
        line_height = size * 1.7
        while text:
            available = BODY_Y1 - y_cursor
            max_lines = max(1, int(available / line_height))
            if max_lines <= 0:
                new_page()
                continue
            chunk_chars = max_lines * chars_per_line
            if len(text) <= chunk_chars:
                page.insert_textbox(
                    pymupdf.Rect(BODY_X0, y_cursor, BODY_X1, BODY_Y1),
                    text,
                    fontname=CJK_FONT_NAME,
                    fontsize=size,
                    color=(0, 0, 0),
                )
                line_count = max(1, (len(text) + chars_per_line - 1) // chars_per_line)
                y_cursor += line_count * line_height + 6
                return
            head = text[:chunk_chars]
            cut = head.rfind("，")
            if cut < chunk_chars * 0.5:
                cut = head.rfind("；")
            if cut < chunk_chars * 0.5:
                cut = head.rfind("。")
            if cut < chunk_chars * 0.5:
                cut = chunk_chars
            head = head[: cut + 1]
            head_lines = max(1, (len(head) + chars_per_line - 1) // chars_per_line)
            if head_lines < 3 or cut >= chunk_chars - 3:
                new_page()
                page.insert_textbox(
                    pymupdf.Rect(BODY_X0, y_cursor, BODY_X1, BODY_Y1),
                    text,
                    fontname=CJK_FONT_NAME,
                    fontsize=size,
                    color=(0, 0, 0),
                )
                line_count = max(1, (len(text) + chars_per_line - 1) // chars_per_line)
                y_cursor += line_count * line_height + 6
                return
            page.insert_textbox(
                pymupdf.Rect(BODY_X0, y_cursor, BODY_X1, BODY_Y1),
                head,
                fontname=CJK_FONT_NAME,
                fontsize=size,
                color=(0, 0, 0),
            )
            y_cursor = BODY_Y1 + 1
            new_page()
            text = text[cut + 1 :]
            if not text:
                return

    draw_title(TITLE, 22.0)
    for paragraph in PARAGRAPHS:
        draw_paragraph(paragraph.render(), 14.0)

    document.subset_fonts()
    document.scrub()
    return document.tobytes(garbage=4)


def main() -> int:
    markdown = render_markdown()
    markdown_path = HERE / "test_cross_page_header_footer.md"
    pdf_path = HERE / "test_cross_page_header_footer.pdf"
    markdown_path.write_text(markdown, encoding="utf-8")
    pdf_path.write_bytes(render_pdf())
    print(f"wrote {markdown_path} ({markdown_path.stat().st_size} bytes)")
    print(f"wrote {pdf_path} ({pdf_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

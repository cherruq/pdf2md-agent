#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pymupdf>=1.24,<2",
# ]
# ///
"""
Verify the cross-page header/footer test fixture.

Reads the paired Markdown and PDF in the same directory and asserts:

* the PDF is exactly 3 A4 pages;
* every page shows the fixed header, footer, and ``第 X / 3 页`` numbering;
* all 8 paragraph markers (``【P01】`` ... ``【P08】``) appear once and
  in order in the concatenated extracted text;
* the paragraph ``【P04-跨页验证】`` is split across two adjacent pages
  with a verifiable opening tail on page 1 and a unique closing fragment
  on page 2;
* the PDF body equals the Markdown body verbatim once the title, header,
  footer, and page numbers are stripped and whitespace is normalised.

Exits with a non-zero status (and a diagnostic message on stderr) on any
mismatch so the script can drive CI gates.

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run tests/fixtures/cross_page_header_footer/verify.py
# ──────────────────
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
MD_PATH = HERE / "test_cross_page_header_footer.md"
PDF_PATH = HERE / "test_cross_page_header_footer.pdf"
HEADER = "跨页连续段落测试 · 固定页眉"
FOOTER = "pdf2md-agent 测试夹具"
TITLE = "跨页连续段落页眉页脚测试"
P04_TAIL = "翻页以后，句子应当无缝接续，"
P04_CONTINUATION = "既不重复页底已经出现的内容"
P04_CLOSING_LINES = (
    "完整结束，",
)
EXPECTED_MARKERS = (
    "【P01】",
    "【P02】",
    "【P03】",
    "【P04-跨页验证】",
    "【P05】",
    "【P06】",
    "【P07】",
    "【P08】",
)


def body_of(page_text: str) -> str:
    lines = [line for line in page_text.splitlines() if line.strip()]
    if lines and HEADER in lines[0]:
        lines = lines[1:]
    while lines and (FOOTER in lines[0] or lines[0].startswith("第 ") and " / 3 页" in lines[0]):
        lines = lines[1:]
    return "".join(lines)


def fail(message: str) -> None:
    print(f"verify_cross_page_fixture: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    if not MD_PATH.is_file():
        fail(f"missing Markdown source: {MD_PATH}")
    if not PDF_PATH.is_file():
        fail(f"missing PDF artifact: {PDF_PATH}")
    markdown = MD_PATH.read_text(encoding="utf-8")
    document = pymupdf.open(PDF_PATH)
    page_texts = [page.get_text("text") for page in document]

    if document.page_count != 3:
        fail(f"expected 3 pages, got {document.page_count}")

    for index, text in enumerate(page_texts, start=1):
        if HEADER not in text:
            fail(f"page {index} missing header {HEADER!r}")
        if FOOTER not in text:
            fail(f"page {index} missing footer {FOOTER!r}")
        page_number = f"第 {index} / 3 页"
        if page_number not in text:
            fail(f"page {index} missing page number {page_number!r}")

    joined = "".join(page_texts)
    for marker in EXPECTED_MARKERS:
        if joined.count(marker) != 1:
            fail(f"marker {marker!r} should appear exactly once, got {joined.count(marker)}")
    for earlier, later in zip(EXPECTED_MARKERS, EXPECTED_MARKERS[1:]):
        if joined.index(earlier) > joined.index(later):
            fail(f"marker order broken: {earlier!r} should precede {later!r}")

    page1_body = body_of(page_texts[0])
    page2_body = body_of(page_texts[1])
    page3_body = body_of(page_texts[2]) if document.page_count > 2 else ""
    if "【P04-跨页验证】" not in page1_body:
        fail("paragraph P04 must start on page 1")
    if "【P04-跨页验证】" in page2_body or "【P04-跨页验证】" in page3_body:
        fail("paragraph P04 must not be repeated on page 2 or 3")
    if not page1_body.rstrip().endswith(P04_TAIL):
        fail(f"page 1 must end with the P04 opening tail {P04_TAIL!r}")
    if not page2_body.lstrip().startswith(P04_CONTINUATION):
        fail(f"page 2 must start with the P04 continuation {P04_CONTINUATION!r}")
    for line in P04_CLOSING_LINES:
        if line not in page2_body:
            fail(f"page 2 must contain the P04 closing line {line!r}")
        if line in page1_body or line in page3_body:
            fail(f"P04 closing line {line!r} leaked to another page")

    numbers = [f"第 {i} / 3 页" for i in range(1, 4)]
    cleaned = [
        "".join(
            body_of(text)
            .replace(HEADER, "")
            .replace(FOOTER, "")
            .replace(number, "")
            .replace(TITLE, "")
            .split()
        )
        for text, number in zip(page_texts, numbers, strict=True)
    ]
    expected = "".join(markdown.split("\n\n", 1)[1].split())
    actual = "".join(cleaned)
    if actual != expected:
        fail(
            "PDF body diverges from Markdown body "
            f"(expected_len={len(expected)}, actual_len={len(actual)})"
        )

    summary = {
        "markdown_bytes": MD_PATH.stat().st_size,
        "pdf_bytes": PDF_PATH.stat().st_size,
        "paragraphs": len(markdown.strip().split("\n\n")) - 1,
        "pages": document.page_count,
        "page_sizes_pt": [
            [round(page.rect.width, 3), round(page.rect.height, 3)]
            for page in document
        ],
        "markers_once_in_order": True,
        "p04_starts_on_page": 1,
        "p04_finishes_on_page": 2,
        "p04_opening_tail_on_page1": P04_TAIL,
        "p04_continuation_on_page2": P04_CONTINUATION,
        "p04_closing_wrapped_lines_on_page2": list(P04_CLOSING_LINES),
        "paired_body_exact_match": True,
        "raster_images_per_page": [len(page.get_images(full=True)) for page in document],
        "body_chars_per_page": [len(body_of(text)) for text in page_texts],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

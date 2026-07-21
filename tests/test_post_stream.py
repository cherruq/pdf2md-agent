"""Tests for StreamingStitcher — cross-page paragraph/list/table stitching.

The stitcher is a pure-function post-processor that runs *after* the
per-page Formatter. It produces a continuous Markdown document by
buffering the last paragraph of each page until it can decide whether
the next page continues it (like a streaming parser holding back the
last token until it sees the next one).

These tests cover the heuristic-only path (no LLM calls).
"""
from __future__ import annotations

import re

from pdf2md_agent.crew.runner import PageResult
from pdf2md_agent.post_stream import StitchMode, stitch_pages, StreamingStitcher


# ---------- helpers ----------------------------------------------------------

def _page(n: int, markdown: str) -> PageResult:
    return PageResult(page_number=n, markdown=markdown, summary="")


def _drain(stitcher: StreamingStitcher, pages: list[str]) -> list[str]:
    """Feed all pages, then finalize. Return list of yielded chunks."""
    out: list[str] = []
    for p in pages:
        out.extend(stitcher.feed(p))
    out.extend(stitcher.finalize())
    return out


# ---------- mode: OFF preserves current behavior ----------------------------


def test_mode_off_uses_dash_separator() -> None:
    """StitchMode.OFF must reproduce the pre-stitcher join behavior."""
    pages = [_page(1, "para one."), _page(2, "para two.")]
    out = stitch_pages(pages, mode=StitchMode.OFF)
    assert out == "para one.\n\n---\n\npara two."


def test_mode_off_keeps_three_dashes_in_body() -> None:
    """Pre-existing `---` in body should not confuse the join."""
    pages = [_page(1, "above\n\n---\n\nbelow"), _page(2, "next")]
    out = stitch_pages(pages, mode=StitchMode.OFF)
    assert out == "above\n\n---\n\nbelow\n\n---\n\nnext"


# ---------- paragraph stitching: the canonical GB/T case ---------------------


def test_split_paragraph_chinese_sentence() -> None:
    """GB/T 27930 A.3.6.1 case: page 1 ends mid-sentence, page 2 continues."""
    p1 = (
        "## A.3.6.1 充电阶段检测\n\n"
        "车辆接口连接完成后，电动汽车应控制断开开关S3，电动汽车应通过检测点3"
    )
    p2 = "电压状态来识别车辆接口连接状态与可充电状态，状态定义按表A.1。"
    pages = [_page(1, p1), _page(2, p2)]
    out = stitch_pages(pages)
    # The two fragments must merge into one paragraph, no `---` between them.
    assert "---" not in out
    assert "通过检测点3电压状态" in out  # CJK-CJK merge, no space
    assert "通过检测点3 电压状态" not in out  # NOT Latin-style with space


def test_split_paragraph_english_sentence() -> None:
    """English mid-sentence wrap: 'shall be ' / 'connected to ...'"""
    p1 = "Before charging, the connector shall be"
    p2 = "connected to the vehicle inlet and locked."
    out = stitch_pages([_page(1, p1), _page(2, p2)])
    assert "shall be connected to" in out
    assert "---" not in out


def test_terminated_paragraphs_do_not_merge() -> None:
    """Two pages each ending with a sentence terminator must NOT merge."""
    p1 = "First paragraph ends here."
    p2 = "Second paragraph starts fresh."
    out = stitch_pages([_page(1, p1), _page(2, p2)])
    assert "---" not in out
    # They should still be separate paragraphs (blank line between)
    assert out == "First paragraph ends here.\n\nSecond paragraph starts fresh."


def test_heading_terminates_paragraph() -> None:
    """A heading on page 1 must NOT merge with page 2's text."""
    p1 = "Some body text.\n\n# A.4 New Section"
    p2 = "This belongs to section A.4."
    out = stitch_pages([_page(1, p1), _page(2, p2)])
    assert "---" not in out
    # Page 2 text should start a new paragraph, not continue the heading
    assert "# A.4 New Section\n\nThis belongs to section A.4." in out


# ---------- list stitching ---------------------------------------------------


def test_split_list_item_merges() -> None:
    """List item wrapping across pages must merge into one item."""
    p1 = "- First item that is quite long and gets cut"
    p2 = "off at the page boundary.\n- Second item."
    out = stitch_pages([_page(1, p1), _page(2, p2)])
    assert "\n\n---\n\n" not in out
    assert "- First item that is quite long and gets cut off at the page boundary." in out
    # The second item should remain a separate bullet
    assert "- Second item." in out


def test_new_top_level_list_does_not_merge() -> None:
    """Page 1 ends mid-paragraph, page 2 starts a fresh top-level list."""
    p1 = "Intro paragraph not terminated"
    p2 = "- First list item\n- Second list item"
    out = stitch_pages([_page(1, p1), _page(2, p2)])
    assert "---" not in out
    # The intro should merge with the first list item text? No — list items
    # are new blocks. The intro should NOT silently attach.
    # Per heuristic: prev unfinished + curr is new_block (starts with `- `)
    # → return False (flush, do NOT merge)
    assert "Intro paragraph not terminated\n\n- First list item" in out


# ---------- table stitching --------------------------------------------------


def test_split_table_row_unclosed() -> None:
    """Table row cut mid-row on page 1 should be closed and merged."""
    p1 = (
        "| Header A | Header B | Header C |\n"
        "|---|---|---|\n"
        "| 1 | 2 | 3 |\n"
        "| 4 | 5 | 6"  # unclosed: no trailing |
    )
    p2 = (
        "| 7 | 8 | 9 |\n"
        "| 10 | 11 | 12 |"
    )
    out = stitch_pages([_page(1, p1), _page(2, p2)])
    assert "\n\n---\n\n" not in out
    # The previously-unclosed row must be closed with " |"
    # and the new rows appended
    lines = out.split("\n")
    assert "| 4 | 5 | 6 |" in lines  # closed by stitcher
    assert "| 7 | 8 | 9 |" in lines
    assert "| 10 | 11 | 12 |" in lines


def test_split_table_with_duplicate_header_drops_header() -> None:
    """When page 2 includes a redundant table header, drop it."""
    p1 = (
        "| A | B |\n"
        "|---|---|\n"
        "| 1 | 2"
    )  # unclosed
    p2 = (
        "| A | B |\n"           # duplicate header
        "|---|---|\n"           # duplicate separator
        "| 3 | 4 |"             # new row
    )
    out = stitch_pages([_page(1, p1), _page(2, p2)])
    assert "\n\n---\n\n" not in out
    # Count header/separator rows flexibly (allow optional trailing `|`).
    sep_re = re.compile(r"^\|?-+(\|\s*-+)+\|?$")
    header_lines = [line for line in out.split("\n") if line.strip() == "| A | B |"]
    sep_lines = [line for line in out.split("\n") if sep_re.match(line.strip())]
    assert len(header_lines) == 1
    assert len(sep_lines) == 1
    assert "| 3 | 4 |" in out


def test_split_table_no_header_repeats() -> None:
    """Page 2 starts directly with body rows (no duplicate header) — keep all."""
    p1 = (
        "| A | B |\n"
        "|---|---|\n"
        "| 1 | 2"
    )
    p2 = "| 3 | 4 |\n| 5 | 6 |"
    out = stitch_pages([_page(1, p1), _page(2, p2)])
    assert "\n\n---\n\n" not in out
    assert "| 3 | 4 |" in out
    assert "| 5 | 6 |" in out


# ---------- empty / single-paragraph pages -----------------------------------


def test_empty_page_does_not_break_buffer() -> None:
    """An empty/whitespace page should not corrupt the held buffer."""
    p1 = "Unfinished paragraph text"
    p2 = ""  # empty page (e.g., blank page)
    p3 = "continues here."
    out = stitch_pages([_page(1, p1), _page(2, p2), _page(3, p3)])
    assert "---" not in out
    assert "Unfinished paragraph textcontinues here." in out or \
           "Unfinished paragraph text continues here." in out


def test_whitespace_only_page_ignored() -> None:
    """A page with only whitespace should be ignored, not break the buffer."""
    p1 = "Sentence fragment"
    p2 = "   \n\n  \t  \n"
    p3 = "rest of sentence."
    out = stitch_pages([_page(1, p1), _page(2, p2), _page(3, p3)])
    assert "---" not in out
    assert "Sentence fragment" in out
    assert "rest of sentence." in out


def test_single_paragraph_page_holds_buffer() -> None:
    """If a page is one paragraph, the buffer must wait for the next page."""
    pages = [
        _page(1, "intro\n\nOnly paragraph."),
        _page(2, "Just one paragraph here."),
        _page(3, "Final paragraph."),
    ]
    out = stitch_pages(pages)
    assert "\n\n---\n\n" not in out
    # 4 paragraphs separated by 3 blank lines
    assert out.count("\n\n") == 3


# ---------- finalize semantics ----------------------------------------------


def test_finalize_flushes_remaining_buffer() -> None:
    """If the document ends mid-paragraph, finalize() must yield it."""
    stitcher = StreamingStitcher()
    out = list(stitcher.feed("starts here but never"))
    # Nothing yielded yet — buffer holds the fragment
    assert out == []
    final = list(stitcher.finalize())
    assert final == ["starts here but never"]


def test_finalize_is_idempotent() -> None:
    """Calling finalize() twice must not double-yield."""
    stitcher = StreamingStitcher()
    list(stitcher.feed("buffered text"))
    assert list(stitcher.finalize()) == ["buffered text"]
    assert list(stitcher.finalize()) == []


def test_finalize_no_buffer_no_op() -> None:
    """If only empty pages were fed, finalize() yields nothing."""
    stitcher = StreamingStitcher()
    list(stitcher.feed(""))  # empty page
    assert list(stitcher.finalize()) == []


# ---------- separator handling ----------------------------------------------


def test_no_double_blank_lines_between_merged_paragraphs() -> None:
    """After merge, no `\n\n\n` should appear (no triple blank lines)."""
    p1 = "First part"
    p2 = "second part."
    out = stitch_pages([_page(1, p1), _page(2, p2)])
    assert "\n\n\n" not in out


def test_separator_no_longer_emitted_by_default() -> None:
    """The default heuristic mode must NOT emit `\\n\\n---\\n\\n`."""
    pages = [
        _page(1, "para 1."),
        _page(2, "para 2."),
        _page(3, "para 3."),
    ]
    out = stitch_pages(pages)
    assert "\n\n---\n\n" not in out


# ---------- CJK vs Latin smart join ------------------------------------------


def test_cjk_cjk_join_has_no_space() -> None:
    """Two CJK fragments must concatenate without space."""
    stitcher = StreamingStitcher()
    chunks = list(stitcher.feed("电动汽车应通过检测点3"))
    chunks.extend(stitcher.feed("电压状态来识别"))
    chunks.extend(stitcher.finalize())
    assert chunks == ["电动汽车应通过检测点3电压状态来识别"]


def test_latin_latin_join_has_space() -> None:
    """Two Latin fragments must have a space between them."""
    stitcher = StreamingStitcher()
    chunks = list(stitcher.feed("shall be"))
    chunks.extend(stitcher.feed("connected"))
    chunks.extend(stitcher.finalize())
    assert chunks == ["shall be connected"]


def test_latin_then_cjk_no_space() -> None:
    """Latin ending in CJK-context: no space."""
    stitcher = StreamingStitcher()
    chunks = list(stitcher.feed("阶段"))
    chunks.extend(stitcher.feed("A.3.6.1 describes"))
    chunks.extend(stitcher.finalize())
    # CJK then Latin: no space (CJK has no word boundary)
    assert chunks == ["阶段A.3.6.1 describes"]
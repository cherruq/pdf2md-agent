"""跨页拼接后处理器。

每页的提取器一次只能看到一页，所以横跨页面边界的句子、
列表项或者表格行会被拆分成以显式的 ``\\n\\n---\\n\\n`` 
作为分隔符的两个片段。这种生硬的分隔符使得断层在输出中显露无疑，
而且这几乎总是错的：原本的文档是一段连续的块。

``StreamingStitcher`` 会在逐页面处理流水线 **之后** 运行并且
将这些片段重新拼接为一段连续的 Markdown 文档。
它是一个不依赖于 LLM 且十分纯粹的后置处理器 —— 两段片段
是否应当被拼接的决策取决于一组小巧而经济的文本启发式算法
（参见 :mod:`pdf2md_agent.post_stream_decision`）：

* 如果上一个片段的末尾不含有标点终结符并且下一个片段不是以新的一块
  （比如标题、列表、引用段落、表格、代码块）开头，它们便会被拼接在一起。
* 如果上一个片段由于未闭合的 markdown 表格行结束
  （存在奇数个 ``|``，且没有尾随 ``|``），该行会被强制闭合，
  然后追加接下来的片段；如果紧接着的片段重复生成了表头，那么这个表头会被丢弃。
* CJK（中日韩）字符拼接时中间不留空格；对于拉丁字母片段拼接时会在中间留单个空格。

这个类被称之为 "流式（Streaming）" 是因为它只会对每页的最后一段
片段进行缓冲 —— 其理念和流式解析器扣留一个 token 直到其看见接下来的信息
从而决定是否发出它是如出一辙的。
``feed()`` 会生成（yield）被确认为定稿的片段，而 ``finalize()`` 会在文档末尾处刷新所有
余留在缓冲区中的内容。最顶层的辅助函数
:func:`stitch_pages` 是对普通用例类的一层包装。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from enum import Enum

from pdf2md_agent.crew.orchestrator import PageResult
from pdf2md_agent.post_stream_decision import (
    _BLOCK_SEPARATOR,
    _decide_continuation,
    _is_unclosed_table_row,
    _last_meaningful_line,
    _smart_join,
    _split_into_blocks,
)
from pdf2md_agent.post_stream_table import _join_table_continuation


# --- Public API --------------------------------------------------------------


class StitchMode(str, Enum):
    """有多激进的将每页之间的片段进行拼接。"""

    OFF = "off"
    """拼接器前的行为：在每一页之间加上生硬的 ``\\n\\n---\\n\\n``。"""

    HEURISTIC = "heuristic"
    """默认选项。基于纯文本启发式算法，无需额外的 LLM 调用，无额外延迟。"""


def stitch_pages(
    pages: Iterable[PageResult],
    *,
    mode: StitchMode = StitchMode.HEURISTIC,
) -> str:
    """连接各页的 Markdown 并进行可选的跨页拼接。

    参数
    ----------
    pages
        每页的输出内容，来自 :func:`pdf2md_agent.crew.orchestrator.run_extraction_phase`。
        任何可迭代对象；只遍历一次。
    mode
        参见 :class:`StitchMode`。``HEURISTIC`` 为默认值，并且
        对于绝大多数散文、列表，以及简单的表格分割已经足够。

    返回
    -------
    str
        作为单个 Markdown 字符串的完整文档。在 ``HEURISTIC`` 模式下，
        页面之间没有 ``---`` 分隔符；在 ``OFF`` 模式下，
        旧版的 ``\\n\\n---\\n\\n`` 分隔符会被逐字保留。
    """
    _LEGACY_SEPARATOR = "\n\n---\n\n"

    if mode == StitchMode.OFF:
        return _LEGACY_SEPARATOR.join(r.markdown for r in pages)

    page_list = list(pages)
    cleaned_mds = _clean_page_markdown(page_list)

    stitcher = StreamingStitcher()
    chunks: list[str] = []
    for md in cleaned_mds:
        chunks.extend(stitcher.feed(md))
    chunks.extend(stitcher.finalize())
    return _BLOCK_SEPARATOR.join(chunks)


def _strip_repeating_header_footer(text: str, compare_text: str) -> str:
    """剥离在相邻页面之间重复出现的页眉或页脚行。"""
    if not text or not compare_text:
        return text

    lines = [ln.rstrip() for ln in text.splitlines()]
    comp_lines = [ln.strip() for ln in compare_text.splitlines() if ln.strip()]
    if not lines or not comp_lines:
        return text

    first_idx = next((k for k, ln in enumerate(lines) if ln.strip()), None)
    if first_idx is not None and len(lines) > 1:
        first_line = lines[first_idx].strip()
        if (
            len(first_line) >= 4
            and len(first_line) <= 120
            and not first_line.startswith(("|", "```", "~~~"))
            and first_line == comp_lines[0]
        ):
            lines[first_idx] = ""

    last_idx = next((k for k in range(len(lines) - 1, -1, -1) if lines[k].strip()), None)
    if last_idx is not None and last_idx != first_idx and len(lines) > 1:
        last_line = lines[last_idx].strip()
        if (
            len(last_line) >= 4
            and len(last_line) <= 120
            and not last_line.startswith(("|", "```", "~~~"))
            and last_line == comp_lines[-1]
        ):
            lines[last_idx] = ""

    return "\n".join(lines).strip()


def _strip_standalone_page_numbers(text: str) -> str:
    """剥离位于起始与末尾且单独占一行的页码行。"""
    if not text:
        return text
    lines = text.splitlines()
    no_pat = re.compile(
        r"^\s*(?:[-–—\s]*\d+[-–—\s]*|page\s*\d+|第\s*\d+\s*[页张])\s*$",
        re.IGNORECASE,
    )
    while lines and no_pat.match(lines[0]):
        lines.pop(0)
    while lines and no_pat.match(lines[-1]):
        lines.pop(-1)
    return "\n".join(lines).strip()


def _clean_page_markdown(pages: list[PageResult]) -> list[str]:
    """在横跨各页间剥离重复循环出现的页眉/页脚以及单独占一行的页码。"""
    cleaned: list[str] = []
    n = len(pages)
    for i, r in enumerate(pages):
        text = r.markdown.strip()
        if not text:
            cleaned.append("")
            continue

        if n > 1:
            neighbors: list[str] = []
            if i > 0:
                neighbors.append(pages[i - 1].markdown)
            if i < n - 1:
                neighbors.append(pages[i + 1].markdown)
            for compare_text in neighbors:
                text = _strip_repeating_header_footer(text, compare_text)

        text = _strip_standalone_page_numbers(text)
        cleaned.append(text)
    return cleaned


class StreamingStitcher:
    """具有最后一段片段向前查找（lookahead）的缓冲兼刷新拼接器。

    用法::

        stitcher = StreamingStitcher()
        for page_md in pages:
            for chunk in stitcher.feed(page_md):
                write(chunk)
        for chunk in stitcher.finalize():
            write(chunk)
    """

    def __init__(self) -> None:
        self._buffer: str = ""

    def feed(self, page_md: str) -> Iterator[str]:
        """生成（Yield）来自单页中已经被定稿确认好的片段；而对末尾的那一段作扣留处理。"""
        page_md = page_md.strip()
        if not page_md:
            return

        fragments = _split_into_blocks(page_md)
        if not fragments:
            return

        if self._buffer:
            decision = _decide_continuation(self._buffer, fragments[0])
            if decision.name == "CONTINUES":
                if _is_unclosed_table_row(_last_meaningful_line(self._buffer)):
                    self._buffer = _join_table_continuation(self._buffer, fragments[0])
                else:
                    self._buffer = _smart_join(self._buffer, fragments[0])
                fragments = fragments[1:]
                if not fragments:
                    # 拼接后的内容可能还在后继的下个页面上延续；
                    # 故保持将其留在缓冲池中直到对 feed() 的再次调用。
                    return
                # 该页面上后续仍有着多余的块 —— 所以此次的拼接被
                # 认定作已完结。直接将其产生（Yield）出去以防其在下方由于作为在末尾
                # 压轴的片段缓存而被无故地覆写掉。
                yield self._buffer
                self._buffer = ""
            else:
                yield self._buffer
                self._buffer = ""

        if not fragments:
            return

        for frag in fragments[:-1]:
            yield frag

        self._buffer = fragments[-1]

    def finalize(self) -> Iterator[str]:
        """将任何处于被扣留池中的片段全都刷新而出。幂等操作 —— 对此的二次调用将不会产出任何新内容。"""
        if self._buffer:
            yield self._buffer
            self._buffer = ""


# 用作于统一转换流程中的步骤 3（Step 3）的接入点别名
step3_stitch_and_clean = stitch_pages

__all__ = [
    "StitchMode",
    "StreamingStitcher",
    "step3_stitch_and_clean",
    "stitch_pages",
]

"""用于跨页面拼接器的决策引擎。

给定缓冲的“前一个片段”和下一页的第一个块，决定它们是否属于同一整体（以及如何拼接它们）。这里有两个子关注点：

* :func:`_split_into_blocks` — 将页面的 Markdown 拆分为由空行分隔的顶层块。
* :func:`_decide_continuation` — 做出 CONTINUES-vs-NEW_BLOCK（继续还是新块）的判定。
* :func:`_smart_join` — 实际的拼接，包含感知 CJK 的间距处理。
* :func:`_is_cjk` — CJK 统一表意文字及扩展区的谓词。

表格行延续逻辑位于 :mod:`pdf2md_agent.post_stream_table` 中，因为它采用与散文智能拼接不同的启发式算法（闭合 + 去重）。
"""

from __future__ import annotations

import re
from enum import Enum


# --- 与表格辅助函数共享的常量 ------------------------------

_BLOCK_SEPARATOR = "\n\n"
"""拼接输出中已确认块之间的空行分隔符。"""

_BLANK_LINE_RE = re.compile(r"\n\s*\n")
"""由 :func:`_split_into_blocks` 使用的空行边界；提升到模块作用域，因此该模式在每个进程中仅被编译一次，而不是每次按页调用时编译。"""


_SENTENCE_END = re.compile(r"[。！？；?!.．)\]】」』\"'`]+\s*$")
"""在 CJK + 拉丁文中标记句子/块结尾的后缀字符。"""

_BLOCK_START = re.compile(
    r"""^\s*(?:
        \#{1,6}\s           |  # ATX 标题
        [-*+]\s            |  # 无序列表（无缩进或列表缩进）
        \d+\.\s            |  # 有序列表
        >\s                |  # 块引用
        ```                |  # 围栏代码块
        \|\s               |  # 表格行
        \s{4,}                 # 4 个以上的前导空格 = 缩进代码块
    )""",
    re.VERBOSE,
)
"""标记新块开始的前缀模式（不得合并）。"""


# --- 拆分块 --------------------------------------------------------


def _split_into_blocks(page_md: str) -> list[str]:
    """将页面的 markdown 拆分为顶层块。

    一个块可以是：

    * 标题 + 其后的段落（直到出现空行）。
    * 列表（连续的 ``-``/``*``/``1.`` 行）。
    * 表格（以 ``|`` 开头的连续行）。
    * 独立的段落。
    * 代码围栏。

    空行用于分隔块。
    """
    chunks = _BLANK_LINE_RE.split(page_md)
    return [chunk.strip("\n") for chunk in chunks if chunk.strip()]


def _last_meaningful_line(text: str) -> str:
    """返回 ``text`` 的最后一个非空行（用于检查前一个尾部）。"""
    lines = [ln for ln in text.rstrip().split("\n") if ln.strip()]
    return lines[-1] if lines else ""


def _first_meaningful_line(text: str) -> str:
    """返回 ``text`` 的第一个非空行（用于检查当前头部）。"""
    for ln in text.lstrip().split("\n"):
        if ln.strip():
            return ln
    return ""


# --- 延续决策 -------------------------------------------------


class _Decision(Enum):
    CONTINUES = "continues"
    NEW_BLOCK = "new_block"


def _decide_continuation(prev: str, curr: str) -> _Decision:
    """决定 ``curr`` 是否延续 ``prev`` 开始的段落。

    启发式表格：

    ===========  =============  ============
    prev 结尾    curr 开始      结果
    ===========  =============  ============
    未完成       延续           CONTINUES
    完整         新块           NEW_BLOCK
    未完成       新块           NEW_BLOCK（例如 prev 是列表项，curr 是标题）
    完整         延续           NEW_BLOCK（不冒错误合并的风险）
    ===========  =============  ============

    特殊情况会覆盖该表：

    * prev 是标题 → 总是 NEW_BLOCK
    * prev 的最后一行是未闭合的表格行 → CONTINUES
      （在拼接时应用表格行语义）
    """
    prev_last = _last_meaningful_line(prev)
    curr_first = _first_meaningful_line(curr)

    if prev_last.lstrip().startswith("#"):
        return _Decision.NEW_BLOCK

    if _is_unclosed_table_row(prev_last):
        return _Decision.CONTINUES

    prev_complete = bool(_SENTENCE_END.search(prev_last.rstrip()))
    curr_new_block = bool(_BLOCK_START.match(curr_first))

    if not prev_complete and not curr_new_block:
        return _Decision.CONTINUES
    return _Decision.NEW_BLOCK


def _is_unclosed_table_row(line: str) -> bool:
    """如果 ``line`` 开始了一个 markdown 表格行但没有闭合的 ``|``，则为 True。

    竖线奇偶校验不是一个可靠的信号（它随单元格数量交替变化）；
    我们仅依赖于对末尾 ``|`` 的检查。
    """
    s = line.lstrip()
    if not s.startswith("|"):
        return False
    if s.rstrip().endswith("|"):
        return False
    return True


# --- 智能拼接 -------------------------------------------------------------


def _smart_join(prev: str, curr: str) -> str:
    """使用合适的分隔符拼接两个片段。

    * 任意一侧有 CJK 字符：无分隔符（无词界）。
    * prev 末尾有左括号 ``( [ {``：无分隔符。
    * 其他情况：单个空格。
    """
    prev = prev.rstrip()
    curr = curr.lstrip()
    if not prev:
        return curr
    if not curr:
        return prev
    last = prev[-1]
    if last in "([{「『【《":
        return prev + curr
    if _is_cjk(last) or _is_cjk(curr[0]):
        return prev + curr
    return prev + " " + curr


def _is_cjk(ch: str) -> bool:
    """如果 ``ch`` 是 CJK 表意文字，则为 True。"""
    if not ch:
        return False
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF
        or 0x3400 <= o <= 0x4DBF
        or 0x20000 <= o <= 0x2A6DF
        or 0x2A700 <= o <= 0x2B73F
        or 0x2B740 <= o <= 0x2B81F
        or 0xF900 <= o <= 0xFAFF
        or 0x2F800 <= o <= 0x2FA1F
    )


__all__ = [
    "_BLOCK_SEPARATOR",
    "_Decision",
    "_decide_continuation",
    "_first_meaningful_line",
    "_is_cjk",
    "_is_unclosed_table_row",
    "_last_meaningful_line",
    "_smart_join",
    "_split_into_blocks",
]

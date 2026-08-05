"""用于跨页拼接器的表格行延续辅助函数。

当页面边界在表格行中间截断 markdown 表格时，下一页的提取器
通常会在延续的行之前重新生成表头 + 分隔符。如果简单地拼接这两个片段会导致
表头重复。此模块包含：

* :func:`_join_table_continuation` — 闭合 ``prev`` 中未闭合的末尾行
  并将 ``curr`` 追加到其后（进行表头去重）。
* :func:`_strip_redundant_table_header` — 从 ``curr`` 中删除多余的
  表头 + 分隔符，以便它可以干净地追加。
"""

from __future__ import annotations

import re


_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
"""markdown 表格分隔符行（例如 ``|---|---|``）。"""


def _join_table_continuation(prev: str, curr: str) -> str:
    """当 ``prev`` 在表格行中间结束时，将 ``curr`` 追加到 ``prev``。

    * 通过追加 ``" |"`` 闭合 prev 中未闭合的最后一行。
    * 如果存在重复的表头 + 分隔符，则从 ``curr`` 中将其剥离。
    * 使用换行符连接，以便每行保持在自己单独的一行上。
    """
    prev_lines = prev.rstrip().split("\n")
    prev_lines[-1] = prev_lines[-1].rstrip() + " |"
    closed = "\n".join(prev_lines)
    deduped = _strip_redundant_table_header(curr)
    return closed + "\n" + deduped.lstrip()


def _strip_redundant_table_header(curr: str) -> str:
    """如果 ``curr`` 以 ``header / separator / rows...`` 开头，则丢弃
    冗余的表头 + 分隔符，使其干净地追加到上一个表格。
    """
    lines = curr.split("\n")
    if len(lines) < 3:
        return curr
    if _TABLE_SEPARATOR.match(lines[1].strip()):
        return "\n".join(lines[2:])
    return curr


__all__ = ["_join_table_continuation", "_strip_redundant_table_header"]

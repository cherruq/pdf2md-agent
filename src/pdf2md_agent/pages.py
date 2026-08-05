"""--pages CLI 标志的页面规范解析和验证。

两个纯函数，无 I/O 操作：

- :func:`parse_page_spec` 是 argparse 的 ``type=`` 可调用对象；它验证语法并在输入错误时引发 :class:`argparse.ArgumentTypeError`，以便 CLI 在打开 PDF 之前拒绝格式错误的规范。

- :func:`resolve_pages` 对解析后的页面列表进行去重、排序，并对照 PDF 的实际页数进行验证；对于超出范围的页面引发带有面向用户消息的 :class:`ValueError`。
"""

from __future__ import annotations

import argparse
import re

# 正则表达式（已锚定，容忍空格）。解析步骤仅对 token 强制执行“正整数”限制；上限随后由 resolve_pages 针对实际的 PDF 页数进行检查。
_TOKEN_RE = re.compile(r"^\s*(\d+)\s*$")
_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")

# 拒绝服务（DoS）保护：限制任何单个 "N-M" 范围的宽度。如果没有这个限制，
# ``--pages 1-999999999999`` 会在我们到达 ``resolve_pages``（它只看到正整数列表，无法判断其来自一个巨大的范围）之前实体化一个包含这么多整数的列表。
# 10_000 页大约是单个用户会处理的最大的现实世界 PDF；超过这个数字，该规范几乎肯定是恶意的或者是一个输入错误（例如原本想输入 1-999，但丢失了尾随的 9）。
_MAX_RANGE_SPAN = 10_000


def _err(msg: str) -> argparse.ArgumentTypeError:
    return argparse.ArgumentTypeError(msg)


def parse_page_spec(spec: str) -> list[int]:
    """将类似 ``'1-5,8,11-13'`` 的 --pages 值解析为 ``[1,2,3,4,5,8,11,12,13]``。

    逗号分隔的项目，每个项目可以是单个页码（``8``）或一个范围（``1-5``）。容忍数字和 ``-`` 周围的空格。

    页码必须是正整数（``>= 1``）；此处不强制执行上限（这是 :func:`resolve_pages` 的工作，因为它取决于实际的 PDF）。

    对于任何格式错误的输入，引发 :class:`argparse.ArgumentTypeError`。
    """
    if not isinstance(spec, str) or not spec.strip():
        raise _err(f"expected integer or N-M, got {spec!r}")

    pages: list[int] = []
    seen: set[int] = set()
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            raise _err(f"expected integer or N-M, got {raw_item!r}")

        m_range = _RANGE_RE.match(item)
        if m_range:
            start = int(m_range.group(1))
            end = int(m_range.group(2))
            if start == 0 or end == 0:
                raise _err(f"page numbers must be >= 1, got {item!r}")
            if start > end:
                raise _err(f"range start must be <= end, got {item!r}")
            if end - start + 1 > _MAX_RANGE_SPAN:
                raise _err(f"range {item!r} exceeds {_MAX_RANGE_SPAN} pages (spec too wide; split into smaller ranges)")
            for p in range(start, end + 1):
                if p not in seen:
                    seen.add(p)
                    pages.append(p)
            continue

        m_token = _TOKEN_RE.match(item)
        if m_token:
            n = int(m_token.group(1))
            if n == 0:
                raise _err(f"page numbers must be >= 1, got {item!r}")
            if n not in seen:
                seen.add(n)
                pages.append(n)
            continue

        raise _err(f"expected integer or N-M, got {item!r}")

    return pages


def resolve_pages(spec: list[int], total: int) -> list[int]:
    """对照 ``total`` 对解析后的页面列表进行去重、排序和验证。

    返回一个新的唯一页码列表，按升序排列，所有页码都在 ``[1, total]`` 范围内。

    在遇到第一个超出范围的页码时引发 :class:`ValueError`，消息格式为 ``"page N out of range (PDF
    has M pages)"``，以便 CLI 可以直接向用户显示。

    不会出现空结果：:func:`parse_page_spec` 保证每个项都是正整数，非空列表的去重结果也是非空的。不需要进行防御性的空列表检查。
    """
    if total < 1:
        raise ValueError(f"PDF has {total} pages; nothing to convert")

    out = sorted(set(spec))
    for n in out:
        if n < 1 or n > total:
            raise ValueError(f"page {n} out of range (PDF has {total} pages)")
    return out

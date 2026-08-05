"""提示文本和内嵌页面图像的 token 计数估算器。

两个纯函数，除了 ``Path.stat`` 之外没有其他 I/O：

* :func:`estimate_text_tokens` — 针对 CJK / 非 CJK 字符序列的粗略启发式算法，用于在调用 LLM 之前估算系统提示词 + 每页提示词的开销预算。
* :func:`estimate_image_tokens` — 将文件大小或字节缓冲区转换为其估计的 base64 token 开销。*永远不会*解码像素；估算器只读取 ``Path.stat().st_size``。

所有的估算器都有意采取保守策略 —— 它们从不调用任何外部的分词器（根据项目规范，禁止使用 ``tiktoken``），并且它们会高估以确保通过预算的调用在实践中一定能装得下。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Final, Union

log = logging.getLogger("pdf2md_agent.token_estimator")

# 根据 400 响应日志中的观察结果，3.5 是经验上安全的上限；
# 跨供应商的 base64 ↔ token 映射是不透明的，因此要高估。
_IMAGE_BYTES_PER_TOKEN: Final[float] = 3.5

# 每个 token 对应的 CJK / 宽字符数。混合的 CJK + 拉丁文散文
# 平均约 1.5 个字符一个 token；我们使用保守的 1/3 比例，以避免包含大量中文文本的页面预算不足。
_CJK_CHARS_PER_TOKEN: Final[float] = 3.0

# 拉丁文比例更接近 4 个字符 1 个 token；成对的引号/标点符号
# 会略微增加这个值，但该启发式算法是有意设计的粗略估算。
_ASCII_CHARS_PER_TOKEN: Final[float] = 4.0

PathOrBytes = Union[str, Path, bytes, bytearray]


def estimate_text_tokens(s: str) -> int:
    """使用混合的 CJK/ASCII 启发式算法估算文本提示词的 token 开销。

    将输入拆分为 CJK 序列（按 3 个字符 1 个 token 处理）和 ASCII 序列（按 4 个字符 1 个 token 处理），然后将两部分求和。该估算是有意设计的粗略估算 —— 它的目的是进行预算*规划*，而不是精确计费。

    Args:
        s: 我们想要为其进行 token 开销预算的提示文本。

    Returns:
        估计的 token 数量，作为 ``int`` (始终 >= 0)。
    """
    if not s:
        return 0

    cjk_chars = 0
    ascii_chars = 0
    for ch in s:
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x20000 <= code <= 0x2A6DF
            or 0xF900 <= code <= 0xFAFF
            or 0xFF00 <= code <= 0xFFEF
        ):
            cjk_chars += 1
        else:
            ascii_chars += 1

    cjk_tokens = math.ceil(cjk_chars / _CJK_CHARS_PER_TOKEN)
    ascii_tokens = math.ceil(ascii_chars / _ASCII_CHARS_PER_TOKEN)
    return cjk_tokens + ascii_tokens


def estimate_image_tokens(path_or_bytes: PathOrBytes, *, mime: str = "image/jpeg") -> int:
    """估算将图像内联为 base64 数据 URL 的 token 开销。

    仅查阅 ``Path.stat().st_size`` —— *不*解码像素。估算器假定每个字节最终都会变成长度为
    ``ceil(N/3) * 4`` 的 base64 字符串，并且每个 token 涵盖约 3.5 个 base64 字符。这
    与模型实际的速率相比高估了开销，但符合 ``400 context window exceeds limit`` 错误所报告的行为。

    Args:
        path_or_bytes: 图像的本地文件 ``Path``/``str`` 或原始 ``bytes``。
            此处不支持 ``http(s)://`` URL —— 调用者应已事先下载它们。
        mime: 仅字节估算器未使用；为了与未来感知 Pillow 的估算器在 API 上保持对称而接受。

    Returns:
        估计的 token 数量，作为 ``int``。
    """
    del mime
    if isinstance(path_or_bytes, (str, Path)):
        path = Path(path_or_bytes)
        if not path.is_file():
            log.debug("estimate_image_tokens: %s is not a file; assuming 0", path)
            return 0
        size = path.stat().st_size
    elif isinstance(path_or_bytes, (bytes, bytearray)):
        size = len(path_or_bytes)
    else:
        raise TypeError(f"estimate_image_tokens: unsupported type {type(path_or_bytes).__name__}")

    b64_chars = ((size + 2) // 3) * 4
    return max(1, math.ceil(b64_chars / _IMAGE_BYTES_PER_TOKEN))


__all__ = [
    "PathOrBytes",
    "estimate_image_tokens",
    "estimate_text_tokens",
]

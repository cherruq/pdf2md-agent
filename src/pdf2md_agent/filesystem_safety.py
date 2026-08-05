"""用于缓存目录命名的文件系统安全辅助函数。

由 CLI 和缓存布局使用的纯函数，用于计算 PDF 中间缓存的
确定性、文件系统安全的目录名。此处解决两个问题：

* :func:`safe_cache_stem` — 净化 PDF 的文件名主干（stem），使其永远不会与 Windows 保留设备名（``CON``，``PRN``，…）冲突，并且永远不会生成只有点/空格的目录。
* :func:`cache_key_for_pdf` — 将净化后的主干与针对过长或包含路径分隔符的主干的基于哈希的回退方案结合，以便两个不同的绝对路径始终对应不同的缓存目录。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_WINDOWS_RESERVED_NAMES: frozenset[str] = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
)
"""Windows 保留设备名：``CON``/``PRN``/``AUX``/``NUL`` 以及
``COM1``-``COM9`` 和 ``LPT1``-``LPT9``。在 Windows 上不区分大小写。

``CreateFile`` 系统调用拒绝将这些作为裸文件名（无论有无扩展名），并且在保留名称上执行 ``mkdir`` 会表现为不透明的 ``OSError``。我们附加一个 ``_`` ，使得缓存位于 ``<reserved>_`` 而不是导致运行崩溃。"""

_MAX_CACHE_STEM_LEN: int = 60


def safe_cache_stem(stem: str) -> str:
    """返回从 ``stem`` 派生的文件系统安全的缓存目录名称。

    剥离尾随的点/空格，并在 Windows 上的 Windows 保留名称后附加 ``_`` ，以便 ``mkdir`` 成功。在非 Windows 平台上，尾随的点/空格仍然被防御性地剥离以保证可移植性。

    大小写冲突 (D16-002)：在不区分大小写的文件系统（NTFS，APFS，HFS+）上，仅大小写不同的两个 PDF 将映射到同一个缓存目录。我们在这里不进行规范化 —— 相反，通过此函数的文档字符串警告调用者选择不同的主干。
    """
    if not stem:
        return "_"
    candidate = stem.rstrip(" .")
    if not candidate:
        return "_"
    if sys.platform == "win32" and candidate.upper() in _WINDOWS_RESERVED_NAMES:
        return candidate + "_"
    return candidate


def cache_key_for_pdf(pdf: Path) -> str:
    """返回 ``pdf`` 的确定性缓存目录名称。

    当 PDF 的主干很短，没有路径分隔符，并且不是 Windows 保留名称时，使用 PDF 的主干。对于过长的主干，包含 ``/`` 的名称（例如，当 PDF 位于深度嵌套的树下时），或 Windows 主机上 Windows 保留的主干，缓存键是绝对 PDF 路径的 16 字符 SHA-256 摘要 —— 每个文件具有确定性，不同绝对路径之间绝不会发生冲突。
    """
    abs_path = pdf.resolve()
    stem = safe_cache_stem(abs_path.stem)
    if (
        0 < len(stem) <= _MAX_CACHE_STEM_LEN
        and "/" not in abs_path.stem
        and "\\" not in abs_path.stem
        and (sys.platform != "win32" or stem.upper() not in _WINDOWS_RESERVED_NAMES)
    ):
        return stem
    return hashlib.sha256(str(abs_path).encode("utf-8")).hexdigest()[:16]


__all__ = ["cache_key_for_pdf", "safe_cache_stem"]

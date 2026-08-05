"""用于按页提取流水线的集中式数据结构。

这些结构代表了流水线各个步骤的输入、上下文和输出。
它们是根据单个页面的处理生命周期中创建和使用的时间按时间顺序排列的。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pdf2md_agent.cache import PageArtifacts


# 1. 流水线上下文
@dataclass(frozen=True, slots=True)
class PageRunContext:
    """当前正在处理的页面的只读全局元数据。
    
    该结构体贯穿整个流水线，以提供日志记录上下文和进度跟踪元数据，
    而不会污染各个函数的签名。
    """

    page_number: int
    """该页的物理页码。"""
    idx: int
    """当前正在处理的页面在流水线批次中基于 1 的索引。"""
    total: int
    """当前流水线批次中的总页数。"""
    page_started: float
    """在页面流水线开始时获取的 ``time.monotonic()`` 快照 ——
    由调用者用于记录每页的耗时。"""


# 2. 渲染的 PDF 页面
@dataclass(frozen=True, slots=True)
class RenderedPage:
    """一个渲染后的 PDF 页面：栅格化图像产物和原生文本层。
    
    这是 PDF 渲染阶段的输出，并作为提取流水线其余部分的主要数据输入。
    """

    width: int
    height: int
    image_path: Path
    ctx: PageRunContext
    text_path: Path | None = None
    text: str = ""


# 3. 图像准备输出
@dataclass(frozen=True, slots=True)
class PreparedPage:
    """为提取循环准备的输入（步骤 1 的输出）。

    此结构包含原始渲染页面和实际应附加到 LLM 的图像（可能已缩小或切片）。
    """

    page: RenderedPage
    """原始渲染的 PDF 页面。"""
    text_hint_str: str
    """文本层内容或空字符串。"""
    attach_image_path: Path
    """任务应引用的路径。对于在原始大小下已经符合预算的页面，
    等于 ``page.image_path``，或者如果调整了大小，则是缩小的缓存路径。"""
    is_tiled: bool
    """如果页面被拆分为 ``tile_paths``（极端预算情况），则为 True。"""
    tile_paths: list[Path]
    """一半重叠的 JPEG 切片；当 ``is_tiled`` 为 False 时为空。"""
    ctx: PageRunContext
    """当前页面的流水线上下文。"""


# 4. 提取步骤输出
@dataclass(frozen=True, slots=True)
class ExtractionOutcome:
    """单页提取循环的结果（步骤 2 的输出）。

    此结构体捕获通过视觉模型提取是否成功，或者是否降级/回退到文本层存根。
    """

    format_md: str
    """生成的最终 markdown 内容（视觉模型输出或回退存根）。"""
    succeeded: bool
    """如果视觉提取顺利成功，则为 True。"""
    fell_back: bool
    """如果视觉模型失败且我们降级到文本层，则为 True。"""
    ctx: PageRunContext
    """当前页面的流水线上下文。"""


# 5. 回退执行输入
@dataclass(frozen=True, slots=True)
class FallbackRecord:
    """为文本层回退处理程序打包的参数。

    打包字段可保持调用站点的可读性；运行器将其中一个传递给每个回退路径上的辅助函数。
    """

    artifacts: PageArtifacts
    """当前页面缓存产物的文件路径。"""
    completion_label: str
    """日志记录标签（例如，'fallback'，'validation-fallback'）。"""
    ctx: PageRunContext
    """当前页面的流水线上下文。"""


# 6. 流水线输出
@dataclass(frozen=True, slots=True)
class PageResult:
    """单页的最终输出（步骤 3 的输出）。

    运行器生成并返回到 CLI 层的最终结果。
    """

    page_number: int
    """物理页码。"""
    markdown: str
    """最终处理的 markdown 内容。"""


__all__ = [
    "RenderedPage",
    "PageRunContext",
    "PreparedPage",
    "ExtractionOutcome",
    "FallbackRecord",
    "PageResult",
]

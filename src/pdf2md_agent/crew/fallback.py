"""用于逐页流水线的文本层降级辅助函数。

当视觉模型在所有重试后都无法访问时，或者当其输出
未能通过 CrewAI 的任务输出验证时，runner 会发出一个带栅栏的
文本层 Markdown 存根，而不是使整个运行崩溃。此模块
隔离了：

* :func:`_text_layer_fallback` — 从 PDF 的
  原生文本层构建单个页面的存根 Markdown。
* :func:`_record_text_layer_fallback` — 将存根写入
  页面的格式化工件中，并返回一个 :class:`PageResult`。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from pdf2md_agent.cache import PageArtifacts
from pdf2md_agent.crew.types import FallbackRecord, PageRunContext
from pdf2md_agent.pdf_renderer import read_page_text

log = logging.getLogger("pdf2md_agent.runner")


def _text_layer_fallback(artifacts: PageArtifacts) -> str:
    """从 PDF 的原生文本层尽力构建一个 Markdown 页面。

    在所有重试后视觉模型仍无法访问时使用。页面的
    PNG 会从输出中被丢弃（我们无法描述它），并且文本会
    被原封不动地输出在一个带栅栏的代码块中，以便审阅者可以发现偏差。
    """
    text = read_page_text(artifacts.page_text).strip()
    if not text:
        return "*(vision model unavailable and PDF text layer is empty for this page — no content recovered)*"
    return (
        "*(vision model unavailable — falling back to PDF text layer; "
        "tables, figures, and layout are NOT preserved)*\n\n"
        "```\n"
        f"{text}\n"
        "```\n"
    )


def _record_text_layer_fallback(
    *,
    ctx: PageRunContext,
    artifacts: PageArtifacts,
    completion_label: str,
) -> str:
    """为一个页面写入降级工件并返回降级 Markdown。

    将降级 Markdown 写入 ``format.md``。
    """
    record = FallbackRecord(
        artifacts=artifacts,
        completion_label=completion_label,
        ctx=ctx,
    )
    format_md = _text_layer_fallback(record.artifacts)
    record.artifacts.format_markdown.write_text(format_md, encoding="utf-8")

    elapsed = time.monotonic() - record.ctx.page_started
    log.info(
        "  [%d/%d] page %d: done in %.1fs (%s, %s chars)",
        record.ctx.idx,
        record.ctx.total,
        record.ctx.page_number,
        elapsed,
        record.completion_label,
        f"{len(format_md):,}",
    )
    return format_md


__all__ = [
    "FallbackRecord",
    "_record_text_layer_fallback",
    "_text_layer_fallback",
]

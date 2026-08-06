"""用于逐页流水线的文本层降级辅助函数。

当视觉模型在所有重试后都无法访问时，或者当其输出
未能通过 CrewAI 的任务输出验证时，runner 会发出一个带栅栏的
文本层 Markdown 存根，而不是使整个运行崩溃。此模块
隔离了该逻辑。
"""

from __future__ import annotations

import logging
import time

from pdf2md_agent.cache import PageArtifacts
from pdf2md_agent.crew.types import PageRunContext
from pdf2md_agent.pdf_renderer import read_page_text

log = logging.getLogger("pdf2md_agent.runner")


def handle_extraction_fallback(
    *,
    ctx: PageRunContext,
    artifacts: PageArtifacts,
    completion_label: str,
) -> str:
    """提取失败时的降级处理：直接读取 PDF 原生文本并写入 format.md 缓存。"""
    text = read_page_text(artifacts.page_text).strip()

    if not text:
        format_md = "*(vision model unavailable and PDF text layer is empty for this page — no content recovered)*"
    else:
        format_md = (
            "*(vision model unavailable — falling back to PDF text layer; "
            "tables, figures, and layout are NOT preserved)*\n\n"
            "```\n"
            f"{text}\n"
            "```\n"
        )

    artifacts.format_markdown.write_text(format_md, encoding="utf-8")

    elapsed = time.monotonic() - ctx.page_started
    log.info(
        "  [%d/%d] page %d: done in %.1fs (%s, %s chars)",
        ctx.idx,
        ctx.total,
        ctx.page_number,
        elapsed,
        completion_label,
        f"{len(format_md):,}",
    )

    return format_md


__all__ = [
    "handle_extraction_fallback",
]

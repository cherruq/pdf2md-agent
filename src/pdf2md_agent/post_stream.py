"""跨页拼接后处理器（大模型滑动窗口版本）。

``LLMStitcher`` 会在逐页面处理流水线 **之后** 运行并且
利用现成的大模型能力将这些片段重新拼接为一段连续的 Markdown 文档。
它会提取每两页交界处的断层面，并通过并发的 LLM 调用智能地恢复句子、
剔除页眉/页脚。
"""

from __future__ import annotations

import concurrent.futures
import logging
from collections.abc import Iterable
from enum import Enum
from typing import Any

from pdf2md_agent.crew.orchestrator import PageResult

log = logging.getLogger("pdf2md_agent.stitch")

# 提取交界处时，保留的上下文最大字符数
_SEAM_CONTEXT_CHARS = 1000

class StitchMode(str, Enum):
    """跨页拼接模式。"""

    OFF = "off"
    """保留旧版的生硬分页符 ``\\n\\n---\\n\\n``，不进行拼接。"""

    AUTO = "auto"
    """默认选项。使用 LLM 自动且智能地缝合跨页断层。"""


def stitch_pages(
    pages: Iterable[PageResult],
    llm: Any = None,
    *,
    mode: StitchMode = StitchMode.AUTO,
) -> str:
    """连接各页的 Markdown 并利用大模型进行跨页断层缝合。

    参数
    ----------
    pages
        每页的输出内容。
    llm
        用于执行缝合的大模型实例 (crewai.LLM)。
    mode
        参见 :class:`StitchMode`。

    返回
    -------
    str
        作为单个 Markdown 字符串的完整文档。
    """
    _LEGACY_SEPARATOR = "\n\n---\n\n"
    page_list = list(pages)

    if not page_list:
        return ""

    if mode == StitchMode.OFF or len(page_list) == 1:
        return _LEGACY_SEPARATOR.join(r.markdown.strip() for r in page_list)

    if llm is None:
        log.warning("No LLM provided for stitching. Falling back to simple concatenation.")
        return _LEGACY_SEPARATOR.join(r.markdown.strip() for r in page_list)

    log.info("Starting concurrent LLM stitching for %d seams...", len(page_list) - 1)
    
    # 构建所有缝合任务
    seams_prompts = []
    for i in range(len(page_list) - 1):
        prev_md = page_list[i].markdown.strip()
        next_md = page_list[i + 1].markdown.strip()
        
        # 提取末尾和开头作为缝合上下文
        prev_tail = prev_md[-_SEAM_CONTEXT_CHARS:] if len(prev_md) > _SEAM_CONTEXT_CHARS else prev_md
        next_head = next_md[:_SEAM_CONTEXT_CHARS] if len(next_md) > _SEAM_CONTEXT_CHARS else next_md
        
        prompt = (
            "你是一个专业的 Markdown 格式整理专家。下面是文档第 N 页结尾和第 N+1 页开头的片段。\n"
            "它们之间存在强制分页符造成的断层，且可能包含不需要的重复页眉或页脚。\n\n"
            "【上一页结尾片段】\n"
            f"{prev_tail}\n"
            "【下一页开头片段】\n"
            f"{next_head}\n\n"
            "任务要求：\n"
            "1. 找出它们真正的连续逻辑，如果语义是连续的（如被截断的句子、列表、表格），请无缝拼合并剔除中间的页眉页脚。\n"
            "2. 如果它们在语义上是不连续的两个独立段落，请用双换行符 (\\n\\n) 将它们隔开。\n"
            "3. 严禁篡改或总结原文，只能进行拼接和去重操作。\n"
            "4. 只输出合并后的交界面 Markdown 文本（不要输出完整的两页内容，只输出被合并处的过渡文本）。"
        )
        seams_prompts.append((i, prompt, prev_tail, next_head))

    seam_results: dict[int, str] = {}
    
    def _stitch_seam(idx: int, prompt: str) -> tuple[int, str]:
        try:
            # 直接调用 LLM (兼容 crewai.LLM.call 签名)
            response = llm.call([{"role": "user", "content": prompt}])
            return idx, str(response).strip()
        except Exception as exc:
            log.warning("Seam %d stitching failed: %s. Using legacy separator.", idx, exc)
            return idx, _LEGACY_SEPARATOR

    # 并发执行所有的缝合调用
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(seams_prompts))) as executor:
        futures = [executor.submit(_stitch_seam, idx, p) for idx, p, _, _ in seams_prompts]
        for future in concurrent.futures.as_completed(futures):
            idx, merged_seam = future.result()
            seam_results[idx] = merged_seam

    # 组装最终文档
    final_chunks = []
    for i in range(len(page_list)):
        current_md = page_list[i].markdown.strip()
        
        if i == 0:
            # 第一页：移除作为缝合上下文的尾部
            tail_len = len(current_md[-_SEAM_CONTEXT_CHARS:]) if len(current_md) > _SEAM_CONTEXT_CHARS else len(current_md)
            if tail_len > 0 and tail_len < len(current_md):
                final_chunks.append(current_md[:-tail_len])
            elif tail_len == len(current_md):
                pass # 整个内容都在 seam 里，这里不添加，由 seam 负责
        elif i == len(page_list) - 1:
            # 最后一页：移除作为缝合上下文的头部
            head_len = len(current_md[:_SEAM_CONTEXT_CHARS]) if len(current_md) > _SEAM_CONTEXT_CHARS else len(current_md)
            
            # 添加上一个断层
            final_chunks.append(seam_results[i - 1])
            
            if head_len > 0 and head_len < len(current_md):
                final_chunks.append(current_md[head_len:])
            elif head_len == len(current_md):
                pass # 整个内容在上一 seam 里
        else:
            # 中间页：移除头部（与 i-1 缝合）和尾部（与 i 缝合）
            head_len = len(current_md[:_SEAM_CONTEXT_CHARS]) if len(current_md) > _SEAM_CONTEXT_CHARS else len(current_md)
            
            # 添加上一个断层
            final_chunks.append(seam_results[i - 1])
            
            # 如果中间还有剩余的内容（即一页超过 2000 字）
            if len(current_md) > 2 * _SEAM_CONTEXT_CHARS:
                middle_part = current_md[_SEAM_CONTEXT_CHARS:-_SEAM_CONTEXT_CHARS]
                final_chunks.append(middle_part)
            # 如果内容被前后 seam 完全覆盖，就不再重复添加中部

    return "".join(final_chunks).strip()


step3_stitch_and_clean = stitch_pages

__all__ = [
    "StitchMode",
    "step3_stitch_and_clean",
    "stitch_pages",
]

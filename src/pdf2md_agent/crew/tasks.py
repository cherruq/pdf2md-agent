"""按页任务工厂：提取 → 格式化。"""

from __future__ import annotations

from pathlib import Path

from crewai import Agent, Task

from pdf2md_agent.crew.multimodal_patch import patch_add_image_tool

# 幂等操作：确保 AddImageTool 将本地路径转换为 data: URL 并且
# 在发送给兼容 OpenAI 的视觉 API（这些 API 拒绝裸路径和超大图像）之前，将其重新编码为 JPEG（长边受限）。
patch_add_image_tool()

_NO_REASONING = (
    "Output ONLY final content; no reasoning, preamble, or "
    + chr(60)
    + "think"
    + chr(62)
    + "..."
    + chr(60)
    + "/think"
    + chr(62)
    + " blocks."
)

_LANG_RULE = (
    "Language rule: write in the exact same language(s) as the source — "
    "preserve every CJK character, Latin word, and punctuation mark; "
    "never translate."
)

_VERBATIM_RULE = (
    "Verbatim rule: copy the page character-for-character — no "
    "translation, summarization, or invented content; write `[illegible]` "
    "for unreadable glyphs; omit running page-margin headers, footers, "
    "and page numbers, but MUST preserve document titles and headings."
)

_COMMON_TASK_RULES: str = f"{_VERBATIM_RULE}\n\n{_LANG_RULE}\n\n{_NO_REASONING}"
TASKS_RULES_TEXT: str = _COMMON_TASK_RULES


def extract_task_intro(
    page_path: Path,
    is_tiled: bool = False,
    tile_paths: list[Path] | None = None,
) -> str:
    if is_tiled and tile_paths:
        tile_str = ", ".join(f"`{p}`" for p in tile_paths)
        return (
            f"This page was too large and has been split into tiles. "
            f"Call your add_image tool with image_url on each of these paths: {tile_str}. "
            f"Then transcribe their combined full content into strict CommonMark markdown.\n\n"
        )
    return (
        f"Call your add_image tool with image_url=`{page_path}` to attach "
        f"the rendered page image, then transcribe its full content "
        f"into strict CommonMark markdown.\n\n"
    )


def _text_hint_block(text: str) -> str:
    """构建附加到提取任务的文本提示块，如果没有则为空字符串。"""
    text = text.strip()
    if not text:
        return ""
    return (
        "Text-hint (extracted from the PDF's native text layer — treat as "
        "ground truth for prose, numbers, units, formula symbols, table "
        "cell content; use the image for layout, figures, and visual "
        "structure. If they disagree on order or wording, follow the image "
        "for structure and the text for exact wording):\n"
        "```\n"
        f"{text}\n"
        "```\n\n"
    )


def build_extract_description(
    page_path: Path,
    text_hint: str = "",
    *,
    is_tiled: bool = False,
    tile_paths: list[Path] | None = None,
) -> str:
    """构建提取任务发送给 LLM 的确切描述字符串。"""
    return (
        f"{_text_hint_block(text_hint)}"
        f"{extract_task_intro(page_path, is_tiled, tile_paths)}"
        "Formatting rule: output strict CommonMark markdown. Normalize table "
        "syntax, fix broken lists, and strip OCR noise.\n\n"
        f"{TASKS_RULES_TEXT}"
    )


def make_extract_task(
    extractor: Agent,
    page_path: Path,
    text_hint: str = "",
    *,
    is_tiled: bool = False,
    tile_paths: list[Path] | None = None,
) -> Task:
    """创建带有图像 + 文本提示的页面提取任务。"""
    description = build_extract_description(
        page_path,
        text_hint,
        is_tiled=is_tiled,
        tile_paths=tile_paths,
    )
    return Task(
        description=description,
        expected_output="Clean CommonMark markdown transcription of the page, verbatim language preserved",
        agent=extractor,
    )

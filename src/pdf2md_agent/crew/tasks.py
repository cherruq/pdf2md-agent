"""Per-page task factories: extract → format.

Each task description embeds small behavioral rules instead of long boilerplate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from crewai import Agent, Task

from pdf2md_agent.crew.multimodal_patch import patch_add_image_tool

# Idempotent: ensures AddImageTool converts local paths to data: URLs and
# re-encodes them as JPEG (long-side capped) before sending them to
# OpenAI-compatible vision APIs (which reject bare paths and oversized images).
patch_add_image_tool()

_NO_REASONING = (
    "Output ONLY final content; no reasoning, preamble, or "
    + chr(60) + "think" + chr(62) + "..."
    + chr(60) + "/think" + chr(62) + " blocks."
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

# Joined rule text shared by every task that asks the LLM to write verbatim
# Markdown: the formatter factory (and its file-fed sibling) and the
# extractor's task description. The ``chr(60)/chr(62)`` escape around
# ``<think>`` is load-bearing (see AGENTS.md → crew/ → CONVENTIONS); do not
# "refactor" to a literal tag.
_COMMON_TASK_RULES: str = f"{_VERBATIM_RULE}\n\n{_LANG_RULE}\n\n{_NO_REASONING}"

# Joined rule text consumed by the token-budget planner.
TASKS_RULES_TEXT: str = _COMMON_TASK_RULES


def extract_task_intro(
    page_path: Path,
    available_images: list[str] | None = None,
    is_tiled: bool = False,
    tile_paths: list[Path] | None = None,
) -> str:
    if is_tiled and tile_paths:
        tile_str = ", ".join(f"`{p}`" for p in tile_paths)
        intro = (
            f"This page was too large and has been split into tiles. "
            f"Call your add_image tool with image_url on each of these paths: {tile_str}. "
            f"Then transcribe their combined full content into raw markdown.\n\n"
        )
    else:
        intro = (
            f"Call your add_image tool with image_url=`{page_path}` to attach "
            f"the rendered page image, then transcribe its full content "
            f"into raw markdown.\n\n"
        )
    if available_images:
        images_str = ", ".join(f"`{img}`" for img in available_images)
        intro += (
            f"Note: The following native images were extracted from this page and are available "
            f"in the assets directory if you need to reference them: {images_str}.\n\n"
        )
    return intro


def _text_hint_block(text: str) -> str:
    """Build the text-hint block appended to the extract task, or empty string."""
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
    *args: Any,
    available_images: list[str] | None = None,
    is_tiled: bool = False,
    tile_paths: list[Path] | None = None,
    **kwargs: Any,
) -> str:
    """Build the exact description string the extract task sends to the LLM.

    Shared between ``make_extract_task`` (which wraps it in a CrewAI Task)
    and the runner's token-budget planner (which needs to estimate the cost
    of the same prompt), guaranteeing the budget never diverges from the
    real payload.
    """
    return (
        f"{_text_hint_block(text_hint)}"
        f"{extract_task_intro(page_path, available_images, is_tiled, tile_paths)}"
        f"{TASKS_RULES_TEXT}"
    )


def make_extract_task(
    extractor: Agent,
    page_path: Path,
    text_hint: str = "",
    *args: Any,
    available_images: list[str] | None = None,
    is_tiled: bool = False,
    tile_paths: list[Path] | None = None,
    **kwargs: Any,
) -> Task:
    """Create the page-extraction task with image + text hint."""
    description = build_extract_description(
        page_path,
        text_hint,
        available_images=available_images,
        is_tiled=is_tiled,
        tile_paths=tile_paths,
    )
    return Task(
        description=description,
        expected_output="Verbatim markdown transcription of the page",
        agent=extractor,
    )


def make_format_task(
    formatter: Agent,
    extract_task: Task,
) -> Task:
    """Create the cleanup task; sees the extractor's output via context."""
    description = (
        "Rewrite the extracted markdown as strict CommonMark. Fix "
        "broken lists, normalize table syntax, strip OCR noise.\n\n"
        f"{_COMMON_TASK_RULES}"
    )
    return Task(
        description=description,
        expected_output="Clean CommonMark markdown of the page, language preserved",
        agent=formatter,
        context=[extract_task],
    )


def make_format_task_from_extract_file(
    formatter: Agent,
    extract_path: Path,
) -> Task:
    """Format task fed from a cached ``page_NNNN_extract.txt`` on disk.

    Used when the runner trusts the cached extract (the ``--no-cache-extract``
    flag is unset) but still wants a fresh formatter pass — typically a
    resume-after-failure retry, or a manual re-run where only the formatter
    has changed. The file's full text is pasted into the description as a
    fenced block, matching the ``_text_hint_block`` seam so the runner has
    no new tool surface to maintain.

    Caller is responsible for ensuring the file exists (gate on
    ``cache.has_cached_extract`` first).
    """
    text = extract_path.read_text(encoding="utf-8")
    return Task(
        description=(
            "Rewrite the extracted markdown below as strict CommonMark. "
            "Preserve every word verbatim — do not drop, translate, or "
            "rewrite content. Only normalize formatting; output language "
            "must exactly match the input.\n\n"
            f"{_COMMON_TASK_RULES}\n\n"
            "Extracted content (read from disk; treat as ground truth):\n"
            "```\n"
            f"{text}\n"
            "```"
        ),
        expected_output="Clean CommonMark markdown of the page, language preserved",
        agent=formatter,
        context=[],
    )

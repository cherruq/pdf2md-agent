"""Per-page task factories: chain extract → format → summarize."""
from __future__ import annotations

from pathlib import Path

from crewai import Agent, Task

from convertpdf.crew.multimodal_patch import patch_add_image_tool

# Idempotent: ensures AddImageTool converts local paths to data: URLs before
# sending them to OpenAI-compatible vision APIs (which reject bare paths).
patch_add_image_tool()

_NO_REASONING = (
    "Output ONLY the final content. Do not include any reasoning, "
    "preamble, or <think>...</think> blocks in your response."
)

_LANG_RULE = (
    "Language rule: write in the EXACT same language(s) as the source "
    "page. Preserve every CJK character, every Latin word, every "
    "punctuation mark. Do NOT translate, even if the page mixes "
    "languages."
)

_VERBATIM_RULE = (
    "Verbatim rule: copy what is on the page character-for-character. "
    "Do not paraphrase, summarize, fix typos, or insert content that "
    "is not visible on the page. If a character is unreadable, write "
    "`[illegible]` rather than guess."
)


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


def make_extract_task(
    extractor: Agent,
    page_path: Path,
    text_hint: str = "",
    previous_summary: str = "",
) -> Task:
    """Create the page-extraction task with image + text hint + cross-page context."""
    summary_block = (
        f"Running summary of preceding pages:\n{previous_summary}\n\n"
        if previous_summary.strip()
        else ""
    )
    description = (
        f"{summary_block}"
        f"{_text_hint_block(text_hint)}"
        f"Call your add_image tool with image_url=`{page_path}` to attach "
        f"the rendered page image, then transcribe its full content "
        f"into raw markdown.\n\n"
        f"{_VERBATIM_RULE}\n\n"
        f"{_LANG_RULE}\n\n"
        f"{_NO_REASONING}"
    )
    return Task(
        description=description,
        expected_output="Verbatim markdown transcription of the page",
        agent=extractor,
    )


def make_format_task(formatter: Agent, extract_task: Task) -> Task:
    """Create the cleanup task; sees the extractor's output via context."""
    return Task(
        description=(
            "Rewrite the extracted markdown as strict CommonMark. Fix "
            "broken lists, normalize table syntax, strip OCR noise.\n\n"
            f"{_VERBATIM_RULE}\n\n"
            f"{_LANG_RULE}\n\n"
            f"{_NO_REASONING}"
        ),
        expected_output="Clean CommonMark markdown of the page, language preserved",
        agent=formatter,
        context=[extract_task],
    )


def make_summarize_task(
    summarizer: Agent,
    format_task: Task,
    previous_summary: str,
) -> Task:
    """Create the summary-update task; sees current page + previous summary."""
    previous_block = (
        f"Previous running summary:\n{previous_summary}\n\n"
        if previous_summary.strip()
        else "This is the first page; start a fresh summary.\n\n"
    )
    return Task(
        description=(
            f"{previous_block}"
            "Update the running summary to incorporate the current "
            "page. Keep it under ~200 words; preserve named entities, "
            "running arguments, and unresolved threads.\n\n"
            f"{_LANG_RULE}\n\n"
            f"{_NO_REASONING}"
        ),
        expected_output="Updated running summary, ~200 words, language matches source",
        agent=summarizer,
        context=[format_task],
    )
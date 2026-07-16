"""Per-page task factories: chain extract → format → summarize.

Each task description embeds three small behavioral rules instead of long
boilerplate — every rule is now a single sentence. A
``MAX_SUMMARY_CHARS`` budget is enforced both when the previous summary is
fed *in* (truncated to fit) and when the summarizer is asked to emit
(constrained via the task prompt).
"""
from __future__ import annotations

from pathlib import Path

from crewai import Agent, Task

from convertpdf.crew.multimodal_patch import patch_add_image_tool

# Idempotent: ensures AddImageTool converts local paths to data: URLs and
# re-encodes them as JPEG (long-side capped) before sending them to
# OpenAI-compatible vision APIs (which reject bare paths and oversized images).
patch_add_image_tool()

# Defaults — runner may override via max_summary_chars parameter.
MAX_SUMMARY_CHARS: int = 800

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
    "for unreadable glyphs."
)

_SUMMARY_TRUNCATION_SUFFIX = "[…summary truncated to fit context window]"

# Joined rule text consumed by the token-budget planner.
TASKS_RULES_TEXT: str = f"{_VERBATIM_RULE}\n\n{_LANG_RULE}\n\n{_NO_REASONING}"


def _truncate_summary(text: str, max_chars: int) -> str:
    """Return ``text`` trimmed to ``max_chars``, preserving head + tail.

    The middle is dropped so the most recent running context (tail) and the
    high-level topic (head) both survive. A sentinel suffix tells the
    extractor agent that the visible state is incomplete.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    budget = max(0, max_chars - len(_SUMMARY_TRUNCATION_SUFFIX) - 2)
    head_budget = budget // 2
    tail_budget = budget - head_budget
    head = text[:head_budget].rstrip()
    tail = text[-tail_budget:].lstrip() if tail_budget > 0 else ""
    if tail:
        return f"{head}\n{_SUMMARY_TRUNCATION_SUFFIX}\n{tail}"
    return f"{head}\n{_SUMMARY_TRUNCATION_SUFFIX}"


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
    *,
    max_summary_chars: int = MAX_SUMMARY_CHARS,
) -> Task:
    """Create the page-extraction task with image + text hint + cross-page context."""
    safe_summary = _truncate_summary(previous_summary, max_summary_chars)
    summary_block = (
        f"Running summary of preceding pages:\n{safe_summary}\n\n"
        if safe_summary.strip()
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
    *,
    max_chars: int = MAX_SUMMARY_CHARS,
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
            f"Update the running summary to incorporate the current page. "
            f"Keep the output under {max_chars} characters — preserve named "
            f"entities, running arguments, and unresolved threads; "
            f"drop settled details. If the previous summary was truncated "
            f"to fit the context window, prioritize newly visible content "
            f"when absorbing this page.\n\n"
            f"{_LANG_RULE}\n\n"
            f"{_NO_REASONING}"
        ),
        expected_output=(
            f"Updated running summary, ≤ {max_chars} characters, "
            "language matches source"
        ),
        agent=summarizer,
        context=[format_task],
    )

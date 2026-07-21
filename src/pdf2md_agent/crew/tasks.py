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

from pdf2md_agent.crew.multimodal_patch import patch_add_image_tool

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

# Joined rule text shared by every task that asks the LLM to write verbatim
# Markdown: the formatter factory (and its file-fed sibling) and the
# extractor's task description. The ``chr(60)/chr(62)`` escape around
# ``<think>`` is load-bearing (see AGENTS.md → crew/ → CONVENTIONS); do not
# "refactor" to a literal tag.
_COMMON_TASK_RULES: str = f"{_VERBATIM_RULE}\n\n{_LANG_RULE}\n\n{_NO_REASONING}"

# Joined rule text consumed by the token-budget planner.
TASKS_RULES_TEXT: str = _COMMON_TASK_RULES


def extract_task_intro(page_path: Path) -> str:
    return (
        f"Call your add_image tool with image_url=`{page_path}` to attach "
        f"the rendered page image, then transcribe its full content "
        f"into raw markdown.\n\n"
    )


def _truncate_summary(text: str, max_chars: int) -> str:
    """Return ``text`` trimmed to ``max_chars``, preserving head + tail.

    The middle is dropped so the most recent running context (tail) and the
    high-level topic (head) both survive. A sentinel suffix tells the
    extractor agent that the visible state is incomplete. When
    ``max_chars`` cannot fit the full marker, the text is truncated
    without a marker so the returned string is guaranteed to satisfy
    ``len(result) <= max_chars``.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    full_marker = len(_SUMMARY_TRUNCATION_SUFFIX) + 2
    if max_chars < full_marker:
        return text[:max_chars]
    suffix = _SUMMARY_TRUNCATION_SUFFIX
    budget = max_chars - len(suffix) - 2
    head_budget = budget // 2
    tail_budget = budget - head_budget
    head = text[:head_budget].rstrip()
    tail = text[-tail_budget:].lstrip() if tail_budget > 0 else ""
    if tail:
        return f"{head}\n{suffix}\n{tail}"
    return f"{head}\n{suffix}"


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


def _summary_block(summary: str) -> str:
    return (
        f"Running summary of preceding pages:\n{summary}\n\n"
        if summary.strip()
        else ""
    )


def build_extract_description(
    page_path: Path,
    text_hint: str,
    previous_summary: str,
    *,
    max_summary_chars: int = MAX_SUMMARY_CHARS,
) -> str:
    """Build the exact description string the extract task sends to the LLM.

    Shared between ``make_extract_task`` (which wraps it in a CrewAI Task)
    and the runner's token-budget planner (which needs to estimate the cost
    of the same prompt), guaranteeing the budget never diverges from the
    real payload.
    """
    safe_summary = _truncate_summary(previous_summary, max_summary_chars)
    return (
        f"{_summary_block(safe_summary)}"
        f"{_text_hint_block(text_hint)}"
        f"{extract_task_intro(page_path)}"
        f"{TASKS_RULES_TEXT}"
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
    description = build_extract_description(
        page_path,
        text_hint,
        previous_summary,
        max_summary_chars=max_summary_chars,
    )
    return Task(
        description=description,
        expected_output="Verbatim markdown transcription of the page",
        agent=extractor,
    )


def make_format_task(
    formatter: Agent,
    extract_task: Task,
    *,
    reformat: bool = False,
) -> Task:
    """Create the cleanup task; sees the extractor's output via context."""
    description = (
        "Rewrite the extracted markdown as strict CommonMark. Fix "
        "broken lists, normalize table syntax, strip OCR noise.\n\n"
        f"{_COMMON_TASK_RULES}"
    )
    if reformat:
        description += (
            "\n\nAdditionally drop headers/footers/page numbers as layout "
            "artifacts. All other text must survive verbatim."
        )
    return Task(
        description=description,
        expected_output=(
            "Clean CommonMark markdown of the page, language preserved"
            + (", headers/footers/page numbers removed" if reformat else "")
        ),
        agent=formatter,
        context=[extract_task],
    )


def make_format_task_from_extract_file(
    formatter: Agent,
    extract_path: Path,
) -> Task:
    """Format task fed from a cached ``page_NNNN_extract.txt`` on disk.

    Used by ``--reformat`` mode when the user wants to re-format without
    re-running the extractor. The file's full text is pasted into the
    description as a fenced block — the same seam used by ``_text_hint_block``
    in the extract pipeline — so the runner has no new tool surface to
    maintain.

    Caller is responsible for ensuring the file exists (gate on
    ``cache.has_cached_extract`` first).
    """
    text = extract_path.read_text(encoding="utf-8")
    return Task(
        description=(
            "Rewrite the extracted markdown below as strict CommonMark. "
            "Drop running headers, page footers, and page numbers as "
            "layout artifacts. Preserve all other text verbatim.\n\n"
            f"{_COMMON_TASK_RULES}\n\n"
            "Extracted content (read from disk; treat as ground truth):\n"
            "```\n"
            f"{text}\n"
            "```"
        ),
        expected_output=(
            "Clean CommonMark markdown of the page, language preserved, "
            "headers/footers/page numbers removed"
        ),
        agent=formatter,
        context=[],
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

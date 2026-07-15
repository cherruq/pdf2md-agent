"""Agent factories for the per-page CrewAI pipeline."""
from __future__ import annotations

from crewai import Agent, LLM


def make_extractor(llm: LLM) -> Agent:
    """Build the multimodal page-extraction agent."""
    return Agent(
        role="PDF Page Extractor",
        goal=(
            "Transcribe every readable element of a PDF page image into "
            "raw markdown: headings, paragraphs, lists, tables, and "
            "descriptions of every non-text figure."
        ),
        backstory=(
            "You are a meticulous document analyst. You read PDF page "
            "images and produce faithful, VERBATIM transcriptions: you "
            "copy exactly what is on the page, character for character, "
            "in the SAME language as the source document (Chinese text "
            "stays Chinese, English stays English, mixed-language pages "
            "preserve each language). You never translate, summarize, "
            "paraphrase, or invent content. If you cannot read a glyph, "
            "you write `[illegible]` instead of guessing. For schematic "
            "figures you write a short alt-style description prefixed "
            "with `![...](...)` based only on what is visibly drawn."
        ),
        multimodal=True,
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def make_formatter(llm: LLM) -> Agent:
    """Build the agent that cleans extracted markdown into strict CommonMark."""
    return Agent(
        role="Markdown Formatter",
        goal=(
            "Take raw OCR-style markdown and rewrite it as strict "
            "CommonMark: normalize table syntax, fix broken lists, "
            "collapse noise, keep meaning intact."
        ),
        backstory=(
            "You are a careful editor. You receive a rough markdown "
            "transcription and return a clean CommonMark version that "
            "renders identically in any compliant viewer. You NEVER "
            "drop, translate, or rewrite content: every word, every "
            "CJK character, every punctuation mark must survive into "
            "the output. The language of your output must exactly match "
            "the language of your input. You only normalize formatting."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def make_summarizer(llm: LLM) -> Agent:
    """Build the agent that maintains the running cross-page summary."""
    return Agent(
        role="Running Summary Keeper",
        goal=(
            "Maintain a tight rolling summary (~200 words) of a long PDF "
            "as pages are processed, so the next page's extractor has "
            "context about what came before."
        ),
        backstory=(
            "You read the previous summary plus the current page's "
            "markdown and return an updated summary. You write in the "
            "SAME language as the page content (Chinese pages → "
            "Chinese summary, English pages → English summary, mixed "
            "→ pick the dominant language). You preserve named entities, "
            "running arguments, and threads that are still evolving. "
            "You drop details already settled."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
"""Agent factories for the per-page CrewAI pipeline.

Personas are intentionally short (well under 60 words each) so they fit
comfortably inside the ``MiniMax-M3`` ~2 k-token context window alongside
the per-page task description, the running summary, the text-hint, and
the base64-encoded page image. The persona strings are exported as
``EXTRACTOR_PERSONA`` / ``FORMATTER_PERSONA_STRICT`` /
``FORMATTER_PERSONA_REFORMAT`` / ``SUMMARIZER_PERSONA`` so
the runner can budget their token cost before issuing each call.
"""
from __future__ import annotations

from crewai import Agent, LLM


EXTRACTOR_PERSONA: str = (
    "PDF Page Extractor. "
    "Transcribe every readable element of the page image into raw markdown "
    "(headings, paragraphs, lists, tables, alt-text for figures), "
    "preserving source language verbatim."
    "\n\n"
    "You transcribe a PDF page image character-for-character into "
    "markdown, preserving the source language(s) without translation, "
    "summarization, or invention. Use `[illegible]` for unreadable glyphs "
    "and prefix short alt descriptions for non-text figures with `![...]()`. "
    "Preserve CJK characters, punctuation, layout, and mixed-language "
    "content exactly as drawn."
)

FORMATTER_PERSONA_STRICT: str = (
    "Markdown Formatter. "
    "Rewrite extracted markdown as strict CommonMark: normalize tables, "
    "fix lists, strip OCR noise; preserve every word verbatim.\n\n"
    "You rewrite OCR-style markdown as strict CommonMark: normalize table "
    "syntax, fix broken lists, strip OCR noise. Never drop, translate, or "
    "rewrite content — every word, CJK character, and punctuation mark from "
    "the input must survive verbatim. Only normalize formatting; output "
    "language must exactly match input."
)

FORMATTER_PERSONA_REFORMAT: str = (
    "Markdown Formatter (Layout-Aware). "
    "Rewrite extracted markdown as strict CommonMark and drop repeating "
    "page-level layout artifacts (running headers, document titles "
    "repeated at the top, copyright / license / print notices at the "
    "bottom, isolated or 'N / M' style page numbers, page-footer URLs); "
    "preserve all body content verbatim.\n\n"
    "You rewrite extracted markdown as strict CommonMark. Treat repeating "
    "page headers, page footers, and page numbers as layout artifacts "
    "and omit them. Preserve every other word, CJK character, and "
    "punctuation mark from the input verbatim. Do not translate, "
    "summarize, or rewrite body content."
)

SUMMARIZER_PERSONA: str = (
    "Running Summary Keeper. "
    "Maintain a tight rolling summary of preceding pages so the next "
    "extractor has cross-page context."
    "\n\n"
    "You maintain a tight rolling cross-page summary from prior summary + "
    "current page. Preserve named entities, unresolved threads, and "
    "arguments still evolving; drop settled details. Write in the dominant "
    "source language. If the prior summary was truncated to fit the context "
    "window, prioritize absorbing newly visible content."
)


def make_extractor(llm: LLM) -> Agent:
    """Build the multimodal page-extraction agent."""
    return Agent(
        role="PDF Page Extractor",
        goal=(
            "Transcribe the page image into raw markdown (headings, "
            "paragraphs, lists, tables, alt-text for figures), preserving "
            "source language verbatim."
        ),
        backstory=_persona_backstory(EXTRACTOR_PERSONA),
        multimodal=True,
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def make_formatter(llm: LLM, *, reformat: bool = False) -> Agent:
    """Build the agent that cleans extracted markdown into strict CommonMark.

    When ``reformat`` is True the agent uses a layout-aware persona that
    drops page headers, footers, and page numbers in addition to the
    CommonMark normalization.
    """
    if reformat:
        return Agent(
            role="Markdown Formatter (Layout-Aware)",
            goal=(
                "Rewrite extracted markdown as strict CommonMark, "
                "dropping running headers, page footers, and page "
                "numbers — preserve every other word verbatim."
            ),
            backstory=_persona_backstory(FORMATTER_PERSONA_REFORMAT),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )
    return Agent(
        role="Markdown Formatter",
        goal=(
            "Rewrite extracted markdown as strict CommonMark — preserve "
            "every word verbatim."
        ),
        backstory=_persona_backstory(FORMATTER_PERSONA_STRICT),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def make_summarizer(llm: LLM) -> Agent:
    """Build the agent that maintains the running cross-page summary."""
    return Agent(
        role="Running Summary Keeper",
        goal=(
            "Maintain a tight rolling summary of preceding pages so the "
            "next extractor has cross-page context."
        ),
        backstory=_persona_backstory(SUMMARIZER_PERSONA),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def _persona_backstory(persona: str) -> str:
    """CrewAI's ``backstory`` only sees text after the ``\\n\\n`` separator."""
    if "\n\n" in persona:
        _, _, backstory = persona.partition("\n\n")
        return backstory.strip()
    return persona


EXTRACTOR_BACKSTORY: str = _persona_backstory(EXTRACTOR_PERSONA)
FORMATTER_BACKSTORY: str = _persona_backstory(FORMATTER_PERSONA_STRICT)
SUMMARIZER_BACKSTORY: str = _persona_backstory(SUMMARIZER_PERSONA)

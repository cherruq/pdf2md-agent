"""Agent factories for the per-page CrewAI pipeline.

Personas are intentionally short so they fit comfortably inside the model's context window.
The persona strings are exported as ``EXTRACTOR_PERSONA`` so the runner can budget their token cost before issuing each call.
"""

from __future__ import annotations

from crewai import Agent, LLM


EXTRACTOR_PERSONA: str = (
    "PDF Page Extractor. "
    "Transcribe every readable element of the page image into strict "
    "CommonMark markdown (headings, paragraphs, lists, tables, alt-text for "
    "figures), preserving source language verbatim."
    "\n\n"
    "You transcribe a PDF page image into clean, strict CommonMark markdown, "
    "preserving the source language(s) without translation, summarization, or "
    "invention. Normalize table syntax, fix broken lists, and strip OCR noise. "
    "Use `[illegible]` for unreadable glyphs and prefix short alt descriptions "
    "for non-text figures with `![...]()`. Preserve CJK characters, punctuation, "
    "and layout exactly."
)


def make_extractor(llm: LLM) -> Agent:
    """Build the multimodal page-extraction agent."""
    return Agent(
        role="PDF Page Extractor",
        goal=(
            "Transcribe the page image into strict CommonMark markdown "
            "(headings, paragraphs, lists, tables, alt-text for figures), "
            "preserving source language verbatim."
        ),
        backstory=_persona_backstory(EXTRACTOR_PERSONA),
        multimodal=True,
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


__all__ = [
    "EXTRACTOR_BACKSTORY",
    "EXTRACTOR_PERSONA",
    "make_extractor",
]

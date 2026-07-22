"""Agent factories for the per-page CrewAI pipeline.

Personas are intentionally short (well under 60 words each) so they fit
comfortably inside the ``MiniMax-M3`` ~2 k-token context window alongside
the per-page task description, the running summary, the text-hint, and
the base64-encoded page image. The persona strings are exported as
``EXTRACTOR_PERSONA`` / ``FORMATTER_PERSONA_STRICT`` /
``SUMMARIZER_PERSONA`` so the runner can budget their token cost before
issuing each call.

``PERSONA_VERSION`` is a 16-char SHA-256 digest of the joined persona
strings; the runner records it in ``meta.json`` so a follow-up run
detects persona-text drift and refuses to re-use stale cached outputs.
"""
from __future__ import annotations

import hashlib

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


PERSONA_VERSION: str = hashlib.sha256(
    "\x00".join(
        (EXTRACTOR_PERSONA, FORMATTER_PERSONA_STRICT, SUMMARIZER_PERSONA)
    ).encode("utf-8")
).hexdigest()[:16]
"""SHA-256[:16] of the active persona strings. Fingerprint recorded in
``meta.json`` so a follow-up run detects text drift in any persona."""


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


def make_formatter(llm: LLM) -> Agent:
    """Build the agent that cleans extracted markdown into strict CommonMark.

    Strict CommonMark is the only formatter persona; the prior layout-aware
    variant was removed when the path-B cache rename dropped ``--reformat``.
    """
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


__all__ = [
    "EXTRACTOR_BACKSTORY",
    "EXTRACTOR_PERSONA",
    "FORMATTER_PERSONA_STRICT",
    "PERSONA_VERSION",
    "SUMMARIZER_PERSONA",
    "make_extractor",
    "make_formatter",
    "make_summarizer",
]

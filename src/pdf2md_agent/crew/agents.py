"""针对逐页 CrewAI 流水线的 Agent 工厂。

人设（Personas）故意做得很短，以便能够舒适地放入模型的上下文窗口内。
人设字符串导出为 ``EXTRACTOR_PERSONA``，以便 runner 可以在发出每次调用之前预算它们的 token 成本。
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
    """构建多模态页面提取 agent。"""
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
    """CrewAI 的 ``backstory`` 只能看到 ``\\n\\n`` 分隔符之后的文本。"""
    if "\n\n" in persona:
        _, _, backstory = persona.partition("\n\n")
        return backstory.strip()
    return persona


EXTRACTOR_BACKSTORY: str = _persona_backstory(EXTRACTOR_PERSONA)


__all__ = [
    "EXTRACTOR_BACKSTORY",
    "make_extractor",
]

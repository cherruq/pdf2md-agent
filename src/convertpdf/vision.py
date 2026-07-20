"""LLM factory: CrewAI LLM preconfigured for MiniMax-M3 vision calls."""
from __future__ import annotations

from crewai import LLM

from convertpdf.config import MODEL_NAME, OPENAI_BASE_URL, require_api_key


def make_vision_llm() -> LLM:
    """Return a CrewAI ``LLM`` pointed at the MiniMax-M3 vision endpoint.

    Uses the native OpenAI provider (``provider="openai"``) so the request is
    dispatched directly via the OpenAI SDK against a custom ``base_url`` — no
    LiteLLM dependency required.
    """
    return LLM(
        model=MODEL_NAME,
        provider="openai",
        base_url=OPENAI_BASE_URL,
        api_key=require_api_key(),
    )
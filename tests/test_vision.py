"""Tests for pdf2md_agent.vision (LLM factory wiring)."""
from __future__ import annotations

import pytest

from pdf2md_agent import config
from pdf2md_agent.vision import make_vision_llm


def test_make_vision_llm_uses_minimax_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OPENAI_BASE_URL", "https://api.minimaxi.com/v1")
    monkeypatch.setattr(config, "MODEL_NAME", "MiniMax-M3")

    llm = make_vision_llm()

    assert "MiniMax-M3" in llm.model
    assert llm.base_url == "https://api.minimaxi.com/v1"
    assert llm.api_key  # non-empty


def test_make_vision_llm_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "_env", lambda name, default="": "")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        make_vision_llm()
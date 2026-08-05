"""LLM 工厂：为 MiniMax-M3 视觉调用预配置的 CrewAI LLM。"""

from __future__ import annotations

from crewai import LLM

from pdf2md_agent.config import (
    MODEL_NAME,
    OPENAI_BASE_URL,
    REQUEST_TIMEOUT_SECONDS,
    require_api_key,
)


def make_vision_llm() -> LLM:
    """返回一个指向 MiniMax-M3 视觉端点的 CrewAI ``LLM``。

    使用原生的 OpenAI provider (``provider="openai"``)，因此请求直接
    通过 OpenAI SDK 发送到自定义的 ``base_url`` —— 不需要 LiteLLM 依赖。
    ``timeout`` 被传递到底层 SDK；它表现为初始连接和主体读取两者的
    每次请求的套接字级截止时间。
    """
    return LLM(
        model=MODEL_NAME,
        provider="openai",
        base_url=OPENAI_BASE_URL,
        api_key=require_api_key(),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

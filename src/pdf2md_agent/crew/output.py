"""从 CrewAI 任务输出中提取纯文本的辅助函数。

CrewAI 的任务输出是一个小型的类似 ``TaskOutput`` 的对象，其
``output.raw`` 字段是模型的原始文本。此模块隔离了两个
小关注点：

* :func:`_strip_think` — 防御性地删除配置的 MiniMax-M3 端点偶尔泄漏
  到其回复中的任何 ``<think>…</think>`` 暂存块。
* :func:`_output` — 从任务的输出对象中提取 ``raw`` 字段，
  强制转换为 ``str`` 并在一个地方剥离思考块（think blocks）。
"""

from __future__ import annotations

import re


_THINK_OPEN = chr(60) + "think" + chr(62)
_THINK_CLOSE = chr(60) + "/think" + chr(62)
_THINK_BLOCK_RE = re.compile(_THINK_OPEN + r".*?" + _THINK_CLOSE, re.DOTALL)


def _strip_think(text: str) -> str:
    """从输出中删除内联的模型推理块。

    一些模型会将它们的暂存过程包装在推理标签中；配置的
    MiniMax-M3 端点有时会将它们留在响应中。
    在下游消费者看到它们之前，防御性地将它们剥离。
    """
    return _THINK_BLOCK_RE.sub("", text).strip()


def _output(output_text: object) -> str:
    """从 CrewAI 任务的输出中提取纯文本。

    处理真实的 ``TaskOutput`` 形状（带有嵌套的 ``output`` 和
    ``.raw``）以及纯 ``str`` 结果（由测试返回）。缺失或
    ``None`` 的输出会降级为空字符串而不是抛出异常。
    """
    out = getattr(output_text, "output", None)
    if out is None:
        return ""
    raw = getattr(out, "raw", None)
    text = raw if isinstance(raw, str) else str(out)
    return _strip_think(text)


__all__ = ["_strip_think", "_output"]

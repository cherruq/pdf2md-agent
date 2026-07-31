"""Helpers for extracting clean text from CrewAI task output.

The crewAI task output is a small ``TaskOutput``-shaped object whose
``output.raw`` field is the model's raw text. This module isolates two
small concerns:

* :func:`_strip_think` — defensively remove any ``<think>…</think>``
  scratchpad blocks the configured MiniMax-M3 endpoint occasionally leaks
  into its replies.
* :func:`_output` — pull the ``raw`` field off a task's output object,
  coercing to ``str`` and stripping think blocks in one place.
"""

from __future__ import annotations

import re


_THINK_OPEN = chr(60) + "think" + chr(62)
_THINK_CLOSE = chr(60) + "/think" + chr(62)
_THINK_BLOCK_RE = re.compile(_THINK_OPEN + r".*?" + _THINK_CLOSE, re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove inline model reasoning blocks from output.

    Some models wrap their scratchpad in reasoning tags; the configured
    MiniMax-M3 endpoint sometimes leaves them in the response. Strip them
    defensively before downstream consumers see them.
    """
    return _THINK_BLOCK_RE.sub("", text).strip()


def _output(output_text: object) -> str:
    """Extract clean text from a CrewAI task's output.

    Handles both real ``TaskOutput`` shapes (with a nested ``output`` and
    ``.raw``) and bare ``str`` results (returned by tests). Missing or
    ``None`` outputs degrade to an empty string instead of raising.
    """
    out = getattr(output_text, "output", None)
    if out is None:
        return ""
    raw = getattr(out, "raw", None)
    text = raw if isinstance(raw, str) else str(out)
    return _strip_think(text)


__all__ = ["_strip_think", "_output"]

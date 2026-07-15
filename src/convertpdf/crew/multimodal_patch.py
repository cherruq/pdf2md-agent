"""Monkey-patch CrewAI's AddImageTool to (a) inline local files as data URLs
and (b) return a ``VISION_IMAGE`` sentinel string instead of a dict.

Two bugs in stock ``AddImageTool``:

1. ``_run(image_url=...)`` forwards the URL verbatim. OpenAI-compatible vision
   APIs reject local paths with HTTP 400 and treat the missing image as
   "no image attached", causing the model to hallucinate the page contents.

2. ``_run`` returns a ``dict`` like
   ``{"role": "user", "content": [{"type": "image_url", ...}]}``. CrewAI's
   ``StepExecutor._build_observation_message`` only knows how to build a
   multimodal content block from a tool result that is the **string**
   ``VISION_IMAGE:<media_type>:<base64>``. Dict/stringified results become a
   plain text ``Observation: <dict>`` message and the image block is lost.

We replace ``_run`` so it returns the proper sentinel string after encoding
local files inline. ``patch_add_image_tool()`` is idempotent.
"""
from __future__ import annotations

import base64
from pathlib import Path


def _to_data_url(value: str) -> str:
    """Inline a local file as a ``data:image/...;base64,...`` URL."""
    if not value or value.startswith(("http://", "https://", "data:")):
        return value
    path = Path(value)
    if not path.is_file():
        return value
    suffix = path.suffix.lstrip(".").lower() or "png"
    mime = "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix}"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _to_sentinel(value: str, action: str | None) -> str:
    """Return the ``VISION_IMAGE:...`` sentinel the step executor recognizes.

    The sentinel format is ``VISION_IMAGE:<media_type>:<base64_data>``.
    We also keep the agent's optional action text in the same string so the
    executor's text fallback still surfaces it to the model.
    """
    url = _to_data_url(value)
    if not url.startswith("data:"):
        # Not an inline image — fall back to plain text the model can read.
        return action or f"(could not inline image at {value})"
    head, b64 = url.split(",", 1)
    # head == "data:image/png;base64"
    media_type = head[len("data:") :].split(";", 1)[0]
    if action:
        return f"{action}\nVISION_IMAGE:{media_type}:{b64}"
    return f"VISION_IMAGE:{media_type}:{b64}"


_patched = False


def patch_add_image_tool() -> None:
    """Wrap ``AddImageTool._run`` so local paths inline and return a sentinel."""
    global _patched
    if _patched:
        return
    from crewai.tools.agent_tools.add_image_tool import AddImageTool

    def _run(self, image_url: str, action=None, **kwargs):  # type: ignore[override]
        return _to_sentinel(image_url, action)

    AddImageTool._run = _run  # type: ignore[assignment]
    _patched = True
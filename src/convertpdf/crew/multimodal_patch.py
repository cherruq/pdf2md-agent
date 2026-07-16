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

To stay inside ``MiniMax-M3``'s ~2 k-token context budget we also re-encode
every local image: long-side capped via ``img.thumbnail(..., LANCZOS)``,
converted to RGB, written as JPEG (``optimize=True``), then base64-encoded.
The cap and quality are configurable per-call by the runner, defaulting to
1536 px / quality 85.

We replace ``_run`` so it returns the proper sentinel string after encoding
local files inline. ``patch_add_image_tool()`` is idempotent.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path


_DEFAULT_TARGET_LONG_SIDE: int = 1536
_DEFAULT_JPEG_QUALITY: int = 85


def _encode_local_image(
    path: Path,
    *,
    target_long_side: int,
    jpeg_quality: int,
) -> bytes:
    """Open ``path`` with Pillow, downscale, return the JPEG bytes.

    Raises ``FileNotFoundError`` if Pillow cannot open the file. The
    runner pre-builds a downsized copy under ``layout.pages_dir`` so this
    function almost never has to do real work, but it remains correct for
    arbitrary inputs in tests.
    """
    from PIL import Image  # local import — Pillow is a hard project dep

    with Image.open(path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        if target_long_side > 0:
            img.thumbnail((target_long_side, target_long_side), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=jpeg_quality, optimize=True)
        return buf.getvalue()


def _to_data_url(
    value: str,
    *,
    target_long_side: int = _DEFAULT_TARGET_LONG_SIDE,
    jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
) -> str:
    """Inline a local file as a ``data:image/jpeg;base64,...`` data URL.

    URLs (``http://``, ``https://``, ``data:``) and paths that don't resolve
    to a real file are returned unchanged. Local images are downscaled in
    memory via Pillow's LANCZOS thumbnail and re-encoded as JPEG at the
    requested quality before base64 encoding — the resulting ``data:`` URL
    is small enough to stay inside the ``MiniMax-M3`` context window.
    """
    if not value or value.startswith(("http://", "https://", "data:")):
        return value
    path = Path(value)
    if not path.is_file():
        return value
    try:
        encoded = _encode_local_image(
            path,
            target_long_side=target_long_side,
            jpeg_quality=jpeg_quality,
        )
    except FileNotFoundError:
        return value
    b64 = base64.b64encode(encoded).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _to_sentinel(
    value: str,
    action: str | None,
    *,
    target_long_side: int = _DEFAULT_TARGET_LONG_SIDE,
    jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
) -> str:
    """Return the ``VISION_IMAGE:...`` sentinel the step executor recognizes.

    The sentinel format is ``VISION_IMAGE:<media_type>:<base64_data>``.
    We also keep the agent's optional action text in the same string so the
    executor's text fallback still surfaces it to the model.
    """
    url = _to_data_url(
        value,
        target_long_side=target_long_side,
        jpeg_quality=jpeg_quality,
    )
    if not url.startswith("data:"):
        return action or f"(could not inline image at {value})"
    head, b64 = url.split(",", 1)
    media_type = head[len("data:") :].split(";", 1)[0]
    if action:
        return f"{action}\nVISION_IMAGE:{media_type}:{b64}"
    return f"VISION_IMAGE:{media_type}:{b64}"


_patched = False


def patch_add_image_tool(
    *,
    target_long_side: int = _DEFAULT_TARGET_LONG_SIDE,
    jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
) -> None:
    """Wrap ``AddImageTool._run`` so local paths inline and return a sentinel.

    Idempotent on repeated calls: the *first* call fixes the resize/quality
    knobs and subsequent calls are ignored. The runner relies on this — it
    patches once at module import time and the values used for the actual
    downscaling live in the pre-resized cache files, not in the patch.
    """
    global _patched
    if _patched:
        return
    from crewai.tools.agent_tools.add_image_tool import AddImageTool

    def _run(self, image_url: str, action=None, **kwargs):  # type: ignore[override]
        return _to_sentinel(
            image_url,
            action,
            target_long_side=target_long_side,
            jpeg_quality=jpeg_quality,
        )

    AddImageTool._run = _run  # type: ignore[assignment]
    _patched = True

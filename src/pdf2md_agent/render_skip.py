"""Render-side cache reuse helpers (PNG / text / resized)."""
from __future__ import annotations

from pathlib import Path

from pdf2md_agent.cache import CacheLayout


def maybe_skip_render(
    layout: CacheLayout, page_number: int, dpi: int
) -> Path | None:
    """Return the cached PNG path if it already exists at the matching DPI.

    The helper inspects the on-disk PNG: a sidecar ``page_NNNN.meta.json``
    file is checked (when present) for a ``dpi`` field; without the
    sidecar, the helper assumes the existing PNG was rendered at the
    requested ``dpi`` (legacy cache directories written before the
    sidecar landed).
    """
    png = layout.page_png_path(page_number)
    if not png.is_file():
        return None
    sidecar = png.with_name(f"{png.stem}.meta.json")
    if sidecar.is_file():
        try:
            import json

            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        cached_dpi = payload.get("dpi") if isinstance(payload, dict) else None
        if isinstance(cached_dpi, int) and cached_dpi != dpi:
            return None
    return png


def maybe_skip_resized(
    layout: CacheLayout, page_number: int, needed_long_side: int
) -> Path | None:
    """Return the cached downscaled JPEG if it matches the needed long side.

    The resize cache is keyed on the rendered long side; a mismatch means
    the cached JPEG would not satisfy the current token-budget planner and
    must be regenerated.
    """
    resized = layout.pages_dir / f"page_{page_number:04d}_resized.jpg"
    if not resized.is_file():
        return None
    sidecar = resized.with_name(f"{resized.stem}.meta.json")
    if sidecar.is_file():
        try:
            import json

            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        cached_long = payload.get("long_side") if isinstance(payload, dict) else None
        if isinstance(cached_long, int) and cached_long != needed_long_side:
            return None
    return resized

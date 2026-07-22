"""Probe an OpenAI-compatible ``/v1/models`` endpoint for the model's
context-window token budget.

The OpenAI base spec for ``GET /v1/models`` only returns ``id``,
``created``, ``object`` and ``owned_by`` per entry. Real providers extend
that schema with their own context-window field — the field name varies
across vendors (``context_window``, ``max_context_tokens``,
``max_input_tokens``, ``context_length``, ``max_tokens``,
``max_sequence_length``). This module tolerates every common spelling and
returns the first positive integer it finds.

Every failure mode (DNS, HTTP error, timeout, malformed JSON, missing
model) returns ``None`` so the caller can fall back to a hardcoded
default without further branching.
"""
from __future__ import annotations

import json
import logging
import urllib.error
from typing import Any, Final
from urllib.request import Request, urlopen


log = logging.getLogger("pdf2md_agent.ctx_probe")


_MAX_CTX_LIMIT: Final[int] = 1_048_576  # 1M, the published MiniMax-M3 ceiling.


_CTX_FIELD_CANDIDATES: Final[tuple[str, ...]] = (
    "context_window",
    "max_context_tokens",
    "max_input_tokens",
    "context_length",
    "max_tokens",
    "max_sequence_length",
)


def _extract_ctx(entry: dict[str, Any]) -> int | None:
    """Return the first positive integer among the candidate field names."""
    for field in _CTX_FIELD_CANDIDATES:
        value = entry.get(field)
        if isinstance(value, bool):
            # ``bool`` is an ``int`` subclass; guard explicitly so ``True``
            # (== 1) is never misread as a context window.
            continue
        if isinstance(value, (int, float)) and value > 0:
            return min(int(value), _MAX_CTX_LIMIT)
    return None


def probe_ctx_limit(
    base_url: str,
    api_key: str,
    model: str,
    *,
    timeout: float = 10.0,
) -> int | None:
    """Return the model's context-window token budget, or ``None``.

    Tries the configured model's exact ``id`` first, then falls back to a
    case-insensitive substring match (some providers expose multiple dated
    snapshots of the same model under different ids). Any network or
    parse error returns ``None`` — the caller's fallback chain is the
    source of truth.
    """
    url = base_url.rstrip("/") + "/models"
    req = Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        log.debug("probe_ctx_limit: %s returned %s", url, exc)
        return None

    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return None

    target = model.lower()
    # Pass 1: exact, case-insensitive match.
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("id", "")).lower() == target:
            value = _extract_ctx(entry)
            if value is not None:
                return value
    # Pass 2: substring fallback for providers that snapshot under dated ids.
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if target in str(entry.get("id", "")).lower():
            value = _extract_ctx(entry)
            if value is not None:
                return value
    return None
"""Pure-function LLM calls — no CrewAI state, no message-history accumulation.

Why raw OpenAI SDK instead of CrewAI? CrewAI's ``_invoke_loop_native_tools``
is a think→action→observe→answer loop. Each turn re-packages the entire
message history (system + user + assistant + tool_result) and sends it again.
The extractor agent therefore emits multiple HTTP requests per ``crew.kickoff()``,
and the inline image (base64 JPEG, ~2.5–5 k tokens at 1536 px) is included
in *every* turn. This made the planner's prompt-token estimate diverge from
the actual sent payload — the planner saw one image, but the model saw two
or three (the "doubled context" trap).

Going direct with the OpenAI SDK means each LLM call is exactly one HTTP
request with exactly one image inline (or zero images for text-only calls).
What the planner budgets is what the wire carries — no hidden doubling,
no tool-result re-injection, no message-history recursion.

Public surface:

- ``EXTRACTOR_PERSONA`` / ``FORMATTER_PERSONA_STRICT`` / ``SUMMARIZER_PERSONA``
  — short persona strings budgeted into the token estimate so planner ==
  actual.
- ``PERSONA_VERSION`` — SHA-256[:16] of the joined personas; recorded in
  ``meta.json`` so a follow-up run detects persona-text drift.
- ``ImageEncodeError`` — raised when a local page image cannot be opened /
  decoded / re-encoded (caller falls back to text layer).
- ``_make_client()`` — single ``openai.OpenAI`` client (callers can patch
  this for tests).
- ``call_extractor(client, *, image_path, text_hint, previous_summary, ...)``
  — one multimodal extraction request; returns the model's raw markdown.
- ``call_formatter(client, *, extract_text, ...)`` — one text-only formatter
  request; the on-disk extract is inlined as a fenced block in the user
  message and head+tail-truncated to the input budget.
- ``call_summarizer(client, *, format_text, previous_summary, ...)`` — one
  text-only summarizer request; returns the updated running summary.
- ``_strip_think(text)`` — defensive removal of ``...`` blocks
  the configured MiniMax-M3 endpoint sometimes emits.

Each ``call_*`` is a pure function: the caller's input arguments are the
entire visible state. There is no hidden CrewAI agent to leak context from
the previous page.
"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import re
from pathlib import Path

from openai import OpenAI
from PIL import Image

from pdf2md_agent.config import MODEL_NAME, OPENAI_BASE_URL, require_api_key

log = logging.getLogger("pdf2md_agent.raw_pipeline")


# ---------------------------------------------------------------------------
# Persona strings
# ---------------------------------------------------------------------------
# Short (well under 60 words each) so they fit comfortably alongside the
# page image and per-page description inside MiniMax-M3's ~2 k-token
# context window. ``PERSONA_VERSION`` is the 16-char SHA-256 of the joined
# strings; recorded in ``meta.json`` to detect drift.

EXTRACTOR_PERSONA: str = (
    "PDF Page Extractor. "
    "Transcribe every readable element of the page image into raw markdown "
    "(headings, paragraphs, lists, tables, alt-text for figures), "
    "preserving source language verbatim."
    "\n\n"
    "You transcribe a PDF page image character-for-character into "
    "markdown, preserving the source language(s) without translation, "
    "summarization, or invention. Use `[illegible]` for unreadable glyphs "
    "and prefix short alt descriptions for non-text figures with `![...]()`. "
    "Preserve CJK characters, punctuation, layout, and mixed-language "
    "content exactly as drawn."
)

FORMATTER_PERSONA_STRICT: str = (
    "Markdown Formatter. "
    "Rewrite extracted markdown as strict CommonMark: normalize tables, "
    "fix lists, strip OCR noise; preserve every word verbatim.\n\n"
    "You rewrite OCR-style markdown as strict CommonMark: normalize table "
    "syntax, fix broken lists, strip OCR noise. Never drop, translate, or "
    "rewrite content — every word, CJK character, and punctuation mark from "
    "the input must survive verbatim. Only normalize formatting; output "
    "language must exactly match input."
)

SUMMARIZER_PERSONA: str = (
    "Running Summary Keeper. "
    "Maintain a tight rolling summary of preceding pages so the next "
    "extractor has cross-page context."
    "\n\n"
    "You maintain a tight rolling cross-page summary from prior summary + "
    "current page. Preserve named entities, unresolved threads, and "
    "arguments still evolving; drop settled details. Write in the dominant "
    "source language. If the prior summary was truncated to fit the context "
    "window, prioritize absorbing newly visible content."
)

PERSONA_VERSION: str = hashlib.sha256(
    "\x00".join(
        (EXTRACTOR_PERSONA, FORMATTER_PERSONA_STRICT, SUMMARIZER_PERSONA)
    ).encode("utf-8")
).hexdigest()[:16]
"""SHA-256[:16] of the active persona strings. Fingerprint recorded in
``meta.json`` so a follow-up run detects text drift in any persona."""


# ---------------------------------------------------------------------------
# Defensive tag stripping
# ---------------------------------------------------------------------------
# ``chr(60) / chr(62)`` escaping (rather than literal ``<`` / ``>``) is
# load-bearing: some XML-processing tools mangle the literal angle brackets
# during copy/paste, which would silently break the regex. See
# ``AGENTS.md → crew/ → CONVENTIONS``.

_THINK_OPEN = chr(60) + "think" + chr(62)
_THINK_CLOSE = chr(60) + "/think" + chr(62)
_THINK_BLOCK_RE = re.compile(_THINK_OPEN + r".*?" + _THINK_CLOSE, re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove inline model reasoning blocks from ``text``.

    Some models wrap their scratchpad in reasoning tags; the configured
    MiniMax-M3 endpoint sometimes leaves them in the response. Strip them
    defensively before downstream consumers see them.
    """
    return _THINK_BLOCK_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Rule constants
# ---------------------------------------------------------------------------
# Shared by every task that asks the LLM to write verbatim Markdown. The
# ``_NO_REASONING`` rule keeps the chr-escaped think-tag phrase so the
# in-prompt instruction is consistent with the defensive strip above.

_NO_REASONING = (
    "Output ONLY final content; no reasoning, preamble, or "
    + chr(60) + "think" + chr(62) + "..."
    + chr(60) + "/think" + chr(62) + " blocks."
)

_LANG_RULE = (
    "Language rule: write in the exact same language(s) as the source — "
    "preserve every CJK character, Latin word, and punctuation mark; "
    "never translate."
)

_VERBATIM_RULE = (
    "Verbatim rule: copy the page character-for-character — no "
    "translation, summarization, or invented content; write `[illegible]` "
    "for unreadable glyphs."
)

_COMMON_TASK_RULES: str = f"{_VERBATIM_RULE}\n\n{_LANG_RULE}\n\n{_NO_REASONING}"


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------
# Lifted verbatim from the old ``crew/multimodal_patch.py`` ``_encode_local_image``
# utility. The Pillow/LANCZOS/JPEG pipeline stays identical so the on-disk
# ``page_NNNN_resized.jpg`` (built by the runner's pre-flight resizer) is
# byte-identical to what we'd encode inline at call time.

try:
    from PIL import UnidentifiedImageError
except ImportError:  # pragma: no cover  defensive — see old multimodal_patch
    UnidentifiedImageError = OSError  # type: ignore[assignment,misc]


class ImageEncodeError(RuntimeError):
    """Raised when a local image cannot be opened / decoded / re-encoded.

    The LLM would otherwise hallucinate the page contents because the
    inline image never made it into the request. Callers (the runner)
    should treat this as a retryable / fallback-able error.
    """


def _encode_local_image(
    path: Path,
    *,
    target_long_side: int,
    jpeg_quality: int,
) -> bytes:
    """Open ``path`` with Pillow, downscale, return the JPEG bytes."""
    try:
        with Image.open(path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            if target_long_side > 0:
                img.thumbnail((target_long_side, target_long_side), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=jpeg_quality, optimize=True)
            return buf.getvalue()
    except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
        log.warning("_encode_local_image: cannot encode %s: %s", path, exc)
        raise ImageEncodeError(f"cannot encode image {path}: {exc}") from exc


def _image_to_data_url(
    path: Path,
    *,
    target_long_side: int,
    jpeg_quality: int,
) -> str:
    """Inline a downscaled local image as ``data:image/jpeg;base64,...``."""
    encoded = _encode_local_image(
        path,
        target_long_side=target_long_side,
        jpeg_quality=jpeg_quality,
    )
    b64 = base64.b64encode(encoded).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# Summary truncation (head+tail, with sentinel)
# ---------------------------------------------------------------------------

_SUMMARY_TRUNCATION_SUFFIX = "[…summary truncated to fit context window]"


def _truncate_summary(text: str, max_chars: int) -> str:
    """Return ``text`` trimmed to ``max_chars``, preserving head + tail.

    The middle is dropped so the most recent running context (tail) and the
    high-level topic (head) both survive. A sentinel suffix tells the
    extractor that the visible state is incomplete. When ``max_chars``
    cannot fit the full marker, the text is truncated without a marker so
    the returned string is guaranteed to satisfy ``len(result) <= max_chars``.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    full_marker = len(_SUMMARY_TRUNCATION_SUFFIX) + 2
    if max_chars < full_marker:
        return text[:max_chars]
    suffix = _SUMMARY_TRUNCATION_SUFFIX
    budget = max_chars - len(suffix) - 2
    head_budget = budget // 2
    tail_budget = budget - head_budget
    head = text[:head_budget].rstrip()
    tail = text[-tail_budget:].lstrip() if tail_budget > 0 else ""
    if tail:
        return f"{head}\n{suffix}\n{tail}"
    return f"{head}\n{suffix}"


# ---------------------------------------------------------------------------
# Per-call budget + content builders
# ---------------------------------------------------------------------------
# The formatter default input budget. Roughly tokens; we treat it as a
# character cap (``* 4`` chars/token) passed to ``_truncate_summary``. The
# estimator in ``token_budget.py`` validates the exact count before each
# extract call.
DEFAULT_FORMATTER_INPUT_TOKENS: int = 8000


def _summary_block(summary: str) -> str:
    return (
        f"Running summary of preceding pages:\n{summary}\n\n"
        if summary.strip()
        else ""
    )


def _text_hint_block(text: str) -> str:
    """Build the text-hint block, or empty string when ``text`` is blank."""
    text = text.strip()
    if not text:
        return ""
    return (
        "Text-hint (extracted from the PDF's native text layer — treat as "
        "ground truth for prose, numbers, units, formula symbols, table "
        "cell content; use the image for layout, figures, and visual "
        "structure. If they disagree on order or wording, follow the image "
        "for structure and the text for exact wording):\n"
        "```\n"
        f"{text}\n"
        "```\n\n"
    )


def build_extractor_user_text(
    *,
    text_hint: str,
    previous_summary: str,
    max_summary_chars: int,
) -> str:
    """Build the extractor user message text (image attached separately).

    This is the *text* component of a multimodal user message; the caller
    appends an ``image_url`` content block before sending. Same shape that
    the token-budget planner uses to estimate the request cost.
    """
    safe_summary = _truncate_summary(previous_summary, max_summary_chars)
    return (
        f"{_summary_block(safe_summary)}"
        f"{_text_hint_block(text_hint)}"
        f"{_COMMON_TASK_RULES}"
    )


def build_formatter_user_text(extract_text: str, max_chars: int) -> str:
    """Build the formatter user message: inlined extract content as a fenced block.

    The extract text is head+tail-truncated to ``max_chars`` so the formatter
    never blows the context window on a single oversized page.
    """
    truncated = _truncate_summary(extract_text, max_chars)
    return (
        "Rewrite the extracted markdown below as strict CommonMark. "
        "Preserve every word verbatim — do not drop, translate, or "
        "rewrite content. Only normalize formatting; output language "
        "must exactly match the input.\n\n"
        f"{_COMMON_TASK_RULES}\n\n"
        "Extracted content (treat as ground truth):\n"
        "```\n"
        f"{truncated}\n"
        "```"
    )


def build_summarizer_user_text(
    *,
    format_text: str,
    previous_summary: str,
    max_chars: int,
) -> str:
    """Build the summarizer user message: previous summary + current page format."""
    previous_block = (
        f"Previous running summary:\n{previous_summary}\n\n"
        if previous_summary.strip()
        else "This is the first page; start a fresh summary.\n\n"
    )
    return (
        f"{previous_block}"
        f"Current page (CommonMark):\n```\n{format_text}\n```\n\n"
        f"Update the running summary to incorporate the current page. "
        f"Keep the output under {max_chars} characters — preserve named "
        f"entities, running arguments, and unresolved threads; "
        f"drop settled details. If the previous summary was truncated "
        f"to fit the context window, prioritize newly visible content "
        f"when absorbing this page.\n\n"
        f"{_LANG_RULE}\n\n"
        f"{_NO_REASONING}"
    )


# ---------------------------------------------------------------------------
# Client factory + per-call LLM functions
# ---------------------------------------------------------------------------

def _make_client() -> OpenAI:
    """Return a shared ``openai.OpenAI`` client pointed at the configured endpoint.

    Tests monkeypatch this at ``pdf2md_agent.crew.runner._make_client`` to
    inject a fake client (the runner re-exports the symbol for that reason).
    """
    return OpenAI(
        api_key=require_api_key(),
        base_url=OPENAI_BASE_URL,
    )


def call_extractor(
    client: OpenAI,
    *,
    image_path: Path,
    text_hint: str,
    previous_summary: str,
    max_summary_chars: int,
    target_long_side: int,
    jpeg_quality: int,
    timeout: float | None = None,
) -> str:
    """One multimodal extraction call. Returns the model's markdown string.

    Args:
        client: shared ``openai.OpenAI`` client.
        image_path: rendered (and possibly pre-resized) page image on disk.
        text_hint: native PDF text-layer hint (may be empty).
        previous_summary: running cross-page summary (may be empty on page 1).
        max_summary_chars: budget for the inlined ``previous_summary``.
        target_long_side: JPEG long-side cap applied to ``image_path``.
        jpeg_quality: JPEG quality applied to ``image_path``.
        timeout: per-call HTTP timeout; ``None`` → SDK default.

    Returns:
        Stripped markdown extraction (think blocks removed).

    Raises:
        ImageEncodeError: when ``image_path`` cannot be encoded.
        openai.OpenAIError: any HTTP / SDK error (caller's
            ``call_with_retry`` decides whether to retry).
    """
    image_data_url = _image_to_data_url(
        image_path,
        target_long_side=target_long_side,
        jpeg_quality=jpeg_quality,
    )
    user_text = build_extractor_user_text(
        text_hint=text_hint,
        previous_summary=previous_summary,
        max_summary_chars=max_summary_chars,
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": EXTRACTOR_PERSONA},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        timeout=timeout,
    )
    raw = _extract_content(response)
    return _strip_think(raw)


def call_formatter(
    client: OpenAI,
    *,
    extract_text: str,
    max_input_tokens: int = DEFAULT_FORMATTER_INPUT_TOKENS,
    timeout: float | None = None,
) -> str:
    """One text-only formatter call. ``extract_text`` is head+tail-truncated.

    Args:
        client: shared ``openai.OpenAI`` client.
        extract_text: full extractor output (read from disk or inlined).
        max_input_tokens: rough token budget for the inlined extract
            (converted to chars via ``* 4`` for ``_truncate_summary``).
        timeout: per-call HTTP timeout; ``None`` → SDK default.

    Returns:
        Strict CommonMark markdown, think blocks stripped.
    """
    user_text = build_formatter_user_text(
        extract_text, max_chars=max_input_tokens * 4
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": FORMATTER_PERSONA_STRICT},
            {"role": "user", "content": user_text},
        ],
        timeout=timeout,
    )
    raw = _extract_content(response)
    return _strip_think(raw)


def call_summarizer(
    client: OpenAI,
    *,
    format_text: str,
    previous_summary: str,
    max_chars: int,
    timeout: float | None = None,
) -> str:
    """One text-only summarizer call. Returns the updated running summary.

    Args:
        client: shared ``openai.OpenAI`` client.
        format_text: current page's strict CommonMark (formatter output).
        previous_summary: summary at the start of this page (may be empty).
        max_chars: budget for the returned summary (runner post-truncates
            to the same cap to guarantee the invariant).
        timeout: per-call HTTP timeout; ``None`` → SDK default.

    Returns:
        Updated running summary string, think blocks stripped.
    """
    user_text = build_summarizer_user_text(
        format_text=format_text,
        previous_summary=previous_summary,
        max_chars=max_chars,
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SUMMARIZER_PERSONA},
            {"role": "user", "content": user_text},
        ],
        timeout=timeout,
    )
    raw = _extract_content(response)
    return _strip_think(raw)


def _extract_content(response: object) -> str:
    """Pluck ``choices[0].message.content`` from an OpenAI chat response.

    Defensive against missing choices / None content (some endpoints return
    ``None`` for the message body when streaming is misconfigured).
    """
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    return content if isinstance(content, str) else ""


__all__ = [
    "DEFAULT_FORMATTER_INPUT_TOKENS",
    "EXTRACTOR_PERSONA",
    "FORMATTER_PERSONA_STRICT",
    "ImageEncodeError",
    "PERSONA_VERSION",
    "SUMMARIZER_PERSONA",
    "_COMMON_TASK_RULES",
    "_encode_local_image",
    "_make_client",
    "_strip_think",
    "_truncate_summary",
    "build_extractor_user_text",
    "build_formatter_user_text",
    "build_summarizer_user_text",
    "call_extractor",
    "call_formatter",
    "call_summarizer",
]

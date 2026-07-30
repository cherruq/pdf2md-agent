"""Per-page extraction loop: build crew, kickoff, optionally reflect on coverage.

The vision model occasionally under-transcribes a page (e.g. drops a
column of a table). When the runner has access to the PDF's native text
layer it can detect that drift cheaply via
:func:`difflib.SequenceMatcher` and re-run the extractor with a penalty
prompt — a deterministic, no-LLM "reflection" loop.
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crewai import LLM, Process
from pydantic import ValidationError

from pdf2md_agent.cache import CacheLayout, PageArtifacts
from pdf2md_agent.crew import runner as _runner
from pdf2md_agent.crew.fallback import _record_text_layer_fallback
from pdf2md_agent.crew.output import _output
from pdf2md_agent.crew.page_image import PreparedPage
from pdf2md_agent.crew.results import PageResult
from pdf2md_agent.llm_retry import (
    RetryConfig,
    _safe_exc_summary,
    call_with_retry,
    is_transient,
)
from pdf2md_agent.pdf_renderer import PageImage

log = logging.getLogger("pdf2md_agent.runner")


@dataclass(frozen=True, slots=True)
class ExtractionOutcome:
    """Result of one page's extraction loop.

    * ``succeeded`` contains the final Markdown output.
    * ``fell_back`` is true if the vision model failed (+ retries exhausted)
      and the returned Markdown is the native text-layer fallback stub.
    """

    succeeded: bool
    fell_back: bool
    page_result: PageResult | None
    extract_text: str
    format_md: str


def _strip_multipage_headers_footers(text: str, compare_text: str) -> str:
    """Strip prefixes/suffixes that recur across pages to avoid coverage penalties."""
    if not text or not compare_text:
        return text

    start_trim = 0
    end_trim = len(text)

    sm_head = difflib.SequenceMatcher(None, text[:200], compare_text[:200])
    head_match = sm_head.find_longest_match(
        0, min(200, len(text)), 0, min(200, len(compare_text))
    )
    if head_match.size > 5 and head_match.a < 20:
        start_trim = head_match.a + head_match.size

    sm_tail = difflib.SequenceMatcher(None, text[-200:], compare_text[-200:])
    tail_match = sm_tail.find_longest_match(
        0, min(200, len(text)), 0, min(200, len(compare_text))
    )
    if tail_match.size > 5 and (len(text[-200:]) - (tail_match.a + tail_match.size)) < 20:
        end_trim = max(start_trim, len(text) - 200 + tail_match.a)

    if start_trim < end_trim:
        trimmed = text[start_trim:end_trim].strip()
        lines = trimmed.splitlines()
        while lines and re.match(r"^\s*[\d\-]+\s*$", lines[0]):
            lines.pop(0)
        while lines and re.match(r"^\s*[\d\-]+\s*$", lines[-1]):
            lines.pop(-1)
        return "\n".join(lines)
    return text


def _clean_for_coverage(text: str) -> str:
    """Drop all whitespace for a cheap character-coverage comparison."""
    return re.sub(r"\s+", "", text)


_REFLECTION_COVERAGE_THRESHOLD = 0.90
_REFLECTION_MAX_ATTEMPTS = 2
_PENALTY_PROMPT = (
    "\n\nCRITICAL WARNING: Your previous output missed significant portions "
    "of the native text. You MUST preserve ALL text. Please re-read the "
    "page carefully and transcribe completely."
)


def _build_crew(
    *,
    extractor: Any,
    formatter: Any,
    prepared: PreparedPage,
    text_hint_str: str,
    penalty_prompt: str,
    available_images: list[str],
    **_kwargs: object,
) -> tuple[Any, Any, Any]:
    """Construct the (extract → format) crew for one attempt."""
    extract_t = _runner.make_extract_task(
        extractor,
        prepared.attach_image_path,
        text_hint=text_hint_str + penalty_prompt,
        available_images=available_images,
        is_tiled=prepared.is_tiled,
        tile_paths=prepared.tile_paths,
    )
    format_t = _runner.make_format_task(formatter, extract_t)
    tasks = [extract_t, format_t]
    agents_list = [extractor, formatter]

    crew = _runner.Crew(
        agents=agents_list,
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )
    return crew, extract_t, format_t


def _maybe_reflect(
    *,
    page: PageImage,
    idx: int,
    total: int,
    format_md: str,
    coverage_text_hint: str,
    reflection_attempts: int,
) -> tuple[bool, str]:
    """Evaluate extraction coverage against the text hint and possibly reflect.

    If the extraction dropped significant text present in the PDF text
    layer, return a penalty prompt instructing the model to fix the
    omissions. If coverage is acceptable, return an empty string.
    """
    if reflection_attempts >= _REFLECTION_MAX_ATTEMPTS:
        return False, ""

    native_clean = _clean_for_coverage(coverage_text_hint)
    if len(native_clean) <= 20:
        return False, ""

    md_clean = _clean_for_coverage(format_md)
    sm = difflib.SequenceMatcher(None, native_clean, md_clean)
    hit_count = sum(block.size for block in sm.get_matching_blocks())
    coverage = hit_count / len(native_clean) if native_clean else 1.0

    if coverage >= _REFLECTION_COVERAGE_THRESHOLD:
        return False, ""

    log.warning(
        "  [%d/%d] page %d: poor text coverage (%.2f < %.2f); "
        "triggering reflection %d",
        idx, total, page.page_number,
        coverage, _REFLECTION_COVERAGE_THRESHOLD, reflection_attempts + 1,
    )
    return True, _PENALTY_PROMPT


def run_extraction_loop(
    *,
    page: PageImage,
    artifacts: PageArtifacts,
    layout: CacheLayout,
    all_pages: list[PageImage],
    idx: int,
    total: int,
    prepared: PreparedPage,
    text_hint_str: str,
    llm: LLM,
    retry_config: RetryConfig | None,
    request_timeout_seconds: float | None,
    assets_dir: Path | None,
    fallback_to_text: bool = True,
    **_kwargs: object,
) -> ExtractionOutcome:
    """Run extract → format → (reflect) for one page.

    Retries are handled by ``call_with_retry`` around ``crew.kickoff()``.
    If the vision LLM is completely unreachable or refuses to output valid
    JSON matching the task schema, the fallback logic kicks in and emits a
    text-layer stub so the run can survive transient provider outages.
    """
    extractor = _runner.make_extractor(llm)
    formatter = _runner.make_formatter(llm)

    coverage_text_hint = text_hint_str
    if len(all_pages) > 1:
        compare_idx = idx if idx < len(all_pages) else idx - 2
        compare_text_path = layout.page_text_path(
            all_pages[compare_idx].page_number
        )
        compare_hint = (
            compare_text_path.read_text(encoding="utf-8")
            if compare_text_path.exists()
            else ""
        )
        coverage_text_hint = _strip_multipage_headers_footers(
            coverage_text_hint, compare_hint
        )

    available_images: list[str] = []
    if assets_dir:
        for img_file in assets_dir.glob(f"page_{page.page_number:04d}_img_*"):
            if img_file.is_file():
                available_images.append(img_file.name)

    extract_text = ""
    format_md = ""
    reflection_attempts = 0
    penalty_prompt = ""

    while True:
        crew, extract_t, format_t = _build_crew(
            extractor=extractor,
            formatter=formatter,
            prepared=prepared,
            text_hint_str=text_hint_str,
            penalty_prompt=penalty_prompt,
            available_images=available_images,
        )

        try:
            call_with_retry(
                crew.kickoff,
                config=retry_config or RetryConfig(),
                label=(
                    f"page {page.page_number}"
                    + (
                        f" (reflection {reflection_attempts})"
                        if reflection_attempts > 0
                        else ""
                    )
                ),
                timeout_seconds=request_timeout_seconds,
            )
        except ValidationError as exc:
            if not fallback_to_text:
                raise
            log.warning(
                "  [%d/%d] page %d: model returned malformed response "
                "(%s, %d validation error(s)); falling back to text layer",
                idx, total, page.page_number,
                type(exc).__name__, len(exc.errors()),
            )
            result = _record_text_layer_fallback(
                idx=idx, total=total,
                page_number=page.page_number,
                page_started=prepared.page_started,
                artifacts=artifacts,
                completion_label="validation-fallback",
            )
            return ExtractionOutcome(
                succeeded=False, fell_back=True,
                page_result=result,
                extract_text="", format_md="",
            )
        except BaseException as exc:  # noqa: BLE001 — see runner/AGENTS.md
            if not fallback_to_text or not is_transient(exc):
                raise
            log.warning(
                "  [%d/%d] page %d: vision pipeline failed after retries (%s); "
                "falling back to text layer",
                idx, total, page.page_number, _safe_exc_summary(exc),
            )
            result = _record_text_layer_fallback(
                idx=idx, total=total,
                page_number=page.page_number,
                page_started=prepared.page_started,
                artifacts=artifacts,
                completion_label="fallback",
            )
            return ExtractionOutcome(
                succeeded=False, fell_back=True,
                page_result=result,
                extract_text="", format_md="",
            )

        extract_text = _output(extract_t)
        format_md = _output(format_t)

        should_continue, penalty_prompt = _maybe_reflect(
            page=page, idx=idx, total=total,
            format_md=format_md,
            coverage_text_hint=coverage_text_hint,
            reflection_attempts=reflection_attempts,
        )
        if not should_continue:
            break
        reflection_attempts += 1

    return ExtractionOutcome(
        succeeded=True, fell_back=False,
        page_result=None,
        extract_text=extract_text, format_md=format_md,
    )


__all__ = [
    "ExtractionOutcome",
    "run_extraction_loop",
]
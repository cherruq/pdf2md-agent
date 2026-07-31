"""Per-page extraction loop: build crew, kickoff, optionally reflect on coverage."""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from typing import Any

from crewai import LLM, Process
from pydantic import ValidationError

from pdf2md_agent.config import ConversionConfig
from pdf2md_agent.crew import runner as _runner
from pdf2md_agent.crew.fallback import _record_text_layer_fallback
from pdf2md_agent.crew.output import _output
from pdf2md_agent.crew.page_image import PreparedPage
from pdf2md_agent.crew.results import PageResult
from pdf2md_agent.llm_retry import (
    _safe_exc_summary,
    call_with_retry,
    is_transient,
)
from pdf2md_agent.pdf_renderer import RenderedPage
from pdf2md_agent.tuning import (
    PENALTY_PROMPT as _PENALTY_PROMPT,
    REFLECTION_COVERAGE_THRESHOLD as _REFLECTION_COVERAGE_THRESHOLD,
    REFLECTION_MAX_ATTEMPTS as _REFLECTION_MAX_ATTEMPTS,
)

log = logging.getLogger("pdf2md_agent.runner")


@dataclass(frozen=True, slots=True)
class ExtractionOutcome:
    """Result of one page's extraction loop."""

    succeeded: bool
    fell_back: bool
    page_result: PageResult | None
    format_md: str


def _clean_for_coverage(text: str) -> str:
    """Drop all whitespace for a cheap character-coverage comparison."""
    return re.sub(r"\s+", "", text)


def _build_crew(
    *,
    extractor: Any,
    prepared: PreparedPage,
    text_hint_str: str,
    penalty_prompt: str,
    **_kwargs: object,
) -> tuple[Any, Any]:
    """Construct the extraction crew for one attempt."""
    extract_t = _runner.make_extract_task(
        extractor,
        prepared.attach_image_path,
        text_hint=text_hint_str + penalty_prompt,
        is_tiled=prepared.is_tiled,
        tile_paths=prepared.tile_paths,
    )
    tasks = [extract_t]
    agents_list = [extractor]

    crew = _runner.Crew(
        agents=agents_list,
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )
    return crew, extract_t


def _maybe_reflect(
    *,
    page: RenderedPage,
    idx: int,
    total: int,
    format_md: str,
    coverage_text_hint: str,
    reflection_attempts: int,
) -> tuple[bool, str]:
    """Evaluate extraction coverage against the text hint and possibly reflect."""
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
        "  [%d/%d] page %d: poor text coverage (%.2f < %.2f); triggering reflection %d",
        idx,
        total,
        page.page_number,
        coverage,
        _REFLECTION_COVERAGE_THRESHOLD,
        reflection_attempts + 1,
    )
    return True, _PENALTY_PROMPT


def run_extraction_loop(
    *,
    page: RenderedPage,
    prepared: PreparedPage,
    config: ConversionConfig,
    llm: LLM,
    text_hint_str: str,
    idx: int,
    total: int,
) -> ExtractionOutcome:
    """Run extraction → (reflect) for one page."""
    artifacts = config.layout.artifacts_for(page)
    extractor = _runner.make_extractor(llm)

    coverage_text_hint = text_hint_str
    format_md = ""
    reflection_attempts = 0
    penalty_prompt = ""

    while True:
        crew, extract_t = _build_crew(
            extractor=extractor,
            prepared=prepared,
            text_hint_str=text_hint_str,
            penalty_prompt=penalty_prompt,
        )

        try:
            call_with_retry(
                crew.kickoff,
                config=config.retry_config,
                label=(
                    f"page {page.page_number}"
                    + (f" (reflection {reflection_attempts})" if reflection_attempts > 0 else "")
                ),
                timeout_seconds=config.request_timeout_seconds,
            )
        except ValidationError as exc:
            if not config.fallback_to_text:
                raise
            log.warning(
                "  [%d/%d] page %d: model returned malformed response "
                "(%s, %d validation error(s)); falling back to text layer",
                idx,
                total,
                page.page_number,
                type(exc).__name__,
                len(exc.errors()),
            )
            result = _record_text_layer_fallback(
                idx=idx,
                total=total,
                page_number=page.page_number,
                page_started=prepared.page_started,
                artifacts=artifacts,
                completion_label="validation-fallback",
            )
            return ExtractionOutcome(
                succeeded=False,
                fell_back=True,
                page_result=result,
                format_md="",
            )
        except BaseException as exc:  # noqa: BLE001 — see runner/AGENTS.md
            if not config.fallback_to_text or not is_transient(exc):
                raise
            log.warning(
                "  [%d/%d] page %d: vision pipeline failed after retries (%s); falling back to text layer",
                idx,
                total,
                page.page_number,
                _safe_exc_summary(exc),
            )
            result = _record_text_layer_fallback(
                idx=idx,
                total=total,
                page_number=page.page_number,
                page_started=prepared.page_started,
                artifacts=artifacts,
                completion_label="fallback",
            )
            return ExtractionOutcome(
                succeeded=False,
                fell_back=True,
                page_result=result,
                format_md="",
            )

        format_md = _output(extract_t)

        should_continue, penalty_prompt = _maybe_reflect(
            page=page,
            idx=idx,
            total=total,
            format_md=format_md,
            coverage_text_hint=coverage_text_hint,
            reflection_attempts=reflection_attempts,
        )
        if not should_continue:
            break
        reflection_attempts += 1

    return ExtractionOutcome(
        succeeded=True,
        fell_back=False,
        page_result=None,
        format_md=format_md,
    )


__all__ = [
    "ExtractionOutcome",
    "run_extraction_loop",
]

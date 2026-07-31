"""argparse builder and CLI argument post-processing."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from pdf2md_agent import __version__
from pdf2md_agent.cache import CacheLayout, CacheNoCacheFlags
from pdf2md_agent.filesystem_safety import cache_key_for_pdf
from pdf2md_agent.llm_retry import RetryConfig
from pdf2md_agent.pages import parse_page_spec
from pdf2md_agent.post_stream import StitchMode

_NO_CACHE_FLAG_NAMES: tuple[str, ...] = (
    "render",
    "text",
    "resized",
    "format",
)


# --- argparse actions & validators -----------------------------------------


class _VersionAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None,
    ) -> None:
        print(f"pdf2md-agent {__version__}")
        parser.exit(0)


class _NoCacheAllAction(argparse.Action):
    """Sets every ``--no-cache-*`` flag to True when ``--no-cache-all`` is set."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None,
    ) -> None:
        for name in _NO_CACHE_FLAG_NAMES:
            setattr(namespace, f"no_cache_{name}", True)
        setattr(namespace, "no_cache_all", True)


def _request_timeout_type(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--request-timeout must be a number, got {raw!r}"
        ) from exc
    if not 0.1 <= value <= 600.0:
        raise argparse.ArgumentTypeError(
            f"--request-timeout must be in [0.1, 600], got {value}"
        )
    return value


def _positive_int_type(name: str, minimum: int) -> Callable[[str], int]:
    def _parser(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--{name} must be an integer, got {raw!r}"
            ) from exc
        if value < minimum:
            raise argparse.ArgumentTypeError(
                f"--{name} must be >= {minimum}, got {value}"
            )
        return value

    return _parser


def _safe_intermediates_dir(value: str) -> Path:
    """argparse validator for ``--intermediates-dir``, rejecting '..' segments."""
    p = Path(value)
    if any(part == ".." for part in p.parts):
        raise argparse.ArgumentTypeError(
            f"--intermediates-dir must not contain '..' segments: {value!r}"
        )
    return p


# --- Post-parse resolvers --------------------------------------------------


def _resolve_no_cache_flags(args: argparse.Namespace) -> CacheNoCacheFlags:
    return CacheNoCacheFlags(
        render=bool(args.no_cache_render),
        text=bool(args.no_cache_text),
        resized=bool(args.no_cache_resized),
        format=bool(args.no_cache_format),
    )


def _resolve_layout(
    pdf: Path,
    override: Path | None,
) -> tuple[CacheLayout, Path]:
    root = (
        override
        if override is not None
        else Path(".pdf2md-agent-cache") / cache_key_for_pdf(pdf)
    )
    return CacheLayout.for_pdf(root, pdf), root / "pages"


def build_retry_config(args: argparse.Namespace) -> RetryConfig | None:
    """Build a :class:`RetryConfig` from CLI args (override) + env (fallback)."""
    from pdf2md_agent.config import (
        RETRY_INITIAL_DELAY,
        RETRY_JITTER,
        RETRY_MAX_ATTEMPTS,
        RETRY_MAX_DELAY,
    )

    cli_max_attempts = args.max_retries
    if cli_max_attempts == 0:
        cli_max_attempts = None
    try:
        return RetryConfig(
            max_attempts=(
                cli_max_attempts
                if cli_max_attempts is not None
                else RETRY_MAX_ATTEMPTS
            ),
            initial_delay=RETRY_INITIAL_DELAY,
            max_delay=RETRY_MAX_DELAY,
            jitter=RETRY_JITTER,
        )
    except ValueError as exc:
        print(f"error: invalid retry argument: {exc}", file=__import__("sys").stderr)
        return None


# --- Parser definition -----------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf2md-agent",
        description=(
            "Render every page of a PDF to an image and feed it through a "
            "CrewAI pipeline (extract → format) to produce "
            "language-preserving Markdown.\n\n"
            "Stages: render → extract → format → stitch.\n\n"
            "Cache: per-resource (render/text/resized/extract/format) "
            "is reused by default and gated by meta.json file path validation. "
            "Any drift → fail loud.\n\n"
            "--no-cache-<resource> opts out a specific resource from cache reuse. "
            "--no-cache-all disables all cache reuse. --no-<feature> disables an "
            "optional feature.\n\n"
            "See CONTRIBUTING.md for naming conventions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    pipeline = parser.add_argument_group(
        "Pipeline",
        "Inputs that drive the per-page pipeline.",
    )
    pipeline.add_argument("pdf", type=Path, help="Input PDF path.")
    pipeline.add_argument("-o", "--output", type=Path, required=True, help="Output markdown path.")
    pipeline.add_argument(
        "--dpi",
        type=int,
        default=144,
        help=(
            "Rendering DPI (default: 144). Recommended: "
            "72 (PDF native, smallest), "
            "150 (text + tables), "
            "200 (small fonts / dense formulas), "
            "300+ (print, usually overkill for vision models)."
        ),
    )
    pipeline.add_argument(
        "-p", "--pages",
        type=parse_page_spec,
        default=None,
        metavar="SPEC",
        help=(
            "Subset of pages to convert. Accepts comma-separated pages and "
            "ranges, e.g. '1-5,8,11-13'. Pages are 1-based; output "
            "is ordered by document position. "
            "Default: all pages."
        ),
    )

    cache = parser.add_argument_group(
        "Cache control",
        "Defaults trust cached resources. Each --no-cache-* opts a single "
        "resource out; --no-cache-all opts every resource out.",
    )
    cache.add_argument(
        "--intermediates-dir",
        type=_safe_intermediates_dir,
        default=None,
        help="Override the intermediates cache directory (default: .pdf2md-agent-cache/<pdf_stem>/).",
    )
    for name in _NO_CACHE_FLAG_NAMES:
        cache.add_argument(
            f"--no-cache-{name}",
            action="store_true",
            default=False,
            dest=f"no_cache_{name}",
            help=argparse.SUPPRESS,
        )
    cache.add_argument(
        "--no-cache-all",
        action=_NoCacheAllAction,
        nargs=0,
        default=False,
        dest="no_cache_all",
        help=(
            "Disable every cache reuse (render/text/resized/format). "
            "Equivalent to passing all four --no-cache-* flags."
        ),
    )

    features = parser.add_argument_group(
        "Feature disable",
        "Optional features; each --no-<feature> opts a single feature out.",
    )
    features.add_argument(
        "--no-text-hint",
        action="store_true",
        help="Disable feeding the PDF's native text layer to the extractor.",
    )
    features.add_argument(
        "--stitch-mode",
        choices=[m.value for m in StitchMode],
        default=StitchMode.HEURISTIC.value,
        help=(
            "How to join per-page Markdown into the final document. "
            "'heuristic' (default) merges paragraphs/list items/table rows "
            "split across page boundaries and drops the '---' page separator. "
            "'off' preserves the legacy '\\n\\n---\\n\\n' separator verbatim."
        ),
    )

    tuning = parser.add_argument_group(
        "Retry & tuning",
        "LLM retry budget, image downscale, and token-budget knobs.",
    )
    tuning.add_argument(
        "--max-retries",
        type=_positive_int_type("max-retries", 0),
        default=None,
        help=(
            "Total LLM call attempts per page (initial + retries). Pass 0 "
            "or omit to retry transient failures indefinitely. Default: 0 "
            "(unlimited)."
        ),
    )
    tuning.add_argument(
        "--image-long-side",
        type=_positive_int_type("image-long-side", 64),
        default=None,
        metavar="PX",
        help="Long-side cap (pixels) for inlined page images. Default: 1536.",
    )
    tuning.add_argument(
        "--image-quality",
        type=_positive_int_type("image-quality", 1),
        default=None,
        metavar="Q",
        help=(
            "JPEG quality (1-100) used when downsampling page images. "
            "Default: 85."
        ),
    )
    tuning.add_argument(
        "--ctx-limit",
        type=_positive_int_type("ctx-limit", 256),
        default=None,
        metavar="TOK",
        help="Model context-window token limit the runner budgets against.",
    )
    tuning.add_argument(
        "--request-timeout",
        type=_request_timeout_type,
        default=None,
        metavar="SEC",
        help="Per-attempt wall-clock timeout (seconds, 0.1-600). Default: 60.0.",
    )

    parser.add_argument(
        "-V", "--version",
        action=_VersionAction,
        nargs=0,
        help="Print the pdf2md-agent version and exit.",
    )
    return parser


__all__ = [
    "_NO_CACHE_FLAG_NAMES",
    "_resolve_layout",
    "_resolve_no_cache_flags",
    "_safe_intermediates_dir",
    "build_parser",
    "build_retry_config",
]
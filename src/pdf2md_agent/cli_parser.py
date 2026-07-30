"""argparse builder + CLI argument post-processing.

Two responsibilities live here:

* :func:`build_parser` — the :class:`argparse.ArgumentParser` definition
  for the ``pdf2md-agent`` CLI. Argument groups (Pipeline / Cache
  control / Feature disable / Retry & tuning / Diagnostic) are surfaced
  in ``--help`` so users can discover flags without reading the README.
* Post-parse resolution helpers (:func:`_resolve_no_cache_flags`,
  :func:`_resolve_layout`, :func:`build_retry_config`) — pure functions
  that turn a parsed :class:`argparse.Namespace` into the typed value
  objects the runtime expects (:class:`CacheNoCacheFlags`,
  :class:`CacheLayout`, :class:`RetryConfig`).

Validation helpers (``_request_timeout_type``, ``_positive_int_type``,
``_safe_intermediates_dir``) live at module scope; the cache-key /
safe-stem helpers moved to :mod:`pdf2md_agent.filesystem_safety`.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Callable

from pdf2md_agent import __about__
from pdf2md_agent.cache import CacheLayout, CacheNoCacheFlags
from pdf2md_agent.filesystem_safety import cache_key_for_pdf, safe_cache_stem
from pdf2md_agent.llm_retry import RetryConfig
from pdf2md_agent.pages import parse_page_spec
from pdf2md_agent.post_stream import StitchMode


# Legacy aliases: tests import these names from cli_parser.
_cache_key_for_pdf = cache_key_for_pdf
_safe_cache_stem = safe_cache_stem


_NO_CACHE_FLAG_NAMES: tuple[str, ...] = (
    "render",
    "text",
    "resized",
    "extract",
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
        print(f"pdf2md-agent {__about__.__version__}")
        parser.exit(0)


class _NoCacheAllAction(argparse.Action):
    """Sets every ``--no-cache-*`` flag to True when ``--no-cache-all`` is set.

    Implemented as a custom ``Action`` so post-parse resolution happens
    automatically regardless of argument order on the command line.
    """

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
    """argparse ``type=`` for ``--request-timeout`` (0.1s–600s)."""
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
    """argparse ``type=`` for ``--intermediates-dir``.

    Rejects values that contain ``..`` path segments so a malicious or
    mistaken flag cannot point the cache directory outside the working
    tree (path-traversal guard, D11-N02).
    """
    p = Path(value)
    if any(part == ".." for part in p.parts):
        raise argparse.ArgumentTypeError(
            f"--intermediates-dir must not contain '..' segments: {value!r}"
        )
    return p


# --- Post-parse resolvers --------------------------------------------------


def _resolve_no_cache_flags(args: argparse.Namespace) -> CacheNoCacheFlags:
    """Build a :class:`CacheNoCacheFlags` from CLI flags.

    The ``--no-cache-all`` action already flips every per-resource flag,
    so this is a straight attribute-to-field copy.
    """
    return CacheNoCacheFlags(
        render=bool(args.no_cache_render),
        text=bool(args.no_cache_text),
        resized=bool(args.no_cache_resized),
        extract=bool(args.no_cache_extract),
        format=bool(args.no_cache_format),
    )


def _resolve_layout(
    pdf: Path,
    override: Path | None,
    keep_intermediates: bool,
) -> tuple[CacheLayout, Path]:
    """Return ``(layout, render_target_pages_dir)``.

    When ``keep_intermediates`` is False the layout lives under a tempdir
    that is removed on context exit.
    """
    if keep_intermediates:
        root = (
            override
            if override is not None
            else Path(".pdf2md-agent-cache") / cache_key_for_pdf(pdf)
        )
        return CacheLayout.for_pdf(root, pdf), root / "pages"

    td = Path(tempfile.mkdtemp(prefix="pdf2md_agent_"))
    pages = td / "pages"
    pages.mkdir()
    return (
        CacheLayout(
            root=td,
            pages_dir=pages,
            meta_path=td / "meta.json",
        ),
        pages,
    )


def build_retry_config(args: argparse.Namespace) -> RetryConfig | None:
    """Build a :class:`RetryConfig` from CLI args (override) + env (fallback).

    Returns ``None`` on invalid input so the caller can print a
    user-facing error and exit non-zero. ``--max-retries 0`` (or env
    ``PDF2MD_AGENT_MAX_RETRIES=0``) means "unlimited" and is normalized
    to ``RetryConfig.max_attempts=None``.
    """
    # Defer the env-fallback imports to keep this module's top light.
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
        "--no-intermediates",
        action="store_true",
        help="Skip writing intermediate cache files (uses a tempdir).",
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
            "Disable every cache reuse (render/text/resized/extract/"
            "format). Equivalent to passing all five --no-cache-* "
            "flags."
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
            "or omit to retry transient failures indefinitely. Overrides "
            "PDF2MD_AGENT_MAX_RETRIES. Default: 0 (unlimited)."
        ),
    )
    tuning.add_argument(
        "--image-long-side",
        type=_positive_int_type("image-long-side", 64),
        default=None,
        metavar="PX",
        help=(
            "Long-side cap (pixels) for inlined page images. Overrides "
            "PDF2MD_AGENT_IMAGE_LONG_SIDE. Default: 1536."
        ),
    )
    tuning.add_argument(
        "--image-quality",
        type=_positive_int_type("image-quality", 1),
        default=None,
        metavar="Q",
        help=(
            "JPEG quality (1-100) used when downsampling page images. "
            "Overrides PDF2MD_AGENT_IMAGE_JPEG_QUALITY. Default: 85."
        ),
    )
    tuning.add_argument(
        "--ctx-limit",
        type=_positive_int_type("ctx-limit", 256),
        default=None,
        metavar="TOK",
        help=(
            "Model context-window token limit the runner budgets against. "
            "Overrides PDF2MD_AGENT_CTX_LIMIT."
        ),
    )
    tuning.add_argument(
        "--request-timeout",
        type=_request_timeout_type,
        default=None,
        metavar="SEC",
        help=(
            "Per-attempt wall-clock timeout (seconds, 0.1-600). Overrides "
            "PDF2MD_AGENT_REQUEST_TIMEOUT. Default: 60.0."
        ),
    )

    parser.add_argument(
        "-V", "--version",
        action=_VersionAction,
        nargs=0,
        help="Print the pdf2md-agent version and exit.",
    )
    return parser


# Legacy aliases under the underscore-prefixed names tests still import.
_safe_cache_stem = safe_cache_stem
_cache_key_for_pdf = cache_key_for_pdf


__all__ = [
    "_NO_CACHE_FLAG_NAMES",
    "_cache_key_for_pdf",
    "_resolve_layout",
    "_resolve_no_cache_flags",
    "_safe_cache_stem",
    "_safe_intermediates_dir",
    "build_parser",
    "build_retry_config",
]
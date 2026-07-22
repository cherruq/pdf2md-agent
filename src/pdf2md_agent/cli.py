"""CLI entry point for pdf2md-agent."""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time

import pymupdf
from pathlib import Path

from pdf2md_agent.cache import (
    CacheLayout,
    CacheNoCacheFlags,
    atomic_write_text,
    write_meta,
)
from pdf2md_agent.config import (
    CTX_LIMIT,
    FALLBACK_TO_TEXT,
    IMAGE_JPEG_QUALITY,
    IMAGE_LONG_SIDE,
    IMAGE_MIN_LONG_SIDE,
    MAX_SUMMARY_CHARS,
    MODEL_NAME,
    RETRY_BACKOFF,
    RETRY_INITIAL_DELAY,
    RETRY_JITTER,
    RETRY_MAX_ATTEMPTS,
    RETRY_MAX_DELAY,
    TOKEN_BUDGET_SAFETY,
)
from pdf2md_agent.crew.agents import PERSONA_VERSION
from pdf2md_agent.crew.runner import run_pipeline
from pdf2md_agent.llm_retry import RetryConfig
from pdf2md_agent.pages import parse_page_spec, resolve_pages
from pdf2md_agent.pdf_renderer import render_pdf
from pdf2md_agent.post_stream import StitchMode, stitch_pages
from pdf2md_agent.vision import make_vision_llm

log = logging.getLogger("pdf2md-agent")


_NO_CACHE_FLAG_NAMES: tuple[str, ...] = (
    "render",
    "text",
    "resized",
    "extract",
    "format",
    "summary",
)


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


# Windows reserved device names. ``CreateFile`` rejects these as bare
# filenames (with or without an extension), and ``mkdir`` on a reserved
# name surfaces as an opaque OSError. ``CON``, ``PRN``, ``AUX``, ``NUL``
# plus ``COM1``-``COM9`` and ``LPT1``-``LPT9``. Case-insensitive on Windows.
_WINDOWS_RESERVED_NAMES: frozenset[str] = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def _safe_cache_stem(stem: str) -> str:
    """Return a filesystem-safe cache directory name derived from ``stem``.

    On Windows, the bare filenames ``CON``, ``PRN``, ``AUX``, ``NUL``,
    ``COM1``-``COM9``, and ``LPT1``-``LPT9`` are reserved device names and
    cannot be used as a directory name — ``mkdir`` on ``.pdf2md-agent-cache/CON``
    fails with an opaque OSError. Trailing dots / spaces and leading
    whitespace are likewise rejected. We append a single ``_`` so the
    cache lives at ``<reserved>_`` instead of crashing the run.

    On non-Windows platforms the reservation does not apply; we still
    strip trailing dots/spaces defensively for portability.

    Case-collision (D16-002): on case-insensitive filesystems (NTFS,
    APFS, HFS+) two PDFs whose stems differ only in case map to the
    same cache directory. We do not canonicalize here — instead, callers
    are warned via this function's docstring to pick distinct stems.
    """
    if not stem:
        return "_"
    candidate = stem.rstrip(" .")
    if not candidate:
        return "_"
    if sys.platform == "win32" and candidate.upper() in _WINDOWS_RESERVED_NAMES:
        return candidate + "_"
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf2md-agent",
        description="Convert a PDF to markdown via a CrewAI vision pipeline (MiniMax-M3).",
    )
    parser.add_argument("pdf", type=Path, help="Input PDF path.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output markdown path.")
    parser.add_argument(
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
    parser.add_argument(
        "-p", "--pages",
        type=parse_page_spec,
        default=None,
        metavar="SPEC",
        help=(
            "Subset of pages to convert. Accepts comma-separated pages and "
            "ranges, e.g. '1-5,8,11-13'. Pages are 1-based; output preserves "
            "original page numbers and is ordered by document position. "
            "Default: all pages."
        ),
    )
    parser.add_argument(
        "--no-intermediates",
        action="store_true",
        help="Skip writing intermediate cache files.",
    )
    parser.add_argument(
        "--intermediates-dir",
        type=_safe_intermediates_dir,
        default=None,
        help="Override the intermediates cache directory (default: .pdf2md-agent-cache/<pdf_stem>/).",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable cross-page running summary (process each page independently).",
    )
    parser.add_argument(
        "--no-text-hint",
        action="store_true",
        help="Disable feeding the PDF's native text layer to the extractor.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help=(
            "Total LLM call attempts per page (initial + retries). Overrides "
            "PDF2MD_AGENT_MAX_RETRIES. Default: 4."
        ),
    )
    parser.add_argument(
        "--retry-initial-delay",
        type=float,
        default=None,
        help=(
            "Initial retry delay in seconds (exponential backoff). Overrides "
            "PDF2MD_AGENT_RETRY_INITIAL_DELAY. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=None,
        help=(
            "Exponential backoff multiplier between retries. Overrides "
            "PDF2MD_AGENT_RETRY_BACKOFF. Default: 2.0."
        ),
    )
    parser.add_argument(
        "--retry-max-delay",
        type=float,
        default=None,
        help=(
            "Per-attempt retry delay cap in seconds. Overrides "
            "PDF2MD_AGENT_RETRY_MAX_DELAY. Default: 30.0."
        ),
    )
    parser.add_argument(
        "--retry-jitter",
        type=float,
        default=None,
        help=(
            "Jitter ratio in [0.0, 1.0] applied to each retry delay to avoid "
            "thundering-herd. Overrides PDF2MD_AGENT_RETRY_JITTER. Default: 0.25."
        ),
    )
    parser.add_argument(
        "--no-fallback-to-text",
        action="store_true",
        default=False,
        dest="no_fallback_to_text",
        help=(
            "On retry exhaustion, raise instead of falling back to the PDF's "
            "native text layer. Default: fallback enabled."
        ),
    )
    parser.add_argument(
        "--image-long-side",
        type=int,
        default=None,
        metavar="PX",
        help=(
            "Long-side cap (pixels) for inlined page images. The runner "
            "rescales each page PNG to this size as JPEG at the configured "
            "quality before base64-encoding it. Lower values shrink the per-"
            "call token cost at the expense of OCR fidelity. Overrides "
            "PDF2MD_AGENT_IMAGE_LONG_SIDE. Default: 1536."
        ),
    )
    parser.add_argument(
        "--image-quality",
        type=int,
        default=None,
        metavar="Q",
        help=(
            "JPEG quality (1-100) used when the runner downsamples page "
            "images. Higher values preserve detail but enlarge the per-call "
            "token cost. 75-95 is the practical sweet spot. Overrides "
            "PDF2MD_AGENT_IMAGE_JPEG_QUALITY. Default: 85."
        ),
    )
    parser.add_argument(
        "--max-summary-chars",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum running-summary size (characters) fed into the next "
            "page's extract call and produced by the summarizer. Overrides "
            "PDF2MD_AGENT_MAX_SUMMARY_CHARS. Default: 800."
        ),
    )
    parser.add_argument(
        "--ctx-limit",
        type=int,
        default=None,
        metavar="TOK",
        help=(
            "Model context-window token limit the runner budgets against. "
            "Used only when PDF2MD_AGENT_CTX_LIMIT is wrong. Default: 2013."
        ),
    )
    parser.add_argument(
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
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=(
            "Model name to record in meta.json for fingerprint validation. "
            "Defaults to PDF2MD_AGENT_MODEL (default: MiniMax-M3)."
        ),
    )
    parser.add_argument(
        "--persona-version",
        default=PERSONA_VERSION,
        help=(
            "Persona fingerprint (16-char hex) recorded in meta.json. The "
            "runner refuses to re-use cache when this drifts. Defaults to "
            "the SHA-256[:16] of the active persona strings."
        ),
    )
    for name in _NO_CACHE_FLAG_NAMES:
        parser.add_argument(
            f"--no-cache-{name}",
            action="store_true",
            default=False,
            dest=f"no_cache_{name}",
            help=argparse.SUPPRESS,
        )
    parser.add_argument(
        "--no-cache-all",
        action=_NoCacheAllAction,
        nargs=0,
        default=False,
        dest="no_cache_all",
        help=(
            "Disable every cache reuse (render/text/resized/extract/"
            "format/summary). Equivalent to passing all six --no-cache-* "
            "flags."
        ),
    )
    return parser


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
        summary=bool(args.no_cache_summary),
    )


def _resolve_layout(
    pdf: Path,
    override: Path | None,
    keep_intermediates: bool,
) -> tuple[CacheLayout, Path]:
    """Return ``(layout, render_target_pages_dir)``.

    When ``keep_intermediates`` is False the layout lives under a tempdir that
    is removed on context exit.
    """
    if keep_intermediates:
        root = override if override is not None else Path(".pdf2md-agent-cache") / _safe_cache_stem(pdf.stem)
        return CacheLayout.for_pdf(root, pdf), root / "pages"

    td = Path(tempfile.mkdtemp(prefix="pdf2md_agent_"))
    pages = td / "pages"
    pages.mkdir()
    return (
        CacheLayout(
            root=td,
            pages_dir=pages,
            summary_path=td / "summary.json",
            meta_path=td / "meta.json",
        ),
        pages,
    )


_atomic_write_text = atomic_write_text


def _build_retry_config(args: argparse.Namespace) -> RetryConfig | None:
    """Build a RetryConfig from CLI args (override) + env (fallback). Returns None on invalid input."""
    try:
        return RetryConfig(
            max_attempts=(
                args.max_retries if args.max_retries is not None else RETRY_MAX_ATTEMPTS
            ),
            initial_delay=(
                args.retry_initial_delay
                if args.retry_initial_delay is not None
                else RETRY_INITIAL_DELAY
            ),
            backoff=(
                args.retry_backoff if args.retry_backoff is not None else RETRY_BACKOFF
            ),
            max_delay=(
                args.retry_max_delay
                if args.retry_max_delay is not None
                else RETRY_MAX_DELAY
            ),
            jitter=(
                args.retry_jitter if args.retry_jitter is not None else RETRY_JITTER
            ),
        )
    except ValueError as e:
        print(f"error: invalid retry argument: {e}", file=sys.stderr)
        return None


def cmd_convert(args: argparse.Namespace) -> int:
    retry_config = _build_retry_config(args)
    if retry_config is None:
        return 1
    fallback_to_text = FALLBACK_TO_TEXT and not args.no_fallback_to_text

    if not args.pdf.exists():
        print(f"error: input PDF not found: {args.pdf}", file=sys.stderr)
        return 1

    # D10-N04: fail fast on non-PDF input before any tempdir/cache work.
    try:
        with args.pdf.open("rb") as _pdf_header_fh:
            _pdf_header = _pdf_header_fh.read(5)
    except OSError as _pdf_header_exc:
        print(
            f"error: cannot read input PDF {args.pdf}: {_pdf_header_exc}",
            file=sys.stderr,
        )
        return 1
    if not _pdf_header.startswith(b"%PDF-"):
        print(
            f"error: input file is not a PDF (missing %PDF- header): {args.pdf}",
            file=sys.stderr,
        )
        return 1

    started = time.monotonic()
    keep_intermediates = not args.no_intermediates
    with_summary = not args.no_summary
    no_cache_flags = _resolve_no_cache_flags(args)

    # Resolve --pages against the PDF's actual page count so out-of-range
    # errors surface before we commit to creating a tempdir or doing render work.
    resolved_pages: list[int] | None
    if args.pages is None:
        resolved_pages = None
    else:
        doc = pymupdf.open(args.pdf)
        try:
            resolved_pages = resolve_pages(args.pages, doc.page_count)
        except ValueError as e:
            print(f"error: --pages {args.pages!r}: {e}", file=sys.stderr)
            return 1
        finally:
            doc.close()

    # Defensive empty-pages guard: ``resolve_pages`` raises on a 0-page
    # PDF, so reaching here implies ``--pages`` filtered everything out.
    if args.pages is not None and not resolved_pages:
        print("ERROR: PDF has no pages to process.", file=sys.stderr)
        raise SystemExit(1)

    if keep_intermediates:
        layout, render_target = _resolve_layout(args.pdf, args.intermediates_dir, True)
        return _run_pipeline(
            args=args,
            layout=layout,
            render_target=render_target,
            resolved_pages=resolved_pages,
            keep_intermediates=True,
            with_summary=with_summary,
            retry_config=retry_config,
            fallback_to_text=fallback_to_text,
            started=started,
            no_cache=no_cache_flags,
        )

    with tempfile.TemporaryDirectory(prefix="pdf2md_agent_") as td_str:
        td = Path(td_str)
        pages_dir = td / "pages"
        pages_dir.mkdir()
        layout = CacheLayout(
            root=td,
            pages_dir=pages_dir,
            summary_path=td / "summary.json",
            meta_path=td / "meta.json",
        )
        return _run_pipeline(
            args=args,
            layout=layout,
            render_target=pages_dir,
            resolved_pages=resolved_pages,
            keep_intermediates=False,
            with_summary=with_summary,
            retry_config=retry_config,
            fallback_to_text=fallback_to_text,
            started=started,
            no_cache=no_cache_flags,
        )


def _run_pipeline(
    *,
    args: argparse.Namespace,
    layout: CacheLayout,
    render_target: Path,
    resolved_pages: list[int] | None,
    keep_intermediates: bool,
    with_summary: bool,
    retry_config: RetryConfig,
    fallback_to_text: bool,
    started: float,
    no_cache: CacheNoCacheFlags,
) -> int:
    log.info("converting %s", args.pdf)
    log.info("  output:          %s", args.output)
    log.info("  cache:           %s", layout.root if keep_intermediates else "(tempdir, discarded)")
    log.info("  dpi:             %d", args.dpi)
    log.info("  pages:           %s", "all" if resolved_pages is None else resolved_pages)
    log.info("  no-cache:        %s", no_cache.as_dict())
    log.info("  cross-page:      %s", "summary" if with_summary else "independent")
    log.info("  text-hint:       %s", "on" if not args.no_text_hint else "off")

    if keep_intermediates:
        write_meta(
            layout.meta_path,
            pdf=args.pdf,
            dpi=args.dpi,
            with_summary=with_summary,
            pages=resolved_pages,
            model=args.model,
            persona_version=args.persona_version,
        )

    log.info("rendering PDF to PNGs at %d dpi%s...", args.dpi, " (subset)" if resolved_pages else "")
    pages = render_pdf(args.pdf, render_target, dpi=args.dpi, pages=resolved_pages)
    log.info("rendered %d page(s) to %s", len(pages), render_target)

    log.info("running pipeline: extract + format%s", " + summarize" if with_summary else "")
    llm = make_vision_llm()
    log.info(
        "  retry:           max_attempts=%d, initial_delay=%.1fs, backoff=%.1fx, max_delay=%.1fs, jitter=±%.0f%%",
        retry_config.max_attempts,
        retry_config.initial_delay,
        retry_config.backoff,
        retry_config.max_delay,
        retry_config.jitter * 100,
    )
    log.info("  fallback:        %s", "text layer" if fallback_to_text else "off")
    image_long_side = args.image_long_side if args.image_long_side is not None else IMAGE_LONG_SIDE
    image_jpeg_quality = args.image_quality if args.image_quality is not None else IMAGE_JPEG_QUALITY
    max_summary_chars = args.max_summary_chars if args.max_summary_chars is not None else MAX_SUMMARY_CHARS
    ctx_limit = args.ctx_limit if args.ctx_limit is not None else CTX_LIMIT
    log.info(
        "  budget:          ctx_limit=%d, safety=%.0f%%, image_long_side=%dpx, "
        "image_q=%d, max_summary=%d chars",
        ctx_limit,
        TOKEN_BUDGET_SAFETY * 100,
        image_long_side,
        image_jpeg_quality,
        max_summary_chars,
    )
    results = run_pipeline(
        pages=pages,
        layout=layout,
        with_summary=with_summary,
        no_cache=no_cache,
        text_hint=not args.no_text_hint,
        llm=llm,
        retry_config=retry_config,
        fallback_to_text=fallback_to_text,
        ctx_limit=ctx_limit,
        image_long_side=image_long_side,
        image_min_long_side=IMAGE_MIN_LONG_SIDE,
        image_jpeg_quality=image_jpeg_quality,
        max_summary_chars=max_summary_chars,
        token_budget_safety=TOKEN_BUDGET_SAFETY,
    )

    stitch_mode = StitchMode(args.stitch_mode)
    if stitch_mode is StitchMode.HEURISTIC:
        markdown = stitch_pages(results)
        log.info("  stitch:          heuristic (cross-page merged)")
    else:
        markdown = stitch_pages(results, mode=stitch_mode)
        log.info("  stitch:          off (legacy '---' separator preserved)")
    _atomic_write_text(args.output, markdown)
    elapsed = time.monotonic() - started
    log.info(
        "wrote %s — %d page(s), %s chars in %.1fs",
        args.output,
        len(results),
        f"{len(markdown):,}",
        elapsed,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    args = build_parser().parse_args(argv)
    return cmd_convert(args)


if __name__ == "__main__":
    sys.exit(main())
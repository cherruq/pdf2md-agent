"""CLI entry point for pdf2md-agent."""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time

import pymupdf
from pathlib import Path

from pdf2md_agent import __about__
from pdf2md_agent.cache import (
    CacheLayout,
    CacheNoCacheFlags,
    atomic_write_text,
    check_meta_matches,
    read_meta,
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
    REQUEST_TIMEOUT_SECONDS,
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
from PIL import Image

from pdf2md_agent.pdf_renderer import PageImage, render_pdf
from pdf2md_agent.post_stream import StitchMode, stitch_pages
from pdf2md_agent.render_skip import (
    maybe_skip_render as _maybe_skip_render,
)
from pdf2md_agent.vision import make_vision_llm

log = logging.getLogger("pdf2md-agent")


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


def _cache_key_for_pdf(pdf: Path) -> str:
    """Return a deterministic cache directory name for ``pdf``.

    Uses the PDF's stem when it is short, free of path separators, and not
    a Windows-reserved name. For long stems, names that contain ``/`` (e.g.
    when the PDF lives under a deeply-nested tree), or Windows-reserved
    stems on a Windows host, the cache key is a 16-character SHA-256
    digest of the absolute PDF path — deterministic per file, never
    collides between different absolute paths.
    """
    abs_path = pdf.resolve()
    stem = _safe_cache_stem(abs_path.stem)
    if (
        0 < len(stem) <= 60
        and "/" not in abs_path.stem
        and "\\" not in abs_path.stem
        and (sys.platform != "win32" or stem.upper() not in _WINDOWS_RESERVED_NAMES)
    ):
        return stem
    import hashlib
    return hashlib.sha256(str(abs_path).encode("utf-8")).hexdigest()[:16]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf2md-agent",
        description=(
            "Render every page of a PDF to an image and feed it through a "
            "CrewAI pipeline (extract → format → summarize) to produce "
            "language-preserving Markdown.\n\n"
            "Stages: render → extract → format → summarize → stitch.\n\n"
            "Cache: per-resource (render/text/resized/extract/format/summary) "
            "is reused by default and gated by meta.json fingerprint validation "
            "(pdf_path, dpi, with_summary, pages, model, persona_version). "
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
            "ranges, e.g. '1-5,8,11-13'. Pages are 1-based; output preserves "
            "original page numbers and is ordered by document position. "
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
            "format/summary). Equivalent to passing all six --no-cache-* "
            "flags."
        ),
    )

    features = parser.add_argument_group(
        "Feature disable",
        "Optional features; each --no-<feature> opts a single feature out.",
    )
    features.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable cross-page running summary (process each page independently).",
    )
    features.add_argument(
        "--no-text-hint",
        action="store_true",
        help="Disable feeding the PDF's native text layer to the extractor.",
    )
    features.add_argument(
        "--no-fallback-to-text",
        action="store_true",
        default=False,
        dest="no_fallback_to_text",
        help=(
            "On retry exhaustion, raise instead of falling back to the PDF's "
            "native text layer. Default: fallback enabled."
        ),
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
        "--retry-initial-delay",
        type=float,
        default=None,
        help=(
            "Initial retry delay in seconds (Fibonacci base unit). Overrides "
            "PDF2MD_AGENT_RETRY_INITIAL_DELAY. Default: 1.0."
        ),
    )
    tuning.add_argument(
        "--retry-max-delay",
        type=float,
        default=None,
        help=(
            "Per-attempt retry delay cap in seconds (Fibonacci growth cap). "
            "Overrides PDF2MD_AGENT_RETRY_MAX_DELAY. Default: 900.0 (15 min)."
        ),
    )
    tuning.add_argument(
        "--retry-jitter",
        type=float,
        default=None,
        help=(
            "Jitter ratio in [0.0, 1.0] applied to each retry delay to avoid "
            "thundering-herd. Overrides PDF2MD_AGENT_RETRY_JITTER. Default: 0.25."
        ),
    )
    tuning.add_argument(
        "--image-long-side",
        type=_positive_int_type("image-long-side", 64),
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
    tuning.add_argument(
        "--image-quality",
        type=_positive_int_type("image-quality", 1),
        default=None,
        metavar="Q",
        help=(
            "JPEG quality (1-100) used when the runner downsamples page "
            "images. Higher values preserve detail but enlarge the per-call "
            "token cost. 75-95 is the practical sweet spot. Overrides "
            "PDF2MD_AGENT_IMAGE_JPEG_QUALITY. Default: 85."
        ),
    )
    tuning.add_argument(
        "--max-summary-chars",
        type=_positive_int_type("max-summary-chars", 100),
        default=None,
        metavar="N",
        help=(
            "Maximum running-summary size (characters) fed into the next "
            "page's extract call and produced by the summarizer. Overrides "
            "PDF2MD_AGENT_MAX_SUMMARY_CHARS. Default: 800."
        ),
    )
    tuning.add_argument(
        "--ctx-limit",
        type=_positive_int_type("ctx-limit", 256),
        default=None,
        metavar="TOK",
        help=(
            "Model context-window token limit the runner budgets against. "
            "Used only when PDF2MD_AGENT_CTX_LIMIT is wrong. Default: 2013."
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

    diagnostic = parser.add_argument_group(
        "Diagnostic",
        "Inspection flags; rarely needed in normal runs.",
    )
    diagnostic.add_argument(
        "--model",
        default=MODEL_NAME,
        help=(
            "Model name to record in meta.json for fingerprint validation. "
            "Defaults to PDF2MD_AGENT_MODEL (default: MiniMax-M3)."
        ),
    )
    diagnostic.add_argument(
        "--persona-version",
        default=PERSONA_VERSION,
        help=(
            "Persona fingerprint (16-char hex) recorded in meta.json. The "
            "runner refuses to re-use cache when this drifts. Defaults to "
            "the SHA-256[:16] of the active persona strings."
        ),
    )

    parser.add_argument(
        "-V", "--version",
        action=_VersionAction,
        nargs=0,
        help="Print the pdf2md-agent version and exit.",
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
        root = override if override is not None else Path(".pdf2md-agent-cache") / _cache_key_for_pdf(pdf)
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
    # ``--max-retries 0`` (or env PDF2MD_AGENT_MAX_RETRIES=0) → unlimited.
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
            initial_delay=(
                args.retry_initial_delay
                if args.retry_initial_delay is not None
                else RETRY_INITIAL_DELAY
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


def _render_pages(
    *,
    pdf: Path,
    render_target: Path,
    dpi: int,
    resolved_pages: list[int] | None,
    keep_intermediates: bool,
    no_cache_render: bool,
    no_cache_text: bool,
) -> list[PageImage]:
    """Render the PDF, optionally reusing per-page PNG/text cache.

    When ``keep_intermediates`` is True and the no-cache flags are unset,
    pages whose PNG/text are already on disk are returned without touching
    PyMuPDF — that's the trust-cache fast path. With either flag set, the
    pipeline always re-renders / re-extracts.
    """
    if not keep_intermediates or no_cache_render or no_cache_text:
        return render_pdf(pdf, render_target, dpi=dpi, pages=resolved_pages)

    layout = CacheLayout(
        root=render_target.parent,
        pages_dir=render_target,
        summary_path=render_target.parent / "summary.json",
        meta_path=render_target.parent / "meta.json",
    )

    target_pages: list[int] = (
        list(resolved_pages) if resolved_pages is not None
        else list(range(1, _pdf_page_count(pdf) + 1))
    )
    missing: list[int] = [
        n for n in target_pages if _maybe_skip_render(layout, n, dpi) is None
    ]
    if missing:
        render_pdf(pdf, render_target, dpi=dpi, pages=missing)
    pages: list[PageImage] = []
    for n in target_pages:
        png = layout.page_png_path(n)
        with Image.open(png) as img:
            pages.append(PageImage(
                page_number=n,
                width=img.width,
                height=img.height,
                image_path=png,
            ))
    return pages


def _pdf_page_count(pdf: Path) -> int:
    doc = pymupdf.open(pdf)
    try:
        return doc.page_count
    finally:
        doc.close()


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
        existing_meta = read_meta(layout.meta_path)
        # ``--no-cache-all`` discards every cached output, so the on-disk
        # fingerprint (about to be overwritten by ``write_meta`` below) is
        # no longer load-bearing — refusing on drift would create a circular
        # error the user can't escape.
        if existing_meta is not None and not no_cache.all():
            reasons = check_meta_matches(
                existing_meta,
                pdf=str(args.pdf.resolve()),
                dpi=args.dpi,
                with_summary=with_summary,
                pages=resolved_pages,
                model=args.model,
                persona_version=args.persona_version,
            )
            if reasons:
                for r in reasons:
                    print(f"error: cache invalid: {r}", file=sys.stderr)
                print(
                    "error: meta.json fingerprint drift detected. "
                    "re-run with --no-cache-all or wipe "
                    f"{layout.root} to rebuild the cache.",
                    file=sys.stderr,
                )
                return 1
        write_meta(
            layout.meta_path,
            pdf=args.pdf,
            dpi=args.dpi,
            with_summary=with_summary,
            pages=resolved_pages,
            model=args.model,
            persona_version=args.persona_version,
        )
        if not with_summary and layout.summary_path.exists():
            layout.summary_path.unlink()

    log.info("rendering PDF to PNGs at %d dpi%s...", args.dpi, " (subset)" if resolved_pages else "")
    pages = _render_pages(
        pdf=args.pdf,
        render_target=render_target,
        dpi=args.dpi,
        resolved_pages=resolved_pages,
        keep_intermediates=keep_intermediates,
        no_cache_render=no_cache.render,
        no_cache_text=no_cache.text,
    )
    log.info("rendered %d page(s) to %s", len(pages), render_target)

    log.info("running pipeline: extract + format%s", " + summarize" if with_summary else "")
    llm = make_vision_llm()
    log.info(
        "  retry:           max_attempts=%s, initial_delay=%.1fs, fibonacci, max_delay=%.1fs, jitter=±%.0f%%",
        retry_config.max_attempts if retry_config.max_attempts is not None else "\u221e",
        retry_config.initial_delay,
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
        request_timeout_seconds=(
            args.request_timeout
            if args.request_timeout is not None
            else REQUEST_TIMEOUT_SECONDS
        ),
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
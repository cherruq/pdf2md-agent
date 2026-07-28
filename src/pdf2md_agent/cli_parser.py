from __future__ import annotations
import argparse
import tempfile
import sys
from pathlib import Path
from typing import Callable
from pdf2md_agent import __about__
from pdf2md_agent.cache import CacheNoCacheFlags, CacheLayout
from pdf2md_agent.config import MODEL_NAME
from pdf2md_agent.crew.agents import PERSONA_VERSION
from pdf2md_agent.pages import parse_page_spec
from pdf2md_agent.post_stream import StitchMode


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
            "Initial retry delay in seconds (Fibonacci base unit). Must be "
            "> 0; a zero or negative value is rejected. Overrides "
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
            "Overrides PDF2MD_AGENT_CTX_LIMIT. Default: probed from "
            "OPENAI_BASE_URL/models, or the hardcoded value for the "
            "active model."
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

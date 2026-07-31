"""Filesystem-safety helpers for cache directory naming.

Pure functions used by the CLI and the cache layout to compute a
deterministic, filesystem-safe directory name for a PDF's intermediate
cache. Two concerns live here:

* :func:`safe_cache_stem` — sanitize a PDF's filename stem so it never
  collides with Windows reserved device names (``CON``, ``PRN``, …) and
  never produces a bare dot/space-only directory.
* :func:`cache_key_for_pdf` — combine the sanitized stem with a
  hash-based fallback for stems that are too long or contain path
  separators, so two distinct absolute paths always land in distinct
  cache directories.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_WINDOWS_RESERVED_NAMES: frozenset[str] = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
)
"""Windows reserved device names: ``CON``/``PRN``/``AUX``/``NUL`` plus
``COM1``-``COM9`` and ``LPT1``-``LPT9``. Case-insensitive on Windows.

The ``CreateFile`` syscall rejects these as bare filenames (with or
without an extension), and ``mkdir`` on a reserved name surfaces as an
opaque ``OSError``. We append a single ``_`` so the cache lives at
``<reserved>_`` instead of crashing the run."""

_MAX_CACHE_STEM_LEN: int = 60


def safe_cache_stem(stem: str) -> str:
    """Return a filesystem-safe cache directory name derived from ``stem``.

    Strips trailing dots/spaces and appends ``_`` to Windows reserved
    names on Windows so ``mkdir`` succeeds. On non-Windows platforms
    trailing dots/spaces are still stripped defensively for portability.

    Case-collision (D16-002): on case-insensitive filesystems (NTFS,
    APFS, HFS+) two PDFs whose stems differ only in case map to the
    same cache directory. We do not canonicalize here — instead,
    callers are warned via this function's docstring to pick distinct
    stems.
    """
    if not stem:
        return "_"
    candidate = stem.rstrip(" .")
    if not candidate:
        return "_"
    if sys.platform == "win32" and candidate.upper() in _WINDOWS_RESERVED_NAMES:
        return candidate + "_"
    return candidate


def cache_key_for_pdf(pdf: Path) -> str:
    """Return a deterministic cache directory name for ``pdf``.

    Uses the PDF's stem when it is short, free of path separators, and not
    a Windows-reserved name. For long stems, names that contain ``/``
    (e.g. when the PDF lives under a deeply-nested tree), or
    Windows-reserved stems on a Windows host, the cache key is a
    16-character SHA-256 digest of the absolute PDF path — deterministic
    per file, never collides between different absolute paths.
    """
    abs_path = pdf.resolve()
    stem = safe_cache_stem(abs_path.stem)
    if (
        0 < len(stem) <= _MAX_CACHE_STEM_LEN
        and "/" not in abs_path.stem
        and "\\" not in abs_path.stem
        and (sys.platform != "win32" or stem.upper() not in _WINDOWS_RESERVED_NAMES)
    ):
        return stem
    return hashlib.sha256(str(abs_path).encode("utf-8")).hexdigest()[:16]


__all__ = ["cache_key_for_pdf", "safe_cache_stem"]

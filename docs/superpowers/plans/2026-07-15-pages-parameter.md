# `--pages` Parameter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--pages` / `-p` flag to `convertpdf convert` that accepts comma-separated page numbers and ranges (e.g. `1-5,8,11-13`) and converts only those pages, preserving original page numbers throughout the pipeline.

**Architecture:** New pure module `convertpdf.pages` exposes `parse_page_spec` (argparse `type=`) and `resolve_pages` (dedupe/sort/validate). `render_pdf` gains an optional `pages: list[int] | None` parameter that drives PyMuDF iteration; output filenames keep the original 1-based page number. `cmd_convert` opens the PDF to count pages, calls `resolve_pages`, and threads the result through. CLI syntax errors fail fast at argparse time; out-of-range pages error after opening the PDF.

**Tech Stack:** Python ≥3.10, PyMuDF (`pymupdf`), `argparse`, `pytest`. Existing project uses `pythonpath = ["src"]` and `testpaths = ["tests"]`.

---

## File Structure

**New files:**
- `src/convertpdf/pages.py` — pure parser + resolver, no I/O, no PyMuDF dependency.
- `tests/test_pages.py` — unit tests for the parser and resolver.

**Modified files:**
- `src/convertpdf/pdf_renderer.py` — `render_pdf()` accepts `pages: list[int] | None`.
- `src/convertpdf/cache.py` — `write_meta()` accepts `pages: list[int] | None`.
- `src/convertpdf/cli.py` — `build_parser()` adds `--pages`; `cmd_convert()` threads it.
- `tests/test_pdf_renderer.py` — add subset-rendering tests.

**Untouched (despite sounding related):**
- `src/convertpdf/crew/runner.py` — already iterates `pages` and uses `page.page_number`; works as-is.
- `src/convertpdf/cache.py:CacheLayout` — page paths are derived from `page.page_number`, so sparse page sets Just Work.

---

## Task 0: Project Setup (git init + verify environment)

**Files:**
- Create: `/home/ss/workspace/my/convertpdf/.gitignore` (verify existing one is enough)

- [ ] **Step 1: Verify `.gitignore` exists and is reasonable**

```bash
cat /home/ss/workspace/my/convertpdf/.gitignore
```

Expected: contains `.venv/`, `.convertpdf-cache/`, `__pycache__/`, `.env`, etc. If anything important is missing (e.g. `.pytest_cache/`, `*.egg-info/`), add it. Do NOT modify unrelated lines.

- [ ] **Step 2: Initialize git repo and make initial commit**

```bash
cd /home/ss/workspace/my/convertpdf
git init
git add .
git status   # review what's staged — should NOT include .venv, .env, caches
git commit -m "chore: initial commit (baseline before --pages feature)"
```

If `git status` shows `.venv`, `.env`, `.convertpdf-cache`, `.pytest_cache`, or `__pycache__` staged, STOP — the `.gitignore` is broken. Fix it before committing.

- [ ] **Step 3: Verify the test suite runs green at baseline**

```bash
cd /home/ss/workspace/my/convertpdf
.venv/bin/python -m pytest -q
```

Expected: all tests pass. Note the count for comparison after changes.

---

## Task 1: `parse_page_spec` — parser + tests (TDD)

**Files:**
- Create: `src/convertpdf/pages.py`
- Create: `tests/test_pages.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pages.py` with the following content (do not add anything else yet):

```python
"""Tests for convertpdf.pages."""
from __future__ import annotations

import argparse
import pytest

from convertpdf.pages import parse_page_spec, resolve_pages


# --- parse_page_spec ---------------------------------------------------------

def test_parse_single_page() -> None:
    assert parse_page_spec("3") == [3]


def test_parse_simple_range() -> None:
    assert parse_page_spec("1-5") == [1, 2, 3, 4, 5]


def test_parse_range_and_list() -> None:
    assert parse_page_spec("1-5,8,11-13") == [1, 2, 3, 4, 5, 8, 11, 12, 13]


def test_parse_overlapping_ranges_dedupe() -> None:
    assert parse_page_spec("1-5,1-3") == [1, 2, 3, 4, 5]


def test_parse_tolerates_whitespace() -> None:
    assert parse_page_spec(" 1 - 5 , 8 ") == [1, 2, 3, 4, 5, 8]


@pytest.mark.parametrize(
    "bad",
    ["", "abc", "0", "-3", "5-3", "3-", "-5", "1-5,abc", "1.5", "1,,3"],
)
def test_parse_rejects_bad_input(bad: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_page_spec(bad)
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
cd /home/ss/workspace/my/convertpdf
.venv/bin/python -m pytest tests/test_pages.py -v
```

Expected: all tests fail with `ModuleNotFoundError: No module named 'convertpdf.pages'` (or `ImportError`).

- [ ] **Step 3: Implement `parse_page_spec` in `src/convertpdf/pages.py`**

Create the file with this exact content:

```python
"""Page-spec parsing and resolution for the --pages CLI flag.

Two pure functions, no I/O:

- :func:`parse_page_spec` is the argparse ``type=`` callable; it validates
  syntax and raises :class:`argparse.ArgumentTypeError` on bad input so
  the CLI rejects malformed specs before opening the PDF.

- :func:`resolve_pages` dedupes, sorts, and validates a parsed page list
  against the PDF's actual page count; raises :class:`ValueError` with a
  user-facing message on out-of-range pages.
"""
from __future__ import annotations

import argparse
import re

# Regexes (anchored, whitespace-tolerant). The parse step only enforces
# "positive integer" for tokens; the upper bound is checked later by
# resolve_pages against the actual PDF page count.
_TOKEN_RE = re.compile(r"^\s*(\d+)\s*$")
_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")


def _err(msg: str) -> argparse.ArgumentTypeError:
    return argparse.ArgumentTypeError(msg)


def parse_page_spec(spec: str) -> list[int]:
    """Parse a --pages value like ``'1-5,8,11-13'`` into ``[1,2,3,4,5,8,11,12,13]``.

    Comma-separated items, each either a single page number (``8``) or a
    range (``1-5``). Whitespace around numbers and around the ``-`` is
    tolerated.

    Page numbers must be positive integers (``>= 1``); the upper bound is
    not enforced here (that's :func:`resolve_pages`'s job, since it
    depends on the actual PDF).

    Raises :class:`argparse.ArgumentTypeError` on any malformed input.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise _err(f"expected integer or N-M, got {spec!r}")

    pages: list[int] = []
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            raise _err(f"expected integer or N-M, got {raw_item!r}")

        m_range = _RANGE_RE.match(item)
        if m_range:
            start = int(m_range.group(1))
            end = int(m_range.group(2))
            if start == 0 or end == 0:
                raise _err(f"page numbers must be >= 1, got {item!r}")
            if start > end:
                raise _err(f"range start must be <= end, got {item!r}")
            pages.extend(range(start, end + 1))
            continue

        m_token = _TOKEN_RE.match(item)
        if m_token:
            n = int(m_token.group(1))
            if n == 0:
                raise _err(f"page numbers must be >= 1, got {item!r}")
            pages.append(n)
            continue

        # Neither a plain integer nor a range -> malformed input.
        raise _err(f"expected integer or N-M, got {item!r}")

    return pages
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
cd /home/ss/workspace/my/convertpdf
.venv/bin/python -m pytest tests/test_pages.py -v
```

Expected: 15 parser tests pass (5 individual + 10 parametrized cases in `test_parse_rejects_bad_input`). The 3 `resolve_pages` tests fail with `ImportError` because that function doesn't exist yet — that's expected; they're fixed in Task 2.

- [ ] **Step 5: Commit**

```bash
cd /home/ss/workspace/my/convertpdf
git add src/convertpdf/pages.py tests/test_pages.py
git commit -m "feat(pages): add parse_page_spec for --pages CLI flag"
```

---

## Task 2: `resolve_pages` — dedupe/sort/validate + tests (TDD)

**Files:**
- Modify: `src/convertpdf/pages.py`
- Modify: `tests/test_pages.py`

- [ ] **Step 1: Add the failing tests to `tests/test_pages.py`**

Append the following to the end of `tests/test_pages.py`:

```python
# --- resolve_pages -----------------------------------------------------------

def test_resolve_sorts_and_dedupes() -> None:
    assert resolve_pages([3, 1, 2], total=10) == [1, 2, 3]
    assert resolve_pages([5, 5, 5], total=10) == [5]
    assert resolve_pages([7, 1, 7, 3, 1], total=10) == [1, 3, 7]


def test_resolve_passes_when_in_range() -> None:
    assert resolve_pages([1, 2, 3], total=3) == [1, 2, 3]
    assert resolve_pages([1], total=1) == [1]


def test_resolve_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"page 99 out of range \(PDF has 10 pages\)"):
        resolve_pages([99], total=10)
    with pytest.raises(ValueError, match=r"page 4 out of range \(PDF has 3 pages\)"):
        resolve_pages([1, 2, 3, 4], total=3)
```

- [ ] **Step 2: Run and confirm they fail**

```bash
cd /home/ss/workspace/my/convertpdf
.venv/bin/python -m pytest tests/test_pages.py -v -k resolve
```

Expected: 3 errors / failures, all pointing at `resolve_pages` not existing.

- [ ] **Step 3: Add `resolve_pages` to `src/convertpdf/pages.py`**

Append the following at the end of `src/convertpdf/pages.py` (do not touch the existing `parse_page_spec`):

```python
def resolve_pages(spec: list[int], total: int) -> list[int]:
    """Dedupe, sort, and validate a parsed page list against ``total``.

    Returns a new list of unique page numbers in ascending order, all
    within ``[1, total]``.

    Raises :class:`ValueError` on the first out-of-range page
    encountered, with a message of the form ``"page N out of range (PDF
    has M pages)"`` so the CLI can surface it directly.

    An empty result cannot arise: :func:`parse_page_spec` guarantees
    each item is a positive integer, and dedupe of a non-empty list is
    non-empty. No defensive empty-list check is needed.
    """
    if total < 1:
        raise ValueError(f"PDF has {total} pages; nothing to convert")

    out = sorted(set(spec))
    for n in out:
        if n < 1 or n > total:
            raise ValueError(f"page {n} out of range (PDF has {total} pages)")
    return out
```

- [ ] **Step 4: Run all `tests/test_pages.py` and confirm they pass**

```bash
cd /home/ss/workspace/my/convertpdf
.venv/bin/python -m pytest tests/test_pages.py -v
```

Expected: 18 tests pass (15 parser + 3 resolver).

- [ ] **Step 5: Commit**

```bash
cd /home/ss/workspace/my/convertpdf
git add src/convertpdf/pages.py tests/test_pages.py
git commit -m "feat(pages): add resolve_pages with range validation"
```

---

## Task 3: `render_pdf` accepts `pages=` parameter (TDD)

**Files:**
- Modify: `src/convertpdf/pdf_renderer.py`
- Modify: `tests/test_pdf_renderer.py`

- [ ] **Step 1: Add failing tests to `tests/test_pdf_renderer.py`**

Append the following to the end of `tests/test_pdf_renderer.py`:

```python
def test_render_pdf_subset_writes_only_requested_pages(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "tiny.pdf", pages=3)
    out = tmp_path / "pages"
    out.mkdir()

    pages = render_pdf(pdf, out, dpi=72, pages=[2])

    assert len(pages) == 1
    assert pages[0].page_number == 2
    assert (out / "page_0002.png").exists()
    assert (out / "page_0002_text.txt").exists()
    assert not (out / "page_0001.png").exists()
    assert not (out / "page_0003.png").exists()


def test_render_pdf_subset_preserves_original_page_numbers(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "tiny.pdf", pages=5)
    out = tmp_path / "pages"
    out.mkdir()

    pages = render_pdf(pdf, out, dpi=72, pages=[3, 1])

    # Sorted ascending in the returned list.
    assert [p.page_number for p in pages] == [1, 3]
    # But output filenames use the ORIGINAL page number.
    assert (out / "page_0001.png").exists()
    assert (out / "page_0003.png").exists()
    assert not (out / "page_0002.png").exists()
    assert not (out / "page_0004.png").exists()
    assert not (out / "page_0005.png").exists()


def test_render_pdf_subset_full_coverage(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "tiny.pdf", pages=2)
    out = tmp_path / "pages"
    out.mkdir()

    pages = render_pdf(pdf, out, dpi=72, pages=[1, 2])

    assert [p.page_number for p in pages] == [1, 2]
    assert (out / "page_0001.png").exists()
    assert (out / "page_0002.png").exists()
```

- [ ] **Step 2: Run and confirm the new tests fail**

```bash
cd /home/ss/workspace/my/convertpdf
.venv/bin/python -m pytest tests/test_pdf_renderer.py -v -k subset
```

Expected: 3 failures with `TypeError: render_pdf() got an unexpected keyword argument 'pages'` (or similar).

- [ ] **Step 3: Extend `render_pdf` with the `pages=` parameter**

Edit `src/convertpdf/pdf_renderer.py`. Replace the existing function signature and loop with the version below. Keep all surrounding code (the `PageImage` dataclass and `read_page_text`) untouched.

Replace this block:

```python
def render_pdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = 144,
    prefix: str = "page",
) -> list[PageImage]:
    """Render every page of ``pdf_path`` into a PNG under ``output_dir``.

    For each page, also writes a sibling ``{prefix}_{NNNN}_text.txt`` containing
    the PDF's native text layer (empty for scanned pages).

    Returns the pages in document order. Caller is responsible for ``output_dir``
    existing; the function writes into it but does not create it.
    """
    doc = pymupdf.open(pdf_path)
    try:
        pages: list[PageImage] = []
        zoom = dpi / 72
        matrix = pymupdf.Matrix(zoom, zoom)
        for index, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            png = output_dir / f"{prefix}_{index:04d}.png"
            pix.save(png)
            text = output_dir / f"{prefix}_{index:04d}_text.txt"
            text.write_text(page.get_text("text"), encoding="utf-8")
            pages.append(
                PageImage(
                    page_number=index,
                    width=pix.width,
                    height=pix.height,
                    image_path=png,
                )
            )
        return pages
    finally:
        doc.close()
```

With:

```python
def render_pdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = 144,
    prefix: str = "page",
    pages: list[int] | None = None,
) -> list[PageImage]:
    """Render ``pdf_path`` into per-page PNGs under ``output_dir``.

    If ``pages`` is ``None`` (default), renders every page in document
    order. If ``pages`` is a list of 1-based page numbers, renders only
    those pages (still in document order — the list is sorted internally)
    and skips the rest. Output filenames always use the **original**
    1-based page number, so cache directories are stable across calls
    with different ``pages`` selections.

    For each rendered page, also writes a sibling
    ``{prefix}_{NNNN}_text.txt`` containing the PDF's native text layer
    (empty for scanned pages).

    Returns the pages in document order. Caller is responsible for
    ``output_dir`` existing; the function writes into it but does not
    create it.
    """
    doc = pymupdf.open(pdf_path)
    try:
        pages_out: list[PageImage] = []
        zoom = dpi / 72
        matrix = pymupdf.Matrix(zoom, zoom)

        if pages is None:
            page_numbers = list(range(1, doc.page_count + 1))
        else:
            # Sort + dedupe; preserve original page numbers.
            page_numbers = sorted(set(pages))

        for page_number in page_numbers:
            page = doc.load_page(page_number - 1)  # PyMuDF is 0-indexed
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            png = output_dir / f"{prefix}_{page_number:04d}.png"
            pix.save(png)
            text = output_dir / f"{prefix}_{page_number:04d}_text.txt"
            text.write_text(page.get_text("text"), encoding="utf-8")
            pages_out.append(
                PageImage(
                    page_number=page_number,
                    width=pix.width,
                    height=pix.height,
                    image_path=png,
                )
            )
        return pages_out
    finally:
        doc.close()
```

- [ ] **Step 4: Run all `tests/test_pdf_renderer.py` and confirm they pass**

```bash
cd /home/ss/workspace/my/convertpdf
.venv/bin/python -m pytest tests/test_pdf_renderer.py -v
```

Expected: all 6 tests pass (3 baseline + 3 new). Also confirm existing behavior is unchanged by running the 3 baseline tests individually and checking they still pass.

- [ ] **Step 5: Run the full test suite to check no regressions**

```bash
cd /home/ss/workspace/my/convertpdf
.venv/bin/python -m pytest -q
```

Expected: all tests pass; total count = 6 (renderer) + 18 (pages) + 2 (vision) + baseline = 26-ish (depending on what's in `test_vision.py`).

- [ ] **Step 6: Commit**

```bash
cd /home/ss/workspace/my/convertpdf
git add src/convertpdf/pdf_renderer.py tests/test_pdf_renderer.py
git commit -m "feat(renderer): accept pages= parameter to render PDF subsets"
```

---

## Task 4: `write_meta` records which pages were converted

**Files:**
- Modify: `src/convertpdf/cache.py`

There is no existing test file for `cache.py`, so this task adds a minimal one alongside the change.

- [ ] **Step 1: Add a failing test for `write_meta` with `pages`**

Create `tests/test_cache.py` with:

```python
"""Tests for convertpdf.cache."""
from __future__ import annotations

import json
from pathlib import Path

from convertpdf.cache import write_meta


def test_write_meta_without_pages(tmp_path: Path) -> None:
    meta = tmp_path / "meta.json"
    write_meta(meta, pdf=tmp_path / "x.pdf", dpi=144, with_summary=True)
    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["pages"] is None
    assert payload["dpi"] == 144
    assert payload["with_summary"] is True


def test_write_meta_with_pages(tmp_path: Path) -> None:
    meta = tmp_path / "meta.json"
    write_meta(
        meta,
        pdf=tmp_path / "x.pdf",
        dpi=144,
        with_summary=False,
        pages=[1, 2, 5],
    )
    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["pages"] == [1, 2, 5]
```

- [ ] **Step 2: Run and confirm the second test fails**

```bash
cd /home/ss/workspace/my/convertpdf
.venv/bin/python -m pytest tests/test_cache.py -v
```

Expected: `test_write_meta_with_pages` fails because the current `write_meta` doesn't accept `pages`. (`test_write_meta_without_pages` should pass since the current signature works — but it asserts `pages is None` which the current writer doesn't include; if it fails too, that's also fine, both will pass after the fix.)

- [ ] **Step 3: Extend `write_meta` with the `pages` field**

Edit `src/convertpdf/cache.py`. Replace the existing `write_meta` function with:

```python
def write_meta(
    meta_path: Path,
    *,
    pdf: Path,
    dpi: int,
    with_summary: bool,
    pages: list[int] | None = None,
) -> None:
    meta_path.write_text(
        json.dumps(
            {
                "pdf": str(pdf),
                "dpi": dpi,
                "with_summary": with_summary,
                "pages": pages,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
```

- [ ] **Step 4: Run and confirm both tests pass**

```bash
cd /home/ss/workspace/my/convertpdf
.venv/bin/python -m pytest tests/test_cache.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/ss/workspace/my/convertpdf
git add src/convertpdf/cache.py tests/test_cache.py
git commit -m "feat(cache): record converted page list in meta.json"
```

---

## Task 5: Add `--pages` flag to the CLI parser

**Files:**
- Modify: `src/convertpdf/cli.py`

CLI parser wiring has no automated test in this codebase (no CLI-level test exists today). The integration test comes from Task 6 + the manual smoke test. This task is a pure argparse change with no behavioral risk beyond what Task 6 exercises end-to-end.

- [ ] **Step 1: Add the `--pages` argument**

In `src/convertpdf/cli.py`, add the following import near the top with the other `convertpdf.*` imports:

```python
from convertpdf.pages import parse_page_spec
```

Then in `build_parser()`, immediately after the `--dpi` argument block (right before the `--no-intermediates` argument), insert:

```python
    cv.add_argument(
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
```

- [ ] **Step 2: Verify the CLI rejects malformed `--pages`**

```bash
cd /home/ss/workspace/my/convertpdf
.venv/bin/python -m convertpdf convert /tmp/nope.pdf -o /tmp/out.md --pages 'abc' 2>&1 | tail -5
echo "exit code: $?"
```

Expected: stderr contains something like `invalid value 'abc' for '--pages'`, exit code 2.

- [ ] **Step 3: Verify the help text shows the new flag**

```bash
cd /home/ss/workspace/my/convertpdf
.venv/bin/python -m convertpdf convert --help
```

Expected: `--pages SPEC` line appears in the output with the help text from Step 1.

- [ ] **Step 4: Commit**

```bash
cd /home/ss/workspace/my/convertpdf
git add src/convertpdf/cli.py
git commit -m "feat(cli): add --pages argument to convert subcommand"
```

---

## Task 6: Wire `--pages` through `cmd_convert`

**Files:**
- Modify: `src/convertpdf/cli.py`

- [ ] **Step 1: Update `cmd_convert` to resolve and pass `pages`**

In `src/convertpdf/cli.py`, replace the entire `cmd_convert` function body with the version below. Keep the imports and `build_parser` unchanged (the import for `parse_page_spec` was added in Task 5; you also need to add imports for `pymupdf` and `resolve_pages`).

Update the imports at the top of `cli.py`. After the existing `from convertpdf.cache import CacheLayout, write_meta` line, ensure these imports exist (add what's missing):

```python
import pymupdf

from convertpdf.pages import parse_page_spec, resolve_pages
```

(`parse_page_spec` was added in Task 5; `pymupdf` and `resolve_pages` are new here.)

Then replace the body of `cmd_convert`. The new body:

```python
def cmd_convert(args: argparse.Namespace) -> int:
    if not args.pdf.exists():
        print(f"error: input PDF not found: {args.pdf}", file=sys.stderr)
        return 1

    started = time.monotonic()
    keep_intermediates = not args.no_intermediates
    with_summary = not args.no_summary

    layout, render_target = _resolve_layout(args.pdf, args.intermediates_dir, keep_intermediates)

    # Resolve --pages against the PDF's actual page count so out-of-range
    # errors surface before any rendering work happens.
    resolved_pages: list[int] | None
    if args.pages is None:
        resolved_pages = None
    else:
        doc = pymupdf.open(args.pdf)
        try:
            resolved_pages = resolve_pages(args.pages, doc.page_count)
        finally:
            doc.close()

    log.info("converting %s", args.pdf)
    log.info("  output:          %s", args.output)
    log.info("  cache:           %s", layout.root if keep_intermediates else "(tempdir, discarded)")
    log.info("  dpi:             %d", args.dpi)
    log.info("  pages:           %s", "all" if resolved_pages is None else resolved_pages)
    log.info("  cross-page:      %s", "summary" if with_summary else "independent")
    log.info("  resume:          %s", "yes" if args.resume else "no")
    log.info("  text-hint:       %s", "on" if not args.no_text_hint else "off")

    if keep_intermediates:
        write_meta(
            layout.meta_path,
            pdf=args.pdf,
            dpi=args.dpi,
            with_summary=with_summary,
            pages=resolved_pages,
        )

    log.info("rendering PDF to PNGs at %d dpi%s...", args.dpi, " (subset)" if resolved_pages else "")
    pages = render_pdf(args.pdf, render_target, dpi=args.dpi, pages=resolved_pages)
    log.info("rendered %d page(s) to %s", len(pages), render_target)

    log.info("running pipeline: extract + format%s", " + summarize" if with_summary else "")
    llm = make_vision_llm()
    results = run_pipeline(
        pages=pages,
        layout=layout,
        with_summary=with_summary,
        resume=args.resume,
        text_hint=not args.no_text_hint,
        llm=llm,
    )

    markdown = "\n\n---\n\n".join(r.markdown for r in results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    elapsed = time.monotonic() - started
    log.info(
        "wrote %s — %d page(s), %s chars in %.1fs",
        args.output,
        len(results),
        f"{len(markdown):,}",
        elapsed,
    )
    return 0
```

- [ ] **Step 2: Verify the full unit test suite still passes**

```bash
cd /home/ss/workspace/my/convertpdf
.venv/bin/python -m pytest -q
```

Expected: same green count as Task 3 Step 5, no regressions. The CLI-level behavior is exercised in Task 7.

- [ ] **Step 3: Commit**

```bash
cd /home/ss/workspace/my/convertpdf
git add src/convertpdf/cli.py
git commit -m "feat(cli): wire --pages through cmd_convert with range validation"
```

---

## Task 7: End-to-end manual smoke test

This task verifies the whole pipeline runs against a real PDF. It produces no commits; it's a verification gate before reporting the feature done.

- [ ] **Step 1: Locate or create a real PDF for testing**

The repo has no committed sample PDF (only `test.md` is a markdown file). Either:

- Use any small PDF on your machine (3+ pages), OR
- Generate one with PyMuDF:

```bash
.venv/bin/python -c "
import pymupdf
doc = pymupdf.open()
for i in range(5):
    p = doc.new_page()
    p.insert_text((72, 72), f'smoke-test page {i+1}')
doc.save('/tmp/convertpdf-smoke.pdf')
print('wrote /tmp/convertpdf-smoke.pdf')
"
```

- [ ] **Step 2: Smoke test — full conversion (no `--pages`) still works**

```bash
cd /home/ss/workspace/my/convertpdf
.venv/bin/python -m convertpdf convert /tmp/convertpdf-smoke.pdf -o /tmp/out-all.md 2>&1 | tail -10
```

Expected: `wrote /tmp/out-all.md — N page(s), ...` appears. N matches the PDF's page count.

- [ ] **Step 3: Smoke test — subset conversion**

```bash
cd /home/ss/workspace/my/convertpdf
.venv/bin/python -m convertpdf convert /tmp/convertpdf-smoke.pdf -o /tmp/out-sub.md --pages '1-2,5' 2>&1 | tail -10
```

Expected: `wrote /tmp/out-sub.md — 3 page(s), ...` appears (pages 1, 2, 5 = 3 pages). The log line `pages: [1, 2, 5]` is visible earlier in the output.

- [ ] **Step 4: Smoke test — out-of-range error**

```bash
cd /home/ss/workspace/my/convertpdf
.venv/bin/python -m convertpdf convert /tmp/convertpdf-smoke.pdf -o /tmp/out-bad.md --pages '99' 2>&1 | tail -3
echo "exit code: $?"
```

Expected: stderr contains `page 99 out of range (PDF has 5 pages)`. Exit code 1. `/tmp/out-bad.md` is NOT created.

- [ ] **Step 5: Smoke test — malformed syntax**

```bash
cd /home/ss/workspace/my/convertpdf
.venv/bin/python -m convertpdf convert /tmp/convertpdf-smoke.pdf -o /tmp/out-bad.md --pages '5-3' 2>&1 | tail -3
echo "exit code: $?"
```

Expected: stderr contains `range start must be <= end`. Exit code 2 (argparse).

- [ ] **Step 6: Clean up the smoke-test PDF**

```bash
rm -f /tmp/convertpdf-smoke.pdf /tmp/out-all.md /tmp/out-sub.md /tmp/out-bad.md
```

- [ ] **Step 7: Final full test suite run**

```bash
cd /home/ss/workspace/my/convertpdf
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 8: Final commit (if any cleanup happened)**

If Step 1 used a generated PDF and you accidentally added it somewhere, clean up. There should be nothing to commit in this task.

```bash
cd /home/ss/workspace/my/convertpdf
git status
```

Expected: clean working tree. If anything is modified, revert or commit per repo policy.

---

## Summary

After all tasks, the project gains:

- `--pages` / `-p` flag on `convertpdf convert` accepting comma-separated page numbers and ranges.
- Strict validation: argparse rejects malformed syntax before opening the PDF; the CLI rejects out-of-range pages after a single page-count lookup.
- Original page numbers preserved in cache file names, markdown output, and pipeline summaries — so `--resume` and existing cache directories continue to work.
- 18 new unit tests (`tests/test_pages.py` + `tests/test_cache.py`) and 3 new renderer tests, all green.
- 5 atomic commits with conventional-commit-style messages.
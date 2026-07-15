# Add `--pages` Parameter to `convertpdf convert`

**Date:** 2026-07-15
**Status:** Approved (pending user review of this written spec)

## Goal

Let the user convert only a subset of a PDF's pages instead of always rendering every page.

Today, `convertpdf convert <pdf> -o <out>` renders all pages to PNG, runs the full CrewAI pipeline on each, and joins the results. For a 500-page document where the user only wants chapters 1 and 3, that wastes render time, vision-model calls, and wall-clock time. This change adds a single `--pages` / `-p` flag that accepts comma-separated page numbers and ranges.

## User-facing CLI

```
convertpdf convert <pdf> -o <out> [--pages SPEC] ...
```

`SPEC` grammar (case-insensitive whitespace tolerance):

```
spec     := item ("," item)*
item     := number | range
range    := number "-" number     # start <= end, both >= 1
number   := [1-9][0-9]*           # no zero, no negative, no signs
```

Examples accepted:

```
--pages 1-5
--pages 1-5,8
--pages 11-13,8,1-5
--pages "1 - 5 , 8"
```

`--pages` is optional. Not passing it preserves today's behavior (all pages).

### Error messages (exit code 1)

| Bad input                          | Message                                                                    |
| ---------------------------------- | -------------------------------------------------------------------------- |
| `--pages 'abc'`                    | `error: --pages 'abc': expected integer or N-M, got 'abc'`                  |
| `--pages '5-3'`                    | `error: --pages '5-3': range start must be <= end`                         |
| `--pages '0'`                      | `error: --pages '0': page numbers must be >= 1`                            |
| `--pages '3-'` / `--pages '-5'`    | `error: --pages '3-': expected integer or N-M, got '3-'`                   |
| `--pages '99'` (PDF has 10 pages)  | `error: --pages '99': page 99 out of range (PDF has 10 pages)`             |

## Architecture

### New module: `src/convertpdf/pages.py`

```python
def parse_page_spec(spec: str) -> list[int]:
    """Parse '1-5,8,11-13' -> [1,2,3,4,5,8,11,12,13].

    Validates syntax. Raises argparse.ArgumentTypeError on bad input so
    the CLI rejects malformed specs before opening the PDF.
    """

def resolve_pages(spec: list[int], total: int) -> list[int]:
    """Dedupe + sort + validate against total page count.

    Raises ValueError with 'page N out of range (PDF has M pages)' on
    the first out-of-range page encountered. (An empty result cannot
    arise: parse_page_spec already ensures each item is a positive
    integer, and dedupe of a non-empty list is non-empty.)
    """
```

`parse_page_spec` is the `type=` callable for the argparse `--pages` argument. Argparse calls it during `parse_args()`, so syntax errors fail fast with no PDF work done.

### Modified: `src/convertpdf/pdf_renderer.py`

`render_pdf()` gains an optional `pages` parameter:

```python
def render_pdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = 144,
    prefix: str = "page",
    pages: list[int] | None = None,  # NEW: 1-based page numbers, or None = all
) -> list[PageImage]:
    ...
```

Behavior:

- `pages=None` → iterate `enumerate(doc, start=1)` as today.
- `pages=[...]` → iterate the requested 1-based numbers; convert to 0-based via `doc.load_page(n - 1)`.
- Output filenames are **always** `page_{N:04d}.png` and `page_{N:04d}_text.txt` using the **original 1-based page number** (`N`), not a renumbered index.
- Returned `PageImage.page_number` is the original 1-based page number.

This keeps cache file names stable across `--pages` values, so `--resume` continues to work and the cache directory is never invalidated.

### Modified: `src/convertpdf/cli.py`

`build_parser()` adds:

```python
cv.add_argument(
    "-p", "--pages",
    type=parse_page_spec,
    default=None,
    metavar="SPEC",
    help=(
        "Subset of pages to convert. Accepts comma-separated pages and ranges, "
        "e.g. '1-5,8,11-13'. Output preserves original page numbers and is "
        "ordered by document position. Default: all pages."
    ),
)
```

`cmd_convert()` flow:

1. Parse args (any `--pages` syntax error already rejected by argparse).
2. Open the PDF briefly with PyMuDF to count pages: `total = doc.page_count`.
3. If `args.pages is None`, `resolved = None`. Otherwise, `resolved = resolve_pages(args.pages, total)`.
4. Pass `pages=resolved` to `render_pdf`.
5. If `keep_intermediates`, write `meta.json` with `"pages": resolved` (or `null` for all).
6. Rest of the flow (pipeline, runner, markdown join) unchanged.

CLI logging gains one line:

```
log.info("  pages:           %s", "all" if resolved is None else resolved)
```

### Modified: `src/convertpdf/cache.py`

`write_meta()` accepts an optional `pages` field:

```python
def write_meta(
    meta_path: Path,
    *,
    pdf: Path,
    dpi: int,
    with_summary: bool,
    pages: list[int] | None = None,  # NEW
) -> None:
    ...
    payload = {
        "pdf": str(pdf),
        "dpi": dpi,
        "with_summary": with_summary,
        "pages": pages,
    }
```

## Data flow (unchanged contract for downstream code)

```
CLI parses --pages
    │
    ▼
CLI opens PDF, calls resolve_pages(spec, total)
    │  (sorted, deduped, validated list of 1-based page numbers)
    ▼
render_pdf(pdf, out_dir, dpi=..., pages=resolved)
    │  writes page_NNNN.png / page_NNNN_text.txt using ORIGINAL page number N
    │  returns PageImage(page_number=N, ...)
    ▼
run_pipeline(pages=[...])        ← unchanged
    │  uses page.page_number for cache paths and summary chain
    ▼
"\n\n---\n\n".join(r.markdown ...) ← unchanged
```

Because `page.page_number` is preserved everywhere, the runner, cache, summary chain, and markdown output are byte-for-byte identical to the all-pages run for any page the user selected.

## Tests

### New: `tests/test_pages.py`

- `parse_page_spec("1-5,8,11-13")` → `[1,2,3,4,5,8,11,12,13]`
- `parse_page_spec(" 1 - 5 , 8 ")` → `[1,2,3,4,5,8]` (whitespace tolerated)
- `parse_page_spec("3")` → `[3]`
- `parse_page_spec("1-5,1-3")` → `[1,2,3,4,5]` (overlapping ranges dedupe)
- `parse_page_spec("0")` → raises `ArgumentTypeError`
- `parse_page_spec("-3")` → raises `ArgumentTypeError`
- `parse_page_spec("5-3")` → raises `ArgumentTypeError`
- `parse_page_spec("abc")` → raises `ArgumentTypeError`
- `parse_page_spec("3-")` → raises `ArgumentTypeError`
- `parse_page_spec("-5")` → raises `ArgumentTypeError`
- `parse_page_spec("")` → raises `ArgumentTypeError`
- `resolve_pages([3,1,2], total=10)` → `[1,2,3]`
- `resolve_pages([3,3,5], total=10)` → `[3,5]`
- `resolve_pages([99], total=10)` → raises `ValueError` matching `r"page 99 out of range"`
- `resolve_pages([1,2,3], total=3)` → `[1,2,3]` (boundary OK)

### Additions to `tests/test_pdf_renderer.py`

- `render_pdf(pdf, out, dpi=72, pages=[2])` on a 3-page PDF:
  - Returns exactly one `PageImage`.
  - That `PageImage.page_number == 2`.
  - `out/page_0002.png` exists; `out/page_0001.png` and `out/page_0003.png` do not.
- `render_pdf(pdf, out, dpi=72, pages=[3, 1])` on a 3-page PDF:
  - Returns two `PageImage`s in order `[1, 3]` (sorted).
  - Both have the correct original `page_number`.
- `render_pdf(pdf, out, dpi=72, pages=[1])` produces `page_0001.png` and `page_0001_text.txt`.

## Out of scope

- **Zero-based indexing.** Pages are 1-based everywhere, matching PyMuDF user-facing conventions and the existing code's `page_number=index+1` line.
- **Preserving user-specified output order.** Output is always document order (sorted ascending). The summary chain in `run_pipeline` is order-sensitive; reordering would silently corrupt the running summary.
- **TOC-based selection.** Future feature.
- **Renaming existing PNG files.** Cache filenames stay tied to the original page number so `--resume` keeps working with no migration.

## Migration / compatibility

No breaking changes. Default behavior (`--pages` omitted) is byte-for-byte identical to today. Existing cache directories are forward-compatible: `meta.json` gains an optional `pages` field; old meta files without it are read as if `pages=null`.

## Implementation order

1. Add `src/convertpdf/pages.py` with `parse_page_spec` + `resolve_pages`.
2. Add `tests/test_pages.py` covering parser and resolver.
3. Extend `render_pdf` with `pages=` parameter.
4. Extend `tests/test_pdf_renderer.py` with subset-rendering tests.
5. Extend `write_meta` with `pages=` field.
6. Extend `build_parser` + `cmd_convert` with `--pages` flag and pass-through.
7. Run full test suite + manual smoke test on a real PDF (e.g. `tests/test.md` source PDF) with both `--pages 1-2` and no flag.
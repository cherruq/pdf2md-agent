# pdf2md-agent

> Convert any PDF to clean, language-preserving Markdown — powered by a CrewAI
> vision pipeline against the MiniMax-M3 endpoint (or any OpenAI-compatible
> vision API you point it at).

`pdf2md-agent` renders each page of a PDF to an image, hands the image to a
chain of small vision-language agents (extractor → formatter → running
summarizer), and emits strict CommonMark Markdown that preserves the source
language verbatim — including CJK content.

It is designed to be robust on adversarial inputs:

- **Token-budgeted** — every per-page call is sized (and the page image
  downscaled) to stay under the model's context window.
- **Retry-aware** — transient API failures retry with exponential backoff
  + jitter; on retry exhaustion the page falls back to the PDF's native
  text layer (with a clearly-marked stub) instead of crashing the run.
- **Resumable** — per-page outputs and the running summary are cached, so
  re-running only fills in the pages that failed. Per-resource opt-outs
  (`--no-cache-{render,text,resized,extract,format,summary}`) let you
  invalidate a single resource without redoing the whole pipeline.

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [CLI reference](#cli-reference)
- [How it works](#how-it-works)
- [Caching and resumption](#caching-and-resumption)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

## Installation

`pdf2md-agent` requires Python **3.10+**.

```bash
# Recommended: use uv (https://github.com/astral-sh/uv)
git clone https://github.com/cherruq/pdf2md-agent.git
cd pdf2md-agent
uv sync
uv run pdf2md-agent --help
```

Or with `pip`:

```bash
git clone https://github.com/cherruq/pdf2md-agent.git
cd pdf2md-agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
pdf2md-agent --help
```

## Quick start

```bash
# 1. Configure credentials
cp .env.example .env
# then edit .env and set OPENAI_API_KEY (and optionally OPENAI_BASE_URL)

# 2. Convert a PDF
pdf2md-agent input.pdf -o output.md
```

Output is a single Markdown file. By default the per-page outputs are
stitched into one continuous document — paragraphs, list items, and table
rows split across a page break are merged, and the `\n\n---\n\n` page
separator is dropped. Pass `--stitch-mode off` to restore the legacy
separator. The original 1-based page numbers are preserved inside the
content where the formatter emits them.

```bash
# Convert only a subset of pages (1-based, ranges allowed)
pdf2md-agent input.pdf -o output.md --pages '1-5,8,11-13'

# Render at higher DPI for dense formulas / small fonts
pdf2md-agent input.pdf -o output.md --dpi 200

# Bypass every cache resource for a one-shot full re-run
pdf2md-agent input.pdf -o output.md --no-cache-all

# Re-format a previously-extracted page (strict CommonMark only)
pdf2md-agent input.pdf -o output.md --no-cache-extract
```

## Configuration

`pdf2md-agent` reads its config from environment variables (and `.env` if
present via `python-dotenv`). Every variable also has a CLI flag that
overrides the env value for the current invocation.

### Credentials and endpoint

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_BASE_URL` | `https://api.minimaxi.com/v1` | OpenAI-compatible endpoint. Override to point at any other vision API. |
| `OPENAI_API_KEY` | _(required)_ | API key for the endpoint above. |
| `PDF2MD_AGENT_MODEL` | `MiniMax-M3` | Vision model name to send in the request. |

### Token-budget / image-downscale

| Variable | Default | Notes |
|---|---|---|
| `PDF2MD_AGENT_CTX_LIMIT` | `2013` | Model context-window token limit the runner budgets against. |
| `PDF2MD_AGENT_TOKEN_BUDGET_SAFETY` | `0.85` | Fraction of `ctx_limit` the planner will spend per call. |
| `PDF2MD_AGENT_IMAGE_LONG_SIDE` | `1536` | Long-side pixel cap for inlined page JPEGs. Lower ⇒ smaller payloads, worse OCR. |
| `PDF2MD_AGENT_IMAGE_MIN_LONG_SIDE` | `768` | Lower bound for the binary search — never resize below this. |
| `PDF2MD_AGENT_IMAGE_JPEG_QUALITY` | `85` | JPEG quality (1–100) used by the in-memory downscaler. |
| `PDF2MD_AGENT_MAX_SUMMARY_CHARS` | `800` | Maximum running-summary size fed into the next extractor. |
| `PDF2MD_AGENT_REQUEST_TIMEOUT` | `60` | Per-attempt wall-clock timeout in seconds (0.1–600). |

### LLM retry / fallback

| Variable | Default | Notes |
|---|---|---|
| `PDF2MD_AGENT_MAX_RETRIES` | `0` | Total LLM call attempts per page (initial + retries). `0` or unset = unlimited; positive integer = bounded budget. |
| `PDF2MD_AGENT_RETRY_INITIAL_DELAY` | `1.0` | Initial retry delay in seconds (Fibonacci base unit). |
| `PDF2MD_AGENT_RETRY_MAX_DELAY` | `900.0` | Per-attempt delay cap (seconds). Fibonacci growth clamps at this ceiling. |
| `PDF2MD_AGENT_RETRY_JITTER` | `0.25` | Jitter ratio in `[0.0, 1.0]`. |
| `PDF2MD_AGENT_FALLBACK_TO_TEXT` | `true` | If `true`, fall back to the PDF's native text layer on retry exhaustion; if `false`, raise. |

Retry delays follow the Fibonacci sequence (1, 1, 2, 3, 5, 8, 13, …) scaled
by `PDF2MD_AGENT_RETRY_INITIAL_DELAY`, capped at `PDF2MD_AGENT_RETRY_MAX_DELAY`
(seconds) per attempt. With the default unlimited setting (`0`), transient
failures are retried forever; non-transient failures (4xx) always propagate
immediately.

### Pointing at a different provider

Any OpenAI-compatible vision endpoint works. Example: Anthropic-via-proxy.

```bash
OPENAI_BASE_URL=https://your-proxy.example/v1 \
OPENAI_API_KEY=sk-your-key \
PDF2MD_AGENT_MODEL=claude-3-5-sonnet \
pdf2md-agent paper.pdf -o paper.md
```

> **Important:** every page image is sent to the configured endpoint
> inline (as a base64 data URL). Image-bearing API requests larger than
> `PDF2MD_AGENT_CTX_LIMIT * PDF2MD_AGENT_TOKEN_BUDGET_SAFETY` tokens are
> automatically downscaled.

## CLI reference

```
pdf2md-agent PDF -o OUTPUT [options]
```

### Pipeline
| Flag | Type | Default | Notes |
|---|---|---|---|
| `pdf` | path | _(required)_ | Input PDF path. |
| `-o`, `--output` | path | _(required)_ | Output markdown path (written atomically). |
| `--dpi` | int | `144` | Render DPI. 72 (smallest), 150 (text + tables), 200 (small fonts / formulas), 300+ usually overkill for vision models. |
| `-p`, `--pages` | spec | all | `1-5,8,11-13` style subset; 1-based; preserves original page numbers in output. |

### Cache control
| Flag | Type | Default | Notes |
|---|---|---|---|
| `--no-intermediates` | flag | off | Skip writing cache files (uses a tempdir instead). |
| `--intermediates-dir` | path | `.pdf2md-agent-cache/<pdf_stem-or-hash>/` | Override cache directory. |
| `--no-cache-render` | flag | off | Re-render PNGs even when on disk. |
| `--no-cache-text` | flag | off | Re-extract the text layer even when on disk. |
| `--no-cache-resized` | flag | off | Re-resize the downscaled JPEG when needed. |
| `--no-cache-extract` | flag | off | Re-run the extractor; cached `extract.txt` is ignored. |
| `--no-cache-format` | flag | off | Re-run the formatter; cached `format.md` is ignored. |
| `--no-cache-summary` | flag | off | Ignore `summary.json`; start the running summary fresh. |
| `--no-cache-all` | flag | off | Equivalent to all six `--no-cache-*` flags above. |

### Feature disable
| Flag | Type | Default | Notes |
|---|---|---|---|
| `--no-summary` | flag | off | Process each page independently; deletes any pre-existing `summary.json`. |
| `--no-text-hint` | flag | off | Don't feed the PDF's native text layer to the extractor. |
| `--no-fallback-to-text` | flag | off | On retry exhaustion, raise instead of falling back. |
| `--stitch-mode` | `off` \| `heuristic` | `heuristic` | Heuristic (default) merges page splits; `off` keeps the legacy `\n\n---\n\n` separator. |

### Retry & tuning
| Flag | Type | Default | Notes |
|---|---|---|---|
| `--max-retries` | int ≥ 1 | `4` | Total LLM attempts per page. |
| `--retry-initial-delay` | float | `1.0` | Initial backoff delay (seconds). |
| `--retry-backoff` | float | `2.0` | Backoff multiplier. |
| `--retry-max-delay` | float | `30.0` | Per-attempt delay cap. |
| `--retry-jitter` | float in `[0, 1]` | `0.25` | Jitter ratio. |
| `--image-long-side` | int ≥ 64 | `1536` | Long-side cap (px) for inlined page JPEGs. |
| `--image-quality` | int 1-100 | `85` | JPEG quality. 75-95 is the practical sweet spot. |
| `--max-summary-chars` | int ≥ 100 | `800` | Running-summary character cap. |
| `--ctx-limit` | int ≥ 256 | `2013` | Model context-window token limit. |
| `--request-timeout` | float 0.1-600 | `60.0` | Per-attempt wall-clock timeout. |

### Diagnostic
| Flag | Type | Default | Notes |
|---|---|---|---|
| `--model` | string | `PDF2MD_AGENT_MODEL` | Model name recorded in `meta.json` for fingerprint validation. |
| `--persona-version` | string | `PERSONA_VERSION` | Persona fingerprint (16-char hex) recorded in `meta.json`. |
| `--version` / `-V` | flag | off | Print the package version and exit. |

## How it works

```
                ┌──────────────────┐
   PDF ───────► │  PyMuPDF render   │ ──► PNG per page (+ native text layer)
                └──────────────────┘
                          │
                          ▼
                ┌──────────────────────────────────────────────┐
                │ Per-page CrewAI crew (extract → format → …)  │
                │                                              │
                │  ① Extractor   (multimodal)                  │
                │     transcribes the page image into raw      │
                │     markdown, preserving CJK + layout.       │
                │                                              │
                │  ② Formatter   (text)                        │
                │     rewrites the extract into strict         │
                │     CommonMark, drops OCR noise.             │
                │                                              │
                │  ③ Summarizer  (text, optional)              │
                │     maintains a tight rolling summary fed    │
                │     into the next page's extractor.          │
                └──────────────────────────────────────────────┘
                          │
                          ▼  (StreamingStitcher: heuristic merge + drop `\n\n---\n\n`)
                   Markdown output
```

### Token-budget planner

Each extract call is sized by `pdf2md_agent.token_budget.plan_for_image`:

1. Estimate the token cost of the **persona** + the **per-page prompt
   variables** (running summary, text-hint, render scaffold).
2. Estimate the cost of the **image** at its current size (only the file's
   size is read — pixels are never decoded by the estimator).
3. If the sum exceeds `ctx_limit * safety`, find the **largest**
   `long_side` (via integer binary search) that keeps the call under
   budget, and pre-render a downscaled JPEG copy of the page.

Conservative heuristic: `~3.5 base64-chars per token`, `1/3 chars per CJK
token`, `1/4 chars per Latin token`.

### Multimodal patch

`AddImageTool._run` is monkey-patched once at import to (a) inline local
file paths as `data:image/jpeg;base64,…` URLs (re-encoded and downscaled
if needed) and (b) return the `VISION_IMAGE:…` sentinel string the
CrewAI step executor expects. OpenAI-compatible vision APIs reject bare
local paths with HTTP 400, so this patch is mandatory.

### Retry & fallback

`call_with_retry` wraps each `crew.kickoff()` in bounded exponential
backoff with jitter. On retry exhaustion (or a `ValidationError` from
malformed model output), the runner can emit a fenced text-layer stub
so the rest of the run keeps moving:

```markdown
*(vision model unavailable — falling back to PDF text layer; tables,
figures, and layout are NOT preserved)*

```[illegible]
<PDF native text for this page>
```
```

Disable with `--no-fallback-to-text` if you'd rather hard-fail.

## Caching and resumption

When `--intermediates` is on (the default) the runner writes:

```
.pdf2md-agent-cache/<stem-or-sha256[:16] of abs path>/
├── meta.json                  # pdf, dpi, with_summary, pages, model, persona_version
├── summary.json               # last running summary
└── pages/
    ├── page_0001.png          # source render
    ├── page_0001_text.txt     # PDF native text layer
    ├── page_0001_resized.jpg  # downscaled JPEG (if needed)
    ├── page_0001_extract.txt  # raw extractor output
    └── page_0001_format.md    # final CommonMark output
```

The cache key is the PDF's stem when it is short and free of path
separators; otherwise the runner hashes the absolute PDF path into a
16-character SHA-256 prefix. The key is deterministic — the same
absolute path always lands in the same cache directory.

### Cache key

The cache directory name is derived from the PDF's absolute path. If the
stem (filename without extension) is ≤ 60 characters and contains no
path separators or Windows-reserved characters, it is used as-is;
otherwise the first 16 characters of `sha256(absolute_path)` are used.

To find the cache directory for a given PDF:

```python
import hashlib, pathlib
pdf = pathlib.Path("/path/to/document.pdf").resolve()
stem = pdf.stem
key = stem if (len(stem) <= 60 and '/' not in stem) else hashlib.sha256(str(pdf).encode()).hexdigest()[:16]
cache_dir = f".pdf2md-agent-cache/{key}/"
```

`meta.json` carries a 6-field fingerprint (`pdf`, `dpi`, `with_summary`,
`pages`, `model`, `persona_version`). On every page the runner compares
the on-disk fingerprint with the current run's configuration; a drift
in any field forces a re-run. The persona version is the 16-char
SHA-256 of the active persona strings, so a text change in any
(extractor / formatter / summarizer) persona invalidates the cache.

Per-resource opt-outs:

- `--no-cache-render` — re-render the PNG even when the on-disk file
  matches the configured `--dpi`.
- `--no-cache-text` — re-extract the text layer.
- `--no-cache-resized` — re-resize the downscaled JPEG.
- `--no-cache-extract` — re-run the extractor. The cached `extract.txt`
  is ignored, but the formatter / summarizer still trust their own
  cache (unless those flags are set).
- `--no-cache-format` — re-run the formatter. When the cached
  `format.md` is missing the runner falls through to the full pipeline.
- `--no-cache-summary` — start the running summary fresh (no
  `summary.json` pre-seed).
- `--no-cache-all` — sets every per-resource flag.

`--no-summary` removes any pre-existing `summary.json` at the start of
the run so a previous cross-page run does not leak into a
single-page-style run.

Use `--no-intermediates` for ephemeral runs (writes go to a tempdir).

## Troubleshooting

### `OPENAI_API_KEY is not set`

Copy `.env.example` to `.env` and fill in your key. `python-dotenv`
auto-loads `.env` from the current working directory at import time.

### `400 context window exceeds limit` from the provider

The token-budget planner already downsizes page images to stay under
`PDF2MD_AGENT_CTX_LIMIT * PDF2MD_AGENT_TOKEN_BUDGET_SAFETY`. If you're still
hitting the limit:

- Lower `--image-long-side` (e.g. 1024) or `--image-quality` (e.g. 70).
- Lower `--max-summary-chars` — the running summary is the largest
  variable token cost per call.
- Raise `--ctx-limit` only if your endpoint actually has a larger window
  than the default `2013`.

### Output has gibberish or hallucinated content

- Try `--dpi 200` (or higher) — small fonts / dense formulas benefit.
- If a specific page failed, inspect
  `.pdf2md-agent-cache/<pdf-stem>/pages/page_NNNN_extract.txt` — that's
  exactly what the extractor returned before the formatter cleaned it up.

### Pages keep falling back to the text-layer stub

- Your endpoint may be returning a non-transient HTTP 4xx for the vision
  payload. Re-run with `--no-fallback-to-text` to surface the real error.
- Verify the model name in `PDF2MD_AGENT_MODEL` matches what the endpoint
  actually serves.
- The runner logs `run complete: N pages, M used fallback (text layer): [...]`
  on completion so you can see at a glance which pages degraded.

### `ImportError` from `crewai.tools.agent_tools.add_image_tool`

Older CrewAI versions don't expose that module path. Pin to
`crewai>=0.80,<2` (the project's required range).

## Development

```bash
# Clone and install with dev deps
git clone https://github.com/cherruq/pdf2md-agent.git
cd pdf2md-agent
uv sync

# Run tests
uv run pytest

# Tests do NOT hit the API; they monkeypatch the LLM and use local PDFs
# (none committed — the test corpus is synthesized in-memory).
```

Module layout:

```
src/pdf2md_agent/
├── __main__.py             # `python -m pdf2md-agent` entry
├── cli.py                  # argparse + atomic write
├── config.py               # env loading, defaults
├── cache.py                # per-PDF cache layout
├── pages.py                # --pages parser
├── pdf_renderer.py         # PyMuPDF wrapper
├── llm_retry.py            # bounded backoff + transient classifier
├── token_budget.py         # image/text estimators + planner
├── vision.py               # CrewAI LLM factory
├── post_stream.py          # cross-page stitcher (StreamingStitcher + heuristic)
└── crew/
    ├── agents.py           # extractor / formatter / summarizer personas
    ├── tasks.py            # build_extract_description + factory functions
    ├── multimodal_patch.py # AddImageTool monkey-patch
    └── runner.py           # per-page crew.kickoff loop
```

### Testing the matrix

The defaults below reproduce the project's CI surface:

```bash
pytest -ra tests/
```

| Test file | Covers |
|---|---|
| `test_cache.py` | `CacheLayout`, `MetaInfo`, fingerprint read/match, atomic writes |
| `test_no_cache.py` | `--no-cache-*` flag family, per-page priority, summary seed |
| `test_render_skip.py` | Render-side cache reuse (PNG / text / resized) |
| `test_pages.py` | `parse_page_spec`, `resolve_pages` |
| `test_pdf_renderer.py` | `render_pdf` shape, PNG + text-layer emit |
| `test_llm_retry.py` | `RetryConfig` validation + `is_transient` + backoff + timeout guard |
| `test_token_budget.py` | `estimate_text_tokens`, `estimate_image_tokens`, `plan_for_image` |
| `test_vision.py` | `make_vision_llm` endpoint wiring + timeout pass-through |
| `test_runner.py` | `run_pipeline` happy-path + extract-then-format + timeout guard |
| `test_post_stream.py` | `StreamingStitcher` heuristic (paragraph/list/table), finalize semantics, smart CJK/Latin join |
| `test_misc_coverage.py` | CLI argument groups, version, numeric validation, atomic write |
| `test_d8_coverage.py` | D8 batch coverage (CLI seams, runner helpers, multimodal patch) |

## License

MIT — see [`LICENSE`](./LICENSE).

## Acknowledgments

- [PyMuPDF](https://pymupdf.readthedocs.io/) for rendering and text extraction.
- [CrewAI](https://github.com/crewAIInc/crewAI) for the agent orchestration.
- The MiniMax-M3 endpoint at `api.minimaxi.com/v1` (or whatever
  `OPENAI_BASE_URL` you point at).

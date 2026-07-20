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
  re-running only fills in the pages that failed.
- **Layout-aware (`--reformat`)** — re-runs only the formatter + summarizer
  on cached extractor output, dropping running headers, footers, and page
  numbers while preserving every word verbatim.

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

Output is a single Markdown file. Pages are separated by `\n\n---\n\n` and
the original 1-based page numbers are preserved.

```bash
# Convert only a subset of pages (1-based, ranges allowed)
pdf2md-agent input.pdf -o output.md --pages '1-5,8,11-13'

# Render at higher DPI for dense formulas / small fonts
pdf2md-agent input.pdf -o output.md --dpi 200

# Re-run only the formatter on previously-extracted pages, dropping headers
# and footers (requires --intermediates / cache)
pdf2md-agent input.pdf -o output.md --reformat

# Resume a partially-failed run (re-uses per-page cached outputs)
pdf2md-agent input.pdf -o output.md --resume
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
| `PDF2MD_AGENT_IMAGE_JPEG_QUALITY` | `85` | JPEG quality (1–95) used by the in-memory downscaler. |
| `PDF2MD_AGENT_MAX_SUMMARY_CHARS` | `800` | Maximum running-summary size fed into the next extractor. |

### LLM retry / fallback

| Variable | Default | Notes |
|---|---|---|
| `PDF2MD_AGENT_MAX_RETRIES` | `4` | Total LLM call attempts per page (initial + retries). |
| `PDF2MD_AGENT_RETRY_INITIAL_DELAY` | `1.0` | Initial retry delay in seconds. |
| `PDF2MD_AGENT_RETRY_BACKOFF` | `2.0` | Exponential backoff multiplier between retries. |
| `PDF2MD_AGENT_RETRY_MAX_DELAY` | `30.0` | Per-attempt delay cap. |
| `PDF2MD_AGENT_RETRY_JITTER` | `0.25` | Jitter ratio in `[0.0, 1.0]`. |
| `PDF2MD_AGENT_FALLBACK_TO_TEXT` | `true` | If `true`, fall back to the PDF's native text layer on retry exhaustion; if `false`, raise. |

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

| Flag | Type | Default | Notes |
|---|---|---|---|
| `pdf` | path | _(required)_ | Input PDF path. |
| `-o`, `--output` | path | _(required)_ | Output markdown path (written atomically). |
| `--dpi` | int | `144` | Render DPI. 72 (smallest), 150 (text + tables), 200 (small fonts / formulas), 300+ usually overkill for vision models. |
| `-p`, `--pages` | spec | all | `1-5,8,11-13` style subset; 1-based; preserves original page numbers in output. |
| `--no-intermediates` | flag | off | Skip writing cache files (uses a tempdir instead). |
| `--intermediates-dir` | path | `.pdf2md-agent-cache/<pdf-stem>/` | Override cache directory. |
| `--resume` | flag | off | Reuse cached per-page outputs; only re-run missing pages. |
| `--reformat` | flag | off | Re-run formatter (+ summarizer) on cached extractor output; drops page headers/footers/numbers. Requires `--intermediates`. |
| `--no-summary` | flag | off | Disable the cross-page running summary (process each page independently). |
| `--no-text-hint` | flag | off | Disable feeding the PDF's native text layer to the extractor. |
| `--no-fallback-to-text` | flag | off | On retry exhaustion, raise instead of falling back. |
| `--max-retries` | int | `4` | Total LLM attempts per page. |
| `--retry-initial-delay` | float | `1.0` | Initial backoff delay. |
| `--retry-backoff` | float | `2.0` | Backoff multiplier. |
| `--retry-max-delay` | float | `30.0` | Per-attempt delay cap. |
| `--retry-jitter` | float | `0.25` | Jitter ratio. |
| `--image-long-side` | int | `1536` | Long-side cap (px) for inlined page JPEGs. |
| `--image-quality` | int | `85` | JPEG quality (1–95). |
| `--max-summary-chars` | int | `800` | Running-summary character cap. |
| `--ctx-limit` | int | `2013` | Model context-window token limit. |

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
                          ▼  (join with `\n\n---\n\n`)
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
.pdf2md-agent-cache/<pdf-stem>/
├── meta.json                  # pdf, dpi, with_summary, pages
├── summary.json               # last running summary
└── pages/
    ├── page_0001.png          # source render
    ├── page_0001_text.txt     # PDF native text layer
    ├── page_0001_resized.jpg  # downscaled JPEG (if needed)
    ├── page_0001_extract.txt  # raw extractor output
    └── page_0001_format.md    # final CommonMark output
```

- `--resume` skips any page whose `extract.txt` + `format.md` already
  exist (and re-reads `summary.json` to keep cross-page state consistent).
- `--reformat` skips the extractor for any page whose `extract.txt` is
  on disk and runs the layout-aware formatter instead. Falls back to the
  full pipeline for pages whose `extract.txt` is missing.

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
| `test_cache.py` | `CacheLayout` + meta/summary read/write |
| `test_pages.py` | `parse_page_spec`, `resolve_pages` |
| `test_pdf_renderer.py` | `render_pdf` shape, PNG + text-layer emit |
| `test_llm_retry.py` | `RetryConfig` validation + `is_transient` + backoff |
| `test_token_budget.py` | `estimate_text_tokens`, `estimate_image_tokens`, `plan_for_image` |
| `test_vision.py` | `make_vision_llm` endpoint wiring |
| `test_runner.py` | `run_pipeline` happy-path + extract-then-format |
| `test_reformat.py` | `--reformat` short-circuit + fallback paths |
| `test_misc_coverage.py` | misc seams |

## License

MIT — see [`LICENSE`](./LICENSE).

## Acknowledgments

- [PyMuPDF](https://pymupdf.readthedocs.io/) for rendering and text extraction.
- [CrewAI](https://github.com/crewAIInc/crewAI) for the agent orchestration.
- The MiniMax-M3 endpoint at `api.minimaxi.com/v1` (or whatever
  `OPENAI_BASE_URL` you point at).

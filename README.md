# pdf2md-agent

> Convert any PDF to clean, language-preserving Markdown — powered by a CrewAI
> vision pipeline against the MiniMax-M3 endpoint (or any OpenAI-compatible
> vision API you point it at).

`pdf2md-agent` renders each page of a PDF to an image, hands the image to a
vision-language extraction agent, and emits
strict CommonMark Markdown that preserves the source language verbatim —
including CJK content.

It is designed to be robust on adversarial inputs:

- **Token-budgeted** — every per-page call is sized (and the page image
  downscaled) to stay under the model's context window.
- **Retry-aware** — transient API failures retry with Fibonacci backoff
  (1, 1, 2, 3, 5, 8, 13, …), capped at 15 min + jitter; on retry exhaustion
  the page falls back to the PDF's native text layer (with a clearly-marked
  stub) instead of crashing the run.
- **Resumable** — per-page outputs are cached, so re-running only fills in
  the pages that failed. Per-resource opt-outs
  (`--no-cache-{render,text,resized,format}`) let you invalidate
  a single resource without redoing the whole pipeline.

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
separator. Page headers, footers, and page numbers are intentionally omitted
to ensure clean continuity of paragraphs and tables across pages.

```bash
# Convert only a subset of pages (1-based, ranges allowed)
pdf2md-agent input.pdf -o output.md --pages '1-5,8,11-13'

# Render at higher DPI for dense formulas / small fonts
pdf2md-agent input.pdf -o output.md --dpi 200

# Bypass every cache resource for a one-shot full re-run
pdf2md-agent input.pdf -o output.md --no-cache-all
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
| `PDF2MD_AGENT_CTX_LIMIT` | _(auto)_ | Model context-window token limit. Unset ⇒ hardcoded default for the active model (e.g. 524288 for MiniMax-M3), or 128000 fallback if unknown. |
| `PDF2MD_AGENT_REQUEST_TIMEOUT` | `60` | Per-attempt wall-clock timeout in seconds (0.1–600). |
| `PDF2MD_AGENT_MAX_RETRIES` | `0` | Total LLM call attempts per page (initial + retries). `0` or unset = unlimited; positive integer = bounded budget. |

Retry delays follow the Fibonacci sequence (1, 1, 2, 3, 5, 8, 13, …),
capped at 900 seconds per attempt. With the default unlimited setting (`0`),
transient failures are retried forever; non-transient failures (4xx) always
propagate immediately.

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
| `-p`, `--pages` | spec | all | `1-5,8,11-13` style subset; 1-based. |

### Cache control
| Flag | Type | Default | Notes |
|---|---|---|---|
| `--no-intermediates` | flag | off | Skip writing cache files (uses a tempdir instead). |
| `--intermediates-dir` | path | `.pdf2md-agent-cache/<pdf_stem-or-hash>/` | Override cache directory. |
| `--no-cache-render` | flag | off | Re-render PNGs even when on disk. |
| `--no-cache-text` | flag | off | Re-extract the text layer even when on disk. |
| `--no-cache-resized` | flag | off | Re-resize the downscaled JPEG when needed. |
| `--no-cache-format` | flag | off | Re-run the extraction to regenerate cached `format.md`. |
| `--no-cache-all` | flag | off | Equivalent to all four `--no-cache-*` flags above. |

### Feature disable
| Flag | Type | Default | Notes |
|---|---|---|---|
| `--no-text-hint` | flag | off | Don't feed the PDF's native text layer to the extractor. |
| `--no-fallback-to-text` | flag | off | On retry exhaustion, raise instead of falling back. |
| `--stitch-mode` | `off` \| `heuristic` | `heuristic` | Heuristic (default) merges page splits; `off` keeps the legacy `\n\n---\n\n` separator. |

### Retry & tuning
| Flag | Type | Default | Notes |
|---|---|---|---|
| `--max-retries` | int ≥ 1 | `0` | Total LLM attempts per page (`0` = unlimited). |
| `--image-long-side` | int ≥ 64 | `1536` | Long-side cap (px) for inlined page JPEGs. |
| `--image-quality` | int 1-100 | `85` | JPEG quality. 75-95 is the practical sweet spot. |
| `--ctx-limit` | int ≥ 256 | _(auto)_ | Model context-window token limit. |
| `--request-timeout` | float 0.1-600 | `60.0` | Per-attempt wall-clock timeout. |

### Diagnostic
| Flag | Type | Default | Notes |
|---|---|---|---|
| `--version` / `-V` | flag | off | Print the package version and exit. |

## How it works

```
                ┌──────────────────┐
   PDF ───────► │  PyMuPDF render   │ ──► PNG per page (+ native text layer)
                └──────────────────┘
                          │
                          ▼
                ┌──────────────────────────────────────────────┐
                │ Per-page CrewAI extraction crew              │
                │                                              │
                │  ① Extractor   (multimodal)                  │
                │     transcribes the page image into strict   │
                │     CommonMark, preserving CJK + layout.     │
                └──────────────────────────────────────────────┘
                          │
                          ▼  (StreamingStitcher: heuristic merge + drop `\n\n---\n\n`)
                   Markdown output
```

### Token-budget planner

Each extract call is sized by `pdf2md_agent.image_budget.plan_for_image`:

1. Estimate the token cost of the **persona** + the **per-page prompt
   variables** (text-hint, render scaffold).
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

`call_with_retry` wraps each `crew.kickoff()` in Fibonacci-capped backoff
(1, 1, 2, 3, 5, 8, 13, …), capped at 15 min (900s) per attempt, with jitter. By default
`--max-retries 0` means unlimited transient retries; pass a positive
integer to bound the budget. On retry exhaustion (or a `ValidationError` from
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
├── meta.json                  # pdf path fingerprint
└── pages/
    ├── page_0001.png          # source render
    ├── page_0001_text.txt     # PDF native text layer
    ├── page_0001_resized.jpg  # downscaled JPEG (if needed)
    └── page_0001_format.md    # final CommonMark output
```

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

`meta.json` carries a simple fingerprint (`pdf`). On every run the
runner compares the on-disk PDF realpath with the current run's file path.
A drift forces a re-run on that directory.

Per-resource opt-outs:

- `--no-cache-render` — re-render the PNG even when the on-disk file
  matches the configured `--dpi`.
- `--no-cache-text` — re-extract the text layer.
- `--no-cache-resized` — re-resize the downscaled JPEG.
- `--no-cache-format` — re-run extraction. When the cached
  `format.md` is missing or bypassed, the runner re-runs extraction.
- `--no-cache-all` — sets every per-resource flag.

Use `--no-intermediates` for ephemeral runs (writes go to a tempdir).

## Troubleshooting

### `OPENAI_API_KEY is not set`

Copy `.env.example` to `.env` and fill in your key. `python-dotenv`
auto-loads `.env` from the current working directory at import time.

### `400 context window exceeds limit` from the provider

The token-budget planner already downsizes page images to stay under
`PDF2MD_AGENT_CTX_LIMIT * PDF2MD_AGENT_TOKEN_BUDGET_SAFETY`. The default
limit is determined at startup from a hardcoded fallback per model. If you're still hitting the limit:

- Lower `--image-long-side` (e.g. 1024) or `--image-quality` (e.g. 70).
- Check the startup log for the resolved `ctx_limit` value; if your model needs a customized window limit, set `PDF2MD_AGENT_CTX_LIMIT` explicitly.

### Output has gibberish or hallucinated content

- Try `--dpi 200` (or higher) — small fonts / dense formulas benefit.
- If a specific page failed, inspect
  `.pdf2md-agent-cache/<pdf-stem>/pages/page_NNNN_format.md` — that's
  exactly what the model returned for that page.

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

For detailed architecture and testing guidelines, see `AGENTS.md` and `CONTRIBUTING.md`.

## License

MIT — see [`LICENSE`](./LICENSE).

## Acknowledgments

- [PyMuPDF](https://pymupdf.readthedocs.io/) for rendering and text extraction.
- [CrewAI](https://github.com/crewAIInc/crewAI) for the agent orchestration.
- The MiniMax-M3 endpoint at `api.minimaxi.com/v1` (or whatever
  `OPENAI_BASE_URL` you point at).

# `pdf2md_agent.crew` — Per-Page Pipeline Orchestrator

Subpackage reduced to a single module after dropping the CrewAI state
machine. The LLM calls themselves now live in
``pdf2md_agent.raw_pipeline`` (top-level); this subpackage retains the
per-page loop, cache semantics, and text-layer fallback helpers.

## FILE RESPONSIBILITIES

| File | Purpose | Entry point |
|---|---|---|
| `__init__.py` | empty marker | |
| `runner.py` | per-page loop: extract → format → (summarize) with text-layer fallback | `run_pipeline`, `_run_format_summarize_only` |

## KEY SYMBOLS (with refs)

| Symbol | File:Line | Role |
|---|---|---|
| `run_pipeline` | runner.py:177 | `CacheNoCacheFlags`-driven per-page loop; per-call `call_with_retry` |
| `_run_format_summarize_only` | runner.py:510 | `--no-cache-extract` path (skip extractor, re-run format + summarizer) |
| `_record_text_layer_fallback` | runner.py:133 | writes extract.txt sentinel + format.md + PageResult when LLM fails |
| `_text_layer_fallback` | runner.py:89 | pure markdown stub from PDF text layer |
| `_resize_page_png` | runner.py:111 | Pillow LANCZOS thumbnail used when token-budget forces a smaller image |
| `_FALLBACK_SENTINEL` | runner.py:162 | prefix written to extract.txt on text-layer fallback (rejected by `has_cached_extract`) |
| `PageResult` | runner.py:168 | frozen+slotted `(page_number, markdown, summary)` dataclass |
| `_make_client` | (re-export from raw_pipeline) | `runner._make_client = raw_pipeline._make_client` so tests patch at `pdf2md_agent.crew.runner._make_client` |

## CONVENTIONS (specific to this subpackage)

- **Per-call retries**: each of `extract` / `format` / `summarize` is wrapped in its own `call_with_retry(label="page N extract"|"format"|"summarize")`. Failure of one call falls back to the text-layer markdown; failure of the others is logged and the previous running summary is preserved.
- **`call_with_retry` expects a zero-arg callable.** The runner wraps `_do_extract` / `call_formatter` / `call_summarizer` in `lambda:` closures that capture the per-page kwargs (see `runner.py:345-358`, `384-394`, `422-435`).
- **The `_make_client` re-export is intentional** — tests patch `pdf2md_agent.crew.runner._make_client` rather than `pdf2md_agent.raw_pipeline._make_client` so the runner is the seam under test, not the raw pipeline module.

## ANTI-PATTERNS (this subpackage)

- **Never reintroduce CrewAI state.** No `from crewai import ...`. No `Agent`/`Task`/`Crew` factories. Each LLM call must remain a pure function whose inputs == its observable state. The `VISION_IMAGE:` sentinel mechanism is gone — the image is inlined directly as an `image_url` data URL in the user message.
- **Never replace `call_with_retry(fn, ...)` with a kwargs-forwarding call.** `call_with_retry` (in `pdf2md_agent.llm_retry`) takes a zero-arg callable. Use a closure that captures the kwargs.
- **Never swallow the fallback decision.** When `fallback_to_text=False` is passed and a transient failure exhausts retries, the exception must propagate. The runner's `if not fallback_to_text or not is_transient(exc): raise` guards are load-bearing.
- **Runner's `except BaseException` in the per-call wrappers** is intentional for retry-exhaustion cleanup. Do not narrow to `Exception`.

## NOTES

- The text-layer fallback writes a non-empty sentinel line into `extract.txt` so `has_cached_extract` rejects it (otherwise `--no-cache-extract` would feed the marker text into the formatter).
- Page render names use the original 1-based numbers from `pdf_renderer.py` — stable across `--pages` selections.

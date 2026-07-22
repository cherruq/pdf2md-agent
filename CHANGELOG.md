# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `--no-cache-all` plus the per-resource `--no-cache-{render,text,resized,extract,format,summary}` flag family. Default semantics flipped: cache is trusted unless explicitly opted out. The single `CacheNoCacheFlags` dataclass (`src/pdf2md_agent/cache.py:CacheNoCacheFlags`) is the typed contract between CLI and runner.
- `--request-timeout` CLI flag + `REQUEST_TIMEOUT_SECONDS` config (default 60s). Both the OpenAI SDK call and the runner's per-attempt guard share the value; a wall-clock overrun reclassifies the attempt as transient so the retry loop re-issues.
- `--version` / `-V` flag that prints the package version (`pdf2md_agent.__about__.__version__`) and exits 0.
- Meta fingerprint validation: `meta.json` now records `model` and `persona_version` (16-char SHA-256 of the active persona strings). The runner refuses to re-use cached outputs when the fingerprint drifts.
- Render-side cache reuse: `pdf2md_agent.render_skip` exposes `maybe_skip_render` / `maybe_skip_text` / `maybe_skip_resized`; the CLI consults them before calling `render_pdf`, so a follow-up run with the same `--dpi` skips the PyMuPDF re-render.
- H1 sentinel: when a page falls back to the text layer, `extract.txt` is now written with a non-empty sentinel line (`(vision model unavailable for page N; ...)`) so `has_cached_extract` and downstream consumers can detect "this page has no real extractor output" instead of silently treating the empty file as success.
- L7 fix: `--no-summary` now deletes `summary.json` at the start of the run so the running summary does not survive across `--no-summary` invocations.
- Cache root: `_cache_key_for_pdf` derives a deterministic, 16-char SHA-256 digest of the absolute PDF path for stems that are too long, contain path separators, or collide with Windows reserved names.
- `--help` argument groups: "Pipeline", "Cache control", "Feature disable", "Retry & tuning", "Diagnostic" — unified naming across the four flag families (`--no-cache-*` for cache resources, `--no-*` for optional features).
- Run-completion log line: when any pages used the text-layer fallback, the runner logs `run complete: N pages, M used fallback (text layer): [pages...]`.
- Numeric CLI validation: `--max-retries`, `--image-quality`, `--image-long-side`, `--max-summary-chars`, `--ctx-limit`, `--request-timeout` now reject out-of-range or non-numeric values at the parser.

### Changed
- Pipeline description in `--help`: explicitly lists the five stages (render → extract → format → summarize → stitch) and the cache fingerprint fields.
- Internal `run_pipeline` signature: `resume: bool` and `reformat: bool = False` removed; the single `no_cache: CacheNoCacheFlags` parameter drives the per-page priority chain (format short-circuit → extract short-circuit → full pipeline).

### Breaking
- `--resume` removed. Use `--no-cache-all` (or selectively `--no-cache-format`) to force a re-run.
- `--reformat` removed. The layout-aware formatter persona was deleted; the only formatter persona is the strict CommonMark one.
- Pre-`0.3.0` `meta.json` (4 fields) will fail fingerprint validation under any cache reuse. Wipe `.pdf2md-agent-cache/<stem>/` (or use `--no-cache-all` once) after upgrading.
- `FORMATTER_PERSONA_REFORMAT` and the `reformat` parameter on `make_formatter` / `make_format_task` are gone. Cache files written under the old `--reformat` mode are no longer trusted by the new extract-short-circuit (the new path re-runs the strict formatter on whatever extract.txt is on disk).

## [0.2.0] — 2026-07-17

### Added
- `--pages` (`-p`) CLI flag to convert a subset of pages (e.g. `--pages '1-5,8,11-13'`); 1-based, supports ranges and out-of-range errors surface before any tempdir or render work.
- `--reformat` mode that re-runs the formatter (+ summarizer) on cached extractor output, dropping running headers, footers, and page numbers while preserving every other word verbatim. Requires `--intermediates`; falls through to the full pipeline for pages whose `extract.txt` is missing.
- Per-page token-budget planner (`pdf2md_agent.token_budget`) — every extract call is sized against the configured context window, with image downscaling via integer binary search when needed.
- Bounded exponential-backoff retry for transient vision-API failures (`APITimeoutError`, `APIConnectionError`, `InternalServerError`, `RateLimitError`, plus 5xx `APIStatusError`). Permanent 4xx errors propagate immediately.
- Text-layer fallback on retry exhaustion (or `ValidationError` from malformed model output): the page is rendered as a fenced stub from the PDF's native text instead of crashing the run. Disable with `--no-fallback-to-text`.
- Layout-aware formatter persona (`FORMATTER_PERSONA_REFORMAT`) in addition to the strict CommonMark formatter.
- `CacheLayout.has_cached_extract` helper, used by `--reformat` to short-circuit the extractor.
- `meta.json` records the converted page list so resumable runs know exactly which pages are covered.

### Changed
- Removed redundant `convert` subcommand — the package is now a single-command CLI (`pdf2md-agent <pdf> -o <output>`).
- Hardened retry / budget / cache seams against corner cases (negative args, oversized summary characters, empty pages).

## [0.1.0] — 2025

### Added
- Initial implementation: PDF → per-page PNG (PyMuPDF), CrewAI crew (extractor → formatter → optional summarizer) against the MiniMax-M3 endpoint, OpenAI-compatible `provider="openai"` to skip LiteLLM.
- `AddImageTool` monkey-patch: local-file inlining as `data:image/jpeg;base64,…` and the `VISION_IMAGE:…` sentinel the CrewAI step executor recognises.
- Per-PDF cache under `.pdf2md-agent-cache/<stem>/` with per-page source PNG, native text layer, extractor text, formatter markdown, and running summary.
- Atomic output write via sibling temp file + `os.replace`.

[Unreleased]: https://github.com/cherruq/pdf2md-agent/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/cherruq/pdf2md-agent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cherruq/pdf2md-agent/releases/tag/v0.1.0

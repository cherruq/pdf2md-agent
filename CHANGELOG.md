# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Cross-page stitching post-processor (`pdf2md_agent.post_stream`) that merges paragraphs, list items, and table rows split across page boundaries, and drops the `\n\n---\n\n` page separator by default. Opt out with `--stitch-mode off` to restore the legacy separator. Default mode is heuristic (no extra LLM calls).

### Changed
- Generalised project description for public distribution (LLM-agnostic: defaults to MiniMax-M3 but any OpenAI-compatible vision endpoint works via `OPENAI_BASE_URL`).
- Internal design documents removed from version control.

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

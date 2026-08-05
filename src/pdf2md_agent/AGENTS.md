# `pdf2md_agent` — Infrastructure Layer

Single flat package (no sub-packages except `crew/`). Concerns split by file, not by directory. All non-CrewAI plumbing lives here.

## MODULE RESPONSIBILITIES

| File | Purpose | Notes |
|---|---|---|
| `__init__.py` | exposes `__version__` (currently `0.2.1`) | must match `pyproject.toml` (update together) |
| `__main__.py` | `from .cli import main` | enables `python -m pdf2md_agent` |
| `config.py` | `ConversionConfig` struct + `dotenv.load_dotenv()` at import + `Final` env bindings | ONE import-time side-effect; never re-import indirectly |
| `tuning.py` | internal system tuning constants | isolates static parameters (budget safety, reflection thresholds, defaults) from runtime config |
| `cli.py` | CLI entry & slim parameter gateway | `print(..., file=sys.stderr)` for CLI errors — NOT logger (preserve) |
| `pipeline.py` | 3-step unified conversion orchestrator (`run_unified_conversion`) | consumes immutable `ConversionConfig`; decoupled from argparse |
| `cli_parser.py` | argparse definition + post-parse resolvers | `_resolve_layout`, `_resolve_no_cache_flags`, `build_retry_config`, `_safe_intermediates_dir`, etc. |
| `filesystem_safety.py` | `safe_cache_stem`, `cache_key_for_pdf` | Windows-reserved-name + path-traversal safety |
| `cache.py` | `CacheLayout`, `PageArtifacts`, `FALLBACK_SENTINEL`, atomic JSON read/write | cached on `.pdf2md-agent-cache/<stem>/` |
| `pages.py` | `parse_page_spec` (argparse `type=`) + `resolve_pages` | 1-based `'1-5,8,11-13'` grammar |
| `pdf_renderer.py` | `render_pdf` → list[`RenderedPage`] + native text layer | PyMuPDF; emits PNG + `.text.txt` per page |
| `vision.py` | `make_vision_llm` factory | `provider="openai"` to bypass LiteLLM |
| `llm_retry.py` | `RetryConfig` + `is_transient` + `call_with_retry` | Fibonacci backoff (1,1,2,3,5,8,13…) + jitter; non-transient (4xx) propagates |
| `token_estimator.py` | `estimate_text_tokens` + `estimate_image_tokens` | NO `tiktoken`; CJK÷3, ASCII÷4, image b64÷3.5 |
| `image_budget.py` | `plan_for_image` + `BudgetDecision` (binary search) | consumes the estimator's token math; Pillow-fallback for corrupt pages |
| `post_stream.py` | `StreamingStitcher` cross-page joiner | `StitchMode.{OFF, HEURISTIC}`; HEURISTIC default |
| `post_stream_decision.py` | block split + continuation decision + smart-join + CJK detection | pure helpers; no LLM, no I/O |
| `post_stream_table.py` | table-row continuation + header dedup | pure helpers; no LLM, no I/O |

## DATA FLOW (CLI → file)

```
cli.main
  └─ cmd_convert(args)
       ├─ _validate_pdf_header(args.pdf)        ← cli.py (fail-fast on missing/wrong-magic)
       ├─ _resolve_requested_pages(args.pdf, args.pages)  ← cli.py (resolve + bounds check)
       ├─ _resolve_layout(...)                  ← cli_parser.py (CacheLayout + render-target)
       ├─ build_retry_config(args)              ← cli_parser.py (CLI override + env fallback)
       ├─ config = ConversionConfig(...)         ← config.py (assemble immutable job configuration)
       └─ pipeline.run_unified_conversion(config)
            ├─ read_meta + check_meta_matches   ← cache.py (drift refusal)
            ├─ write_meta                       ← cache.py (fingerprint)
            ├─ Step 1: _render_pages (pdf_renderer.py: static render & real-time text drift invalidation)
            ├─ make_vision_llm(...)             (timeout=REQUEST_TIMEOUT_SECONDS)
            ├─ Step 2: crew.runner.run_pipeline (parallel per-page loop; format skip if valid cache)
            └─ Step 3: post_stream.stitch_pages (global post-processing: strip headers/footers/page-numbers & stitch)
```

## CONVENTIONS (specific to this layer)

- **Atomic writes only.** `cli.py:atomic_write_text` (re-export from `cache.py`) writes to a sibling temp file then `os.replace`. Never `Path(out).write_text()` directly.
- **`@dataclass(slots=True, frozen=True)`** for every value type. No `__init__` overrides unless mutable buffers are needed.
- CLI args override env vars (parsed in `cmd_convert` after `config.py` has already populated defaults).
- Page render names use **original 1-based** numbers — stable across `--pages` selections.
- `# type: ignore` / `# noqa: XXXX` may only be used **with an explanatory inline comment**. Stripping the comment is a violation even if the suppression itself stays.
- `secrets.SystemRandom()` (not `random`) for retry jitter — `llm_retry.py`. Backoff sequences must not sync across parallel clients.

## ANTI-PATTERNS (infrastructure layer)

- **Do not** bypass `atomic_write_text` with a direct `Path.write_text` — partial writes on crash corrupt the output.
- **Do not** read PDF pixel data outside `pdf_renderer.py` / `image_budget.py`. Other modules accept the already-decoded `RenderedPage` or its `path`.
- **Do not** strip the load-bearing suppressions:
  - `cache.py` `atomic_write_text` bare `except Exception:` — cleans up the sibling-tempfile before re-raising.
  - `llm_retry.py` bare `except Exception` / `except BaseException` with `# noqa: BLE001` — `is_transient(exc)` predicate discriminates and the daemon-thread joiner re-raises on purpose.
  - `image_budget.py` `_open_for_size` bare `except Exception` — Pillow-fallback for corrupt images.
- **Do not** narrow `except BaseException` to `Exception` in `crew/extraction.py` — retry-exhaustion cleanup needs it.

## NOTES

- `config.py` runs `load_dotenv()` at import. Any module that imports it transitively will read `.env` once — keep imports of `config.py` at module top, not inside functions, to preserve single-load semantics.
- `post_stream.py` (and its `post_stream_decision` / `post_stream_table` helpers) are **pure** — no LLM, no I/O beyond string splitting. Reusable for any CommonMark-ish input.
- `_safe_exc_summary` in `llm_retry.py` redacts provider payloads before they hit logs — preserve the `APIStatusError` branch so user content never lands in log files.
- The token-budget functions live separated into `token_estimator.py` (text + image estimators) and `image_budget.py` (binary-search planner).
- The post-stream module split mirrors the same idea: `post_stream.py` is the public API, with the decision engine and table helpers in dedicated private modules.
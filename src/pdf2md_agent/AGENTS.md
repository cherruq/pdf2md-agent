# `pdf2md_agent` — Infrastructure Layer

Single flat package (no sub-packages except `crew/`). Concerns split by file, not by directory. All non-CrewAI plumbing lives here.

## MODULE RESPONSIBILITIES

| File | Purpose | Notes |
|---|---|---|
| `__init__.py` | exposes `__version__` (currently `0.2.0`) | must match `pyproject.toml` (update together) |
| `__about__.py` | canonical version string | mirrored from `pyproject.toml` |
| `__main__.py` | `from .cli import main` | enables `python -m pdf2md_agent` |
| `config.py` | `dotenv.load_dotenv()` at import + `Final` env bindings | ONE import-time side-effect; never re-import indirectly |
| `cli.py` | CLI entry + per-run orchestration | `print(..., file=sys.stderr)` for CLI errors — NOT logger (preserve) |
| `cli_parser.py` | argparse definition + post-parse resolvers | `_resolve_layout`, `_resolve_no_cache_flags`, `build_retry_config`, `_safe_intermediates_dir`, etc. |
| `filesystem_safety.py` | `safe_cache_stem`, `cache_key_for_pdf` | Windows-reserved-name + path-traversal safety |
| `cache.py` | `CacheLayout`, `PageArtifacts`, `FALLBACK_SENTINEL`, atomic JSON read/write | cached on `.pdf2md-agent-cache/<stem>/` |
| `pages.py` | `parse_page_spec` (argparse `type=`) + `resolve_pages` | 1-based `'1-5,8,11-13'` grammar |
| `pdf_renderer.py` | `render_pdf` → list[`PageImage`] + native text layer | PyMuPDF; emits PNG + `.text.txt` per page |
| `render_skip.py` | `maybe_skip_render` + `maybe_skip_resized` | trust-cache gates for PNG / downscaled JPEG reuse |
| `vision.py` | `make_vision_llm` factory | `provider="openai"` to bypass LiteLLM |
| `llm_retry.py` | `RetryConfig` + `is_transient` + `call_with_retry` | Fibonacci backoff (1,1,2,3,5,8,13…) + jitter; non-transient (4xx) propagates |
| `token_budget.py` | back-compat shim → `token_estimator` + `image_budget` | re-exports `BudgetDecision`, `plan_for_image`, `estimate_*` |
| `token_estimator.py` | `estimate_text_tokens` + `estimate_image_tokens` | NO `tiktoken`; CJK÷3, ASCII÷4, image b64÷3.5 |
| `image_budget.py` | `plan_for_image` + `BudgetDecision` (binary search) | consumes the estimator's token math; Pillow-fallback for corrupt pages |
| `ctx_probe.py` | probe `/v1/models` for `ctx_limit` | called from `cli._run_pipeline`; falls through to hardcoded per-model defaults |
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
       └─ _run_pipeline
            ├─ read_meta + check_meta_matches  ← cache.py (drift refusal)
            ├─ write_meta                      ← cache.py (fingerprint)
            ├─ _render_pages  (trust-cache fast path → render_pdf only on cache miss)
            ├─ make_vision_llm(...)            (timeout=REQUEST_TIMEOUT_SECONDS)
            ├─ resolve_ctx_limit()              (probe → hardcoded fallback)
            └─ crew.runner.run_pipeline(...)  ← per page (format → extract → full)
                 └─ post_stream.stitch_pages(...)
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
- **Do not** read PDF pixel data outside `pdf_renderer.py` / `image_budget.py`. Other modules accept the already-decoded `PageImage` or its `path`.
- **Do not** strip the load-bearing suppressions:
  - `cache.py` `atomic_write_text` bare `except Exception:` — cleans up the sibling-tempfile before re-raising.
  - `llm_retry.py` bare `except Exception` / `except BaseException` with `# noqa: BLE001` — `is_transient(exc)` predicate discriminates and the daemon-thread joiner re-raises on purpose.
  - `image_budget.py` `_open_for_size` bare `except Exception` — Pillow-fallback for corrupt images.
  - `ctx_probe.py:75,80` `# noqa: S310` — scheme is pre-validated at line 72, so the bandit warning is a false positive.
- **Do not** narrow `except BaseException` to `Exception` in `crew/extraction.py` — retry-exhaustion cleanup needs it.

## NOTES

- `config.py` runs `load_dotenv()` at import. Any module that imports it transitively will read `.env` once — keep imports of `config.py` at module top, not inside functions, to preserve single-load semantics.
- `post_stream.py` (and its `post_stream_decision` / `post_stream_table` helpers) are **pure** — no LLM, no I/O beyond string splitting. Reusable for any CommonMark-ish input.
- `render_skip.py` is the trust-cache gate used by `_render_pages`; flipping either predicate changes cache reuse behavior globally — coordinate with `cache.py` semantics.
- `_safe_exc_summary` in `llm_retry.py` redacts provider payloads before they hit logs — preserve the `APIStatusError` branch so user content never lands in log files.
- The token-budget module split is purely organizational: `token_estimator.py` (text + image estimators) and `image_budget.py` (binary-search planner) live separately; `token_budget.py` is the back-compat shim.
- The post-stream module split mirrors the same idea: `post_stream.py` is the public API, with the decision engine and table helpers in dedicated private modules.
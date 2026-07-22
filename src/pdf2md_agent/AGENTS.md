# `pdf2md_agent` — Infrastructure Layer

Single flat package (no sub-packages except `crew/`). Concerns split by file, not by directory. All non-CrewAI plumbing lives here.

## MODULE RESPONSIBILITIES

| File | Purpose | Notes |
|---|---|---|
| `__init__.py` | exposes `__version__` (currently `0.2.0`) | |
| `__main__.py` | `from .cli import main` | enables `python -m pdf2md_agent` |
| `config.py` | `dotenv.load_dotenv()` at import + `Final` env bindings | ONE import-time side-effect; never re-import indirectly |
| `cli.py` | argparse + atomic output write | `print(..., file=sys.stderr)` for CLI errors — NOT logger (preserve) |
| `cache.py` | `CacheLayout`, `PageArtifacts`, atomic JSON read/write | cached on `.pdf2md-agent-cache/<stem>/` |
| `pages.py` | `parse_page_spec` (argparse `type=`) + `resolve_pages` | 1-based `'1-5,8,11-13'` grammar |
| `pdf_renderer.py` | `render_pdf` → list[`PageImage`] + native text layer | PyMuPDF; emits PNG + `.text.txt` per page |
| `vision.py` | `make_vision_llm` factory | `provider="openai"` to bypass LiteLLM |
| `llm_retry.py` | `RetryConfig` + `is_transient` + `call_with_retry` | exp backoff + jitter; non-transient (4xx) propagates |
| `token_budget.py` | heuristic estimator + `plan_for_image` binary search | NO `tiktoken`; CJK÷3, ASCII÷4, image b64÷3.5 |
| `post_stream.py` | `StreamingStitcher` cross-page joiner | `StitchMode.{OFF, HEURISTIC}`; HEURISTIC default |

## DATA FLOW (CLI → file)

```
cli.main
  └─ cmd_convert(args)
       ├─ CacheLayout  (--no-intermediates → tempdir; else .pdf2md-agent-cache/<key>/)
       ├─ _atomic_write_text(out, md)         ← cli.py:252  (sibling-tempfile + os.replace)
       └─ _run_pipeline
            ├─ _render_pages  (trust-cache fast path → render_pdf only on cache miss)
            ├─ make_vision_llm(...)            (timeout=REQUEST_TIMEOUT_SECONDS)
            └─ crew.runner.run_pipeline(...)  ← per page (format → extract → full)
                 └─ post_stream.stitch_pages(...)
```

## CONVENTIONS (specific to this layer)

- **Atomic writes only.** `cli.py:_atomic_write_text` writes to a sibling temp file then `os.replace`. Never `Path(out).write_text()` directly.
- **`@dataclass(slots=True, frozen=True)`** for every value type. No `__init__` overrides unless mutable buffers are needed.
- CLI args override env vars (parsed in `cmd_convert` after `config.py` has already populated defaults).
- Page render names use **original 1-based** numbers — stable across `--pages` selections (so `--resume` survives partial re-runs).

## ANTI-PATTERNS (infrastructure layer)

- **Do not** bypass `_atomic_write_text` with a direct `Path.write_text` — partial writes on crash corrupt the output.
- **Do not** read PDF pixel data outside `pdf_renderer.py` / `token_budget.py`. Other modules accept the already-decoded `PageImage` or its `path`.
- The bare `except Exception:` blocks in `cli.py:271,276` and `llm_retry.py:112` and `token_budget.py:181` have explanatory comments — they are load-bearing (atomic-write cleanup, transient-classifier BLE001, Pillow fallback). Do not "clean them up".

## NOTES

- `config.py` runs `load_dotenv()` at import. Any module that imports it transitively will read `.env` once — keep imports of `config.py` at module top, not inside functions, to preserve single-load semantics.
- `post_stream.py` is **pure** — no LLM, no I/O beyond string splitting. Reusable for any CommonMark-ish input.

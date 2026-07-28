# `pdf2md_agent` — Infrastructure Layer

Single flat package (no sub-packages except `crew/`). Concerns split by file, not by directory. All non-CrewAI plumbing lives here.

## MODULE RESPONSIBILITIES

| File | Purpose | Notes |
|---|---|---|
| `__init__.py` | exposes `__version__` (currently `0.2.0`) | must match `pyproject.toml` (update together) |
| `__about__.py` | canonical version string | mirrored from `pyproject.toml` |
| `__main__.py` | `from .cli import main` | enables `python -m pdf2md_agent` |
| `config.py` | `dotenv.load_dotenv()` at import + `Final` env bindings | ONE import-time side-effect; never re-import indirectly |
| `cli.py` | argparse + atomic output write | `print(..., file=sys.stderr)` for CLI errors — NOT logger (preserve) |
| `cache.py` | `CacheLayout`, `PageArtifacts`, atomic JSON read/write | cached on `.pdf2md-agent-cache/<stem>/` |
| `pages.py` | `parse_page_spec` (argparse `type=`) + `resolve_pages` | 1-based `'1-5,8,11-13'` grammar |
| `pdf_renderer.py` | `render_pdf` → list[`PageImage`] + native text layer | PyMuPDF; emits PNG + `.text.txt` per page |
| `render_skip.py` | `maybe_skip_render` + `maybe_skip_resized` | trust-cache gates for PNG / downscaled JPEG reuse |
| `vision.py` | `make_vision_llm` factory | `provider="openai"` to bypass LiteLLM |
| `llm_retry.py` | `RetryConfig` + `is_transient` + `call_with_retry` | Fibonacci backoff (1,1,2,3,5,8,13…) + jitter; non-transient (4xx) propagates |
| `token_budget.py` | heuristic estimator + `plan_for_image` binary search | NO `tiktoken`; CJK÷3, ASCII÷4, image b64÷3.5 |
| `ctx_probe.py` | probe `/v1/models` for `ctx_limit` | called from `cli._run_pipeline`; falls through to hardcoded per-model defaults |
| `post_stream.py` | `StreamingStitcher` cross-page joiner | `StitchMode.{OFF, HEURISTIC}`; HEURISTIC default |

## DATA FLOW (CLI → file)

```
cli.main
  └─ cmd_convert(args)
       ├─ CacheLayout  (--no-intermediates → tempdir; else .pdf2md-agent-cache/<key>/)
       ├─ _atomic_write_text(out, md)         ← cli.py (sibling-tempfile + os.replace)
       └─ _run_pipeline
            ├─ _render_pages  (trust-cache fast path → render_pdf only on cache miss)
            ├─ make_vision_llm(...)            (timeout=REQUEST_TIMEOUT_SECONDS)
            ├─ resolve_ctx_limit()              (probe → hardcoded fallback)
            └─ crew.runner.run_pipeline(...)  ← per page (format → extract → full)
                 └─ post_stream.stitch_pages(...)
```

## CONVENTIONS (specific to this layer)

- **Atomic writes only.** `cli.py:_atomic_write_text` writes to a sibling temp file then `os.replace`. Never `Path(out).write_text()` directly.
- **`@dataclass(slots=True, frozen=True)`** for every value type. No `__init__` overrides unless mutable buffers are needed.
- CLI args override env vars (parsed in `cmd_convert` after `config.py` has already populated defaults).
- Page render names use **original 1-based** numbers — stable across `--pages` selections.
- `# type: ignore` / `# noqa: XXXX` may only be used **with an explanatory inline comment**. Stripping the comment is a violation even if the suppression itself stays.
- `secrets.SystemRandom()` (not `random`) for retry jitter — `llm_retry.py:30`. Backoff sequences must not sync across parallel clients.

## ANTI-PATTERNS (infrastructure layer)

- **Do not** bypass `_atomic_write_text` with a direct `Path.write_text` — partial writes on crash corrupt the output.
- **Do not** read PDF pixel data outside `pdf_renderer.py` / `token_budget.py`. Other modules accept the already-decoded `PageImage` or its `path`.
- **Do not** strip the three load-bearing suppressions:
  - `cache.py:69,74` — bare `except Exception:` cleans up the sibling-tempfile before re-raising (atomic-write cleanup).
  - `llm_retry.py:209` — bare `except Exception` with `# noqa: BLE001`; `is_transient(exc)` predicate below discriminates.
  - `llm_retry.py:273` — `except BaseException as exc` with `# noqa: BLE001`; the daemon-thread joiner re-raises on purpose.
  - `token_budget.py:197,207` — bare `except Exception` for the Pillow-fallback path when the page image is corrupt.
  - `ctx_probe.py:75,80` — `# noqa: S310` on `Request()` and `urlopen()`; scheme is pre-validated at line 72, so the bandit warning is a false positive.
- **Do not** narrow `except BaseException` to `Exception` — `runner.py:477` and `runner.py:640` (in `crew/`) need it for retry-exhaustion cleanup.

## NOTES

- `config.py` runs `load_dotenv()` at import. Any module that imports it transitively will read `.env` once — keep imports of `config.py` at module top, not inside functions, to preserve single-load semantics.
- `post_stream.py` is **pure** — no LLM, no I/O beyond string splitting. Reusable for any CommonMark-ish input.
- `render_skip.py` is the trust-cache gate used by `_render_pages`; flipping either predicate changes cache reuse behavior globally — coordinate with `cache.py` semantics.
- `_safe_exc_summary` in `llm_retry.py:41-53` redacts provider payloads before they hit logs — preserve the `APIStatusError` branch so user content never lands in log files.

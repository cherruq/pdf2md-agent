# PROJECT KNOWLEDGE BASE

**Generated:** 2026-07-21
**Commit:** 23f448d (main)
**Stack:** Python ≥3.10 · PyMuPDF · OpenAI SDK 1.x · Pillow · hatchling · uv

## OVERVIEW

`pdf2md-agent` renders each PDF page to an image and feeds it through explicit per-page LLM calls (`extractor → formatter → summarizer`) via the OpenAI SDK to produce language-preserving Markdown. Defaults to the `MiniMax-M3` model at `https://api.minimaxi.com/v1`; any OpenAI-compatible vision endpoint works via `OPENAI_BASE_URL`.

## STRUCTURE

```
pdf2md-agent/
├── AGENTS.md                 # this file
├── README.md                 # user-facing quickstart
├── CONTRIBUTING.md           # conventions + anti-patterns (canonical)
├── CHANGELOG.md              # 0.1.0 → 0.2.0 evolution
├── pyproject.toml            # uv-managed, hatchling backend
├── .env.example              # OPENAI_* + PDF2MD_AGENT_* template
├── src/pdf2md_agent/         # main package — see src/pdf2md_agent/AGENTS.md
│   ├── raw_pipeline.py       # explicit LLM calls (extractor / formatter / summarizer)
│   └── crew/                 # per-page loop + cache + text-layer fallback — see crew/AGENTS.md
└── tests/                    # 10 files; stub OpenAI client, synthesize PDFs
```

## WHERE TO LOOK

| Task | Location |
|---|---|
| Run / install / env vars / flags | `README.md` |
| Style + commit + branching rules | `CONTRIBUTING.md` |
| Version history | `CHANGELOG.md` |
| Per-page artifacts produced | `.pdf2md-agent-cache/<stem>/pages/page_NNNN.{png,text.txt,resized.jpg,extract.txt,format.md}` |
| Explicit LLM calls (extract/format/summarize) | `src/pdf2md_agent/raw_pipeline.py` |
| Per-page loop + cache + text-layer fallback | `src/pdf2md_agent/crew/runner.py` |
| CLI / cache / retry / budget math | `src/pdf2md_agent/` |

## CODE MAP (top exports)

| Symbol | Location | Role |
|---|---|---|
| `pdf2md_agent.cli:main` | cli.py:486 | CLI entry; `pdf2md-agent` script + `python -m pdf2md_agent`; `--version`, `--no-cache-*`, `--request-timeout` |
| `pdf2md_agent.crew.runner.run_pipeline` | runner.py:177 | per-page loop with `CacheNoCacheFlags` (format → extract → full pipeline) + per-call `call_with_retry` |
| `pdf2md_agent.cache.CacheNoCacheFlags` | cache.py:280 | Typed per-resource opt-out switches (`render/text/resized/extract/format/summary`) |
| `pdf2md_agent.cache.MetaInfo` | cache.py:200 | Frozen fingerprint of `meta.json` (pdf, dpi, with_summary, pages, model, persona_version) |
| `pdf2md_agent.raw_pipeline.PERSONA_VERSION` | raw_pipeline.py | 16-char SHA-256 of the active persona strings; recorded in `meta.json` |
| `pdf2md_agent.raw_pipeline.call_extractor` / `call_formatter` / `call_summarizer` | raw_pipeline.py | pure LLM-call functions; inputs == observable state |
| `pdf2md_agent.raw_pipeline._make_client` | raw_pipeline.py | shared OpenAI client factory (also re-exported as `runner._make_client`) |
| `pdf2md_agent.render_skip.maybe_skip_render` | render_skip.py:18 | Trust-cache gate for per-page PNG re-render |
| `pdf2md_agent.post_stream.stitch_pages` | post_stream.py:54 | cross-page markdown joining (StreamingStitcher, default mode) |
| `pdf2md_agent.llm_retry.call_with_retry` | llm_retry.py | Fibonacci backoff + jitter, default unlimited retries, per-attempt cap `max_delay` (15 min), per-attempt timeout guard |
| `pdf2md_agent.token_budget.plan_for_image` | token_budget.py:192 | binary-search largest `long_side` that fits budget |

## CONVENTIONS (project-specific only)

- **`from __future__ import annotations`** at top of every module.
- Frozen + slotted `@dataclass` for value types (`RetryConfig`, `BudgetDecision`, `CacheLayout`, `CacheNoCacheFlags`, `MetaInfo`, `PageArtifacts`, `PageImage`, `PageResult`); avoid pydantic.
- Module-local logger: `log = logging.getLogger("pdf2md_agent.<area>")` (root logger name `"pdf2md-agent"`).
- Env vars prefixed `PDF2MD_AGENT_*`; loaded once at `config.py` import via `dotenv.load_dotenv()`. CLI flags override env.
- **LLM calls are pure functions** in `raw_pipeline.py`. Each `call_*` takes its full input as kwargs and returns the model's string output. The runner wraps each call in `call_with_retry(label="page N extract"|"format"|"summarize")` so transient failures retry per-call rather than per-page.
- Tests patch `pdf2md_agent.crew.runner._make_client` (re-export of `raw_pipeline._make_client`) to return a fake OpenAI client — no real API calls. See `tests/test_runner.py` for the `_FakeClient` / `_FakeCompletions` pattern.
- Conventional Commits (`feat:`/`fix:`/`refactor:`/`test:`/`docs:`/`chore:`); branches `feat/<name>` or `fix/<name>` from `main`.
- Cache control flags use the inverted `--no-cache-<resource>` pattern; resource names (render, text, resized, extract, format, summary) match the on-disk filenames exactly.

## ANTI-PATTERNS (do not violate)

- **Do not** import `tiktoken` — heuristic estimator in `token_budget.py` is the budget source of truth.
- **Do not** import `crewai` anywhere. The runner orchestrates explicit `client.chat.completions.create(...)` calls; reintroducing a CrewAI agent/task state machine would re-create the message-history doubling that this refactor eliminated.
- **Do not** replace `print(..., file=sys.stderr)` in `cli.py` with logger calls — CLI user-facing errors are intentional.
- **Do not** wrap `call_with_retry(fn, ...)` in a way that forwards kwargs to `fn` — `call_with_retry` takes a zero-arg callable. Use a `lambda:` closure that captures the kwargs (see `crew/runner.py:345-358`).
- **Do not** mutate `EXTRACTOR_PERSONA` / `FORMATTER_PERSONA_STRICT` / `SUMMARIZER_PERSONA` without bumping `PERSONA_VERSION` — `meta.json` records the hash so a follow-up run detects drift.
- **Do not** commit `.env`, `.pdf2md-agent-cache/`, `.venv/`, rendered PDFs.
- **Do not** bring back `--resume` or `--reformat` — the path-B rename is permanent. Use `--no-cache-format` (full re-format) or `--no-cache-extract` (re-format only when extract is cached).

## COMMANDS

```bash
uv sync                          # install deps + dev group
uv run pytest                    # run all tests (no API needed)
uv run pdf2md-agent input.pdf -o out.md
uv run python -m pdf2md_agent input.pdf -o out.md   # equivalent entry
```

## NOTES

- Cache key = PDF stem (≤ 60 chars, no path separators) **or** the first 16 chars of `sha256(absolute PDF path)`. Two PDFs at different absolute paths always land in different cache directories.
- `meta.json` carries a 6-field fingerprint (`pdf`, `dpi`, `with_summary`, `pages`, `model`, `persona_version`). A drift in any field forces a re-run on the next invocation. The persona version is the SHA-256 of the active persona strings, so editing a persona invalidates all dependent cache files.
- `--no-cache-format` short-circuits the entire per-page pipeline when `format.md` is on disk. `--no-cache-extract` re-runs only the formatter when `extract.txt` is on disk. `--no-cache-all` is the universal kill switch.
- The MiniMax-M3 endpoint occasionally returns scratchpad blocks (delimited by XML-like tags) in formatter output; `_strip_think()` in `raw_pipeline.py` removes them defensively.
- `StreamingStitcher` (post-`#5`) defaults ON via `--stitch-mode heuristic`; legacy `\n\n---\n\n` separator retained only when `--stitch-mode off`.

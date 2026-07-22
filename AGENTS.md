# PROJECT KNOWLEDGE BASE

**Generated:** 2026-07-21
**Commit:** 23f448d (main)
**Stack:** Python ≥3.10 · PyMuPDF · CrewAI ≥0.80,<2 · OpenAI SDK 1.x · Pillow · hatchling · uv

## OVERVIEW

`pdf2md-agent` renders each PDF page to an image and feeds it through a CrewAI pipeline of vision agents (extractor → formatter → summarizer) to produce language-preserving Markdown. Defaults to the `MiniMax-M3` model at `https://api.minimaxi.com/v1`; any OpenAI-compatible vision endpoint works via `OPENAI_BASE_URL`.

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
│   └── crew/                 # CrewAI orchestration — see crew/AGENTS.md
└── tests/                    # 10 files; monkeypatch LLM, synthesize PDFs
```

## WHERE TO LOOK

| Task | Location |
|---|---|
| Run / install / env vars / flags | `README.md` |
| Style + commit + branching rules | `CONTRIBUTING.md` |
| Version history | `CHANGELOG.md` |
| Per-page artifacts produced | `.pdf2md-agent-cache/<stem>/pages/page_NNNN.{png,text.txt,resized.jpg,extract.txt,format.md}` |
| Pipeline orchestration | `src/pdf2md_agent/crew/` |
| CLI / cache / retry / budget math | `src/pdf2md_agent/` |

## CODE MAP (top exports)

| Symbol | Location | Role |
|---|---|---|
| `pdf2md_agent.cli:main` | cli.py:486 | CLI entry; `pdf2md-agent` script + `python -m pdf2md_agent`; `--version`, `--no-cache-*`, `--request-timeout` |
| `pdf2md_agent.crew.runner.run_pipeline` | runner.py:160 | per-page crew orchestration with `CacheNoCacheFlags` (format → extract → full pipeline) |
| `pdf2md_agent.cache.CacheNoCacheFlags` | cache.py:280 | Typed per-resource opt-out switches (`render/text/resized/extract/format/summary`) |
| `pdf2md_agent.cache.MetaInfo` | cache.py:200 | Frozen fingerprint of `meta.json` (pdf, dpi, with_summary, pages, model, persona_version) |
| `pdf2md_agent.crew.agents.PERSONA_VERSION` | agents.py:42 | 16-char SHA-256 of the active persona strings; recorded in `meta.json` |
| `pdf2md_agent.render_skip.maybe_skip_render` | render_skip.py:18 | Trust-cache gate for per-page PNG re-render |
| `pdf2md_agent.multimodal_patch.patch_add_image_tool` | multimodal_patch.py:134 | idempotent monkey-patch on `AddImageTool._run` (REQUIRED) |
| `pdf2md_agent.post_stream.stitch_pages` | post_stream.py:54 | cross-page markdown joining (StreamingStitcher, default mode) |
| `pdf2md_agent.llm_retry.call_with_retry` | llm_retry.py:130 | bounded exponential backoff + jitter + per-attempt timeout guard |
| `pdf2md_agent.token_budget.plan_for_image` | token_budget.py:192 | binary-search largest `long_side` that fits budget |

## CONVENTIONS (project-specific only)

- **`from __future__ import annotations`** at top of every module.
- Frozen + slotted `@dataclass` for value types (`RetryConfig`, `BudgetDecision`, `CacheLayout`, `CacheNoCacheFlags`, `MetaInfo`, `PageArtifacts`, `PageImage`, `PageResult`); avoid pydantic.
- Module-local logger: `log = logging.getLogger("pdf2md_agent.<area>")` (root logger name `"pdf2md-agent"`).
- Env vars prefixed `PDF2MD_AGENT_*`; loaded once at `config.py` import via `dotenv.load_dotenv()`. CLI flags override env.
- Tests monkeypatch `make_vision_llm` at `pdf2md_agent.crew.runner.make_vision_llm` (re-exported `noqa: F401`) — no real API calls.
- Conventional Commits (`feat:`/`fix:`/`refactor:`/`test:`/`docs:`/`chore:`); branches `feat/<name>` or `fix/<name>` from `main`.
- Cache control flags use the inverted `--no-cache-<resource>` pattern; resource names (render, text, resized, extract, format, summary) match the on-disk filenames exactly.

## ANTI-PATTERNS (do not violate)

- **Do not** import `tiktoken` — heuristic estimator in `token_budget.py` is the budget source of truth.
- **Do not** raise `crewai` pin above `0.80,<2` — older versions don't expose `crewai.tools.agent_tools.add_image_tool`.
- **Do not** replace `print(..., file=sys.stderr)` in `cli.py` with logger calls — CLI user-facing errors are intentional.
- **Do not** work around `crew/multimodal_patch.py` by importing `crewai.tools` directly in tests — patch `pdf2md_agent.crew.runner.<name>` instead.
- **Do not** strip `# type: ignore` comments in `multimodal_patch.py` — three are load-bearing.
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
- The MiniMax-M3 endpoint occasionally returns scratchpad blocks (delimited by XML-like tags) in formatter output; `_strip_think()` in `crew/runner.py:62` removes them defensively.
- `StreamingStitcher` (post-`#5`) defaults ON via `--stitch-mode heuristic`; legacy `\n\n---\n\n` separator retained only when `--stitch-mode off`.

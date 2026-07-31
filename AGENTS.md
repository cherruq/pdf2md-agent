# PROJECT KNOWLEDGE BASE

**Generated:** 2026-07-27
**Commit:** a654115 (main)
**Stack:** Python ≥3.10 · PyMuPDF · CrewAI ≥0.80,<2 · OpenAI SDK 1.x · Pillow · hatchling · uv

## OVERVIEW

`pdf2md-agent` renders each PDF page to an image and feeds it through a CrewAI vision extraction agent to produce language-preserving Markdown. Defaults to the `MiniMax-M3` model at `https://api.minimaxi.com/v1`; any OpenAI-compatible vision endpoint works via `OPENAI_BASE_URL`.

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
│   ├── cli.py                # CLI entry gateway + arg parser invocation
│   ├── pipeline.py           # ConversionConfig struct + run_unified_conversion 3-step engine
│   ├── cli_parser.py         # argparse definition + post-parse resolvers
│   ├── filesystem_safety.py  # cache-key / Windows-reserved-name helpers
│   ├── cache.py              # CacheLayout + meta validation + atomic writes
│   ├── config.py             # env bindings + ctx-limit resolver
│   ├── tuning.py             # internal tuning constants (budget, reflection, stitch mode)
│   ├── llm_retry.py          # RetryConfig + Fibonacci backoff + timeout guard
│   ├── token_estimator.py    # text + image token heuristics
│   ├── image_budget.py       # plan_for_image + BudgetDecision (binary search)
│   ├── pages.py              # --pages parser
│   ├── pdf_renderer.py       # PyMuPDF render
│   ├── post_stream.py        # StreamingStitcher public API
│   ├── post_stream_decision.py  # block split + continuation + smart join + CJK
│   ├── post_stream_table.py  # table-row continuation + header dedup
│   ├── vision.py             # make_vision_llm factory
│   └── crew/                 # CrewAI orchestration — see crew/AGENTS.md
│       ├── runner.py         # per-page orchestrator (run_pipeline)
│       ├── extraction.py     # per-page extraction loop
│       ├── page_image.py     # budget planning + tiling + downscale
│       ├── fallback.py       # text-layer fallback helpers + FallbackRecord
│       ├── output.py         # _strip_think + _output
│       ├── results.py        # PageResult shared value type
│       ├── agents.py         # 1 persona + agent factory
│       ├── tasks.py          # task factories + patch_add_image_tool()
│       └── multimodal_patch.py  # AddImageTool monkey-patch
└── tests/                    # 14 files; monkeypatch LLM, synthesize PDFs — see tests/AGENTS.md
```

## WHERE TO LOOK

| Task | Location |
|---|---|
| Run / install / env vars / flags | `README.md` |
| Style + commit + branching rules | `CONTRIBUTING.md` |
| Version history | `CHANGELOG.md` |
| Per-page artifacts produced | `.pdf2md-agent-cache/<stem>/pages/page_NNNN.{png,text.txt,resized.jpg,format.md}` |
| Pipeline orchestration | `src/pdf2md_agent/crew/` |
| CLI / cache / retry / budget math | `src/pdf2md_agent/` |
| Test conventions (mocking, fixtures, run cmd) | `tests/AGENTS.md` |
| End-to-end PDF + Markdown samples | `sample/{test.pdf,test.md}` |

## CODE MAP (top exports)

| Symbol | Location | Role |
|---|---|---|
| `pdf2md_agent.cli:main` | cli.py | CLI entry; `pdf2md-agent` script + `python -m pdf2md_agent`; slim gateway into `pipeline.py` |
| `pdf2md_agent.pipeline.run_unified_conversion` | pipeline.py | 3-step pipeline orchestrator running off immutable `ConversionConfig` struct |
| `pdf2md_agent.crew.runner.run_pipeline` | crew/runner.py | per-page crew orchestration with `CacheNoCacheFlags` (format short-circuit → full pipeline) |
| `pdf2md_agent.crew.extraction.run_extraction_loop` | crew/extraction.py | one page's extraction loop |
| `pdf2md_agent.crew.page_image.prepare_page_image` | crew/page_image.py | budget + downscale + tile-split for one page |
| `pdf2md_agent.crew.fallback._record_text_layer_fallback` | crew/fallback.py | write the fenced text-layer stub on retry exhaustion |
| `pdf2md_agent.cache.CacheNoCacheFlags` | cache.py | Typed per-resource opt-out switches (`render/text/resized/format`) |
| `pdf2md_agent.cache.MetaInfo` | cache.py | Frozen fingerprint of `meta.json` (`pdf` file path) |
| `pdf2md_agent.multimodal_patch.patch_add_image_tool` | crew/multimodal_patch.py | idempotent monkey-patch on `AddImageTool._run` (REQUIRED) |
| `pdf2md_agent.post_stream.stitch_pages` | post_stream.py | cross-page markdown joining (StreamingStitcher, default mode) |
| `pdf2md_agent.post_stream_decision` | post_stream_decision.py | block split + continuation + smart join + CJK detection (private helpers) |
| `pdf2md_agent.post_stream_table` | post_stream_table.py | table-row continuation + header dedup (private helpers) |
| `pdf2md_agent.llm_retry.call_with_retry` | llm_retry.py | Fibonacci backoff + jitter, default unlimited retries, per-attempt cap `max_delay` (15 min), per-attempt timeout guard |
| `pdf2md_agent.token_estimator.estimate_text_tokens` | token_estimator.py | coarse CJK + ASCII heuristic |
| `pdf2md_agent.token_estimator.estimate_image_tokens` | token_estimator.py | byte-count → base64 token estimate (no pixel decode) |
| `pdf2md_agent.image_budget.plan_for_image` | image_budget.py | binary-search largest `long_side` that fits budget |
| `pdf2md_agent.filesystem_safety.cache_key_for_pdf` | filesystem_safety.py | deterministic cache directory name (Windows-reserved-name + path-traversal safe) |
| `pdf2md_agent.cli_parser.build_parser` | cli_parser.py | argparse builder; groups: Pipeline / Cache control / Feature disable / Retry & tuning / Diagnostic |
| `pdf2md_agent.cli_parser.build_retry_config` | cli_parser.py | args → `RetryConfig` (CLI override + env fallback) |

## CONVENTIONS (project-specific only)

- **`from __future__ import annotations`** at top of every module.
- Frozen + slotted `@dataclass` for value types (`ConversionConfig`, `RetryConfig`, `BudgetDecision`, `CacheLayout`, `CacheNoCacheFlags`, `MetaInfo`, `PageArtifacts`, `RenderedPage`, `PageResult`); avoid pydantic.
- Module-local logger: `log = logging.getLogger("pdf2md_agent.<area>")` (root logger name `"pdf2md-agent"`).
- Env vars prefixed `PDF2MD_AGENT_*`; loaded once at `config.py` import via `dotenv.load_dotenv()`. CLI flags override env.
- Tests monkeypatch `make_vision_llm` at `pdf2md_agent.crew.runner.make_vision_llm` (re-exported `noqa: F401`) — no real API calls.
- Conventional Commits (`feat:`/`fix:`/`refactor:`/`test:`/`docs:`/`chore:`); branches `feat/<name>` or `fix/<name>` from `main`.
- Cache control flags use the inverted `--no-cache-<resource>` pattern; resource names (render, text, resized, format) match the on-disk filenames exactly.
- `# type: ignore` and `# noqa: XXXX` may only be used **with an explanatory inline comment** (CONTRIBUTING.md:57-60). Stripping the comment is a violation even if the suppression itself stays.
- `secrets.SystemRandom()` (not `random`) for retry jitter — backoff sequences cannot sync across parallel clients (`llm_retry.py:30`).
- End-to-end PDF/Markdown test samples live in `sample/` (`test.pdf` + `test.md`), not in `tests/`. `tests/AGENTS.md` forbids committed rendered PDFs and `.pdf2md-agent-cache/`; the `sample/` directory is outside pytest's auto-discovery and is the home for committed reproducible test material.

## ANTI-PATTERNS (do not violate)

- **Do not** import `tiktoken` — heuristic estimator in `token_estimator.py` is the budget source of truth.
- **Do not** raise `crewai` pin above `0.80,<2` — older versions don't expose `crewai.tools.agent_tools.add_image_tool`.
- **Do not** replace `print(..., file=sys.stderr)` in `cli.py` with logger calls — CLI user-facing errors are intentional.
- **Do not** work around `crew/multimodal_patch.py` by importing `crewai.tools` directly in tests — patch `pdf2md_agent.crew.runner.<name>` instead.
- **Do not** strip `# type: ignore` comments in `multimodal_patch.py` — three are load-bearing (lines 45, 153, 161).
- **Do not** strip any `# noqa: F401` re-export comments in `crew/runner.py` — tests patch every helper at `pdf2md_agent.crew.runner.<name>`; removing them forces `create=True` and breaks the test surface.
- **Do not** strip the `# noqa: BLE001` comments in `llm_retry.py` — the `is_transient` predicate below discriminates, and the daemon-thread joiner re-raises.
- **Do not** commit `.env`, `.pdf2md-agent-cache/`, `.venv/`, rendered PDFs.
- **Do not** bring back `--resume` or `--reformat` — the path-B rename is permanent. Use `--no-cache-format` (full extraction re-run).
- **Do not** move the lazy import of `pdf2md_agent.crew.extraction.run_extraction_loop` to module top in `crew/runner.py` — it's lazy specifically so the runner ↔ extraction import order stays acyclic (extraction looks up agent / task factories via the runner namespace to preserve the test patch surface).

## COMMANDS

```bash
uv sync                          # install deps + dev group
uv run pytest                    # run all tests (no API needed)
uv run pytest -ra tests/         # CI-equivalent: -ra = summary, no passed lines
uv run pdf2md-agent input.pdf -o out.md
uv run python -m pdf2md_agent input.pdf -o out.md   # equivalent entry
uv run pdf2md-agent --no-cache-all -o sample_output.md sample/test.pdf   # end-to-end evaluation on sample PDF
```

## NOTES

- Cache key = PDF stem (≤ 60 chars, no path separators) **or** the first 16 chars of `sha256(absolute PDF path)`. Two PDFs at different absolute paths always land in different cache directories.
- `meta.json` carries a simple validation fingerprint (`pdf` absolute file path). A drift in the canonical file path forces a re-run on the next invocation.
- Version is duplicated: `pyproject.toml` and `src/pdf2md_agent/__init__.py` both pin `0.2.1` — update both together.
- `--no-cache-format` short-circuits the entire per-page pipeline when `format.md` is on disk. `--no-cache-all` is the universal kill switch.
- The MiniMax-M3 endpoint occasionally returns scratchpad blocks (delimited by XML-like tags) in model output; `_strip_think()` in `crew/output.py` removes them defensively.
- `StreamingStitcher` (post-`#5`) defaults ON via `--stitch-mode heuristic`; legacy `\n\n---\n\n` separator retained only when `--stitch-mode off`.
- **No CI workflow exists** in `.github/workflows/`. CONTRIBUTING.md says "CI runs the same `uv run pytest`"; keep that promise by running the command locally before pushing.

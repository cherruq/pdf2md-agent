# `pdf2md_agent.crew` — CrewAI Orchestration

Subpackage containing the agents, task definitions, runner, focused
per-page helpers, and a load-bearing monkey-patch on CrewAI's
`AddImageTool`. All `from crewai` imports concentrated here.

## FILE RESPONSIBILITIES

| File | Purpose | Entry point |
|---|---|---|
| `__init__.py` | empty marker | |
| `agents.py` | 4 personas + agent factories | `make_extractor`, `make_formatter`, `make_summarizer` |
| `tasks.py` | crewAI task builders + `patch_add_image_tool()` call | `make_extract_task`, `make_format_task`, `make_format_task_from_extract_file`, `make_summarize_task` |
| `runner.py` | per-page pipeline orchestrator | `run_pipeline`, `_run_format_summarize_only` |
| `extraction.py` | per-page extract → format → (reflect) → summarize loop | `run_extraction_loop`, `_strip_multipage_headers_footers` |
| `page_image.py` | token-budget planning + downscale + tile splitting | `prepare_page_image`, `_resize_page_png`, `_make_tiles` |
| `fallback.py` | text-layer fallback markdown + sentinel | `_text_layer_fallback`, `_record_text_layer_fallback`, `FallbackRecord` |
| `output.py` | CrewAI output extraction + think-block stripping | `_output`, `_strip_think` |
| `results.py` | `PageResult` (shared value type) | `PageResult` |
| `multimodal_patch.py` | **monkey-patch on `AddImageTool._run`** | `patch_add_image_tool`, `_encode_local_image`, `_to_sentinel` |

## KEY SYMBOLS (with refs)

| Symbol | File | Role |
|---|---|---|
| `run_pipeline` | runner.py | `CacheNoCacheFlags`-driven per-page pipeline (format short-circuit → extract short-circuit → full) |
| `run_extraction_loop` | extraction.py | One page's extract → format → (optional reflect) → (optional summarize) |
| `prepare_page_image` | page_image.py | Estimate persona + prompt + image cost, downscale or tile the page |
| `_record_text_layer_fallback` | fallback.py | Write the fenced text-layer stub + sentinel on retry exhaustion / validation error |
| `patch_add_image_tool` | multimodal_patch.py | idempotent install; safe to re-call to refresh dims |
| `_encode_local_image` | multimodal_patch.py | Pillow LANCZOS downscale + JPEG + b64 |
| `_to_data_url` | multimodal_patch.py | builds `data:image/jpeg;base64,…` URL |
| `_to_sentinel` | multimodal_patch.py | builds `VISION_IMAGE:<media-type>:<b64>` string |
| `_strip_think` | output.py | defensive removal of scratchpad tags the configured MiniMax-M3 endpoint occasionally emits |
| `_persona_backstory` | agents.py | splits persona `"role\n\nbackstory"` (CrewAI's `backstory` reads after `\n\n`) |
| `_truncate_summary` | tasks.py | head+tail truncation with sentinel; preserves summary shape |
| `_text_layer_fallback` | fallback.py | fenced text-layer stub for transient / validation failures |
| `EXTRACTOR_PERSONA` / `_BACKSTORY` | agents.py | exported separately so token-budget planner can pre-compute cost |
| `MAX_SUMMARY_CHARS` | tasks.py | default 800; injected into summary task description |
| `FALLBACK_SENTINEL` | re-exported from cache.py | sentinel string; `cache.has_cached_extract` checks the same prefix |

## CONVENTIONS (specific to this subpackage)

- **Persona strings are short** (~60 words each) to fit `MiniMax-M3`'s 512K-1M context window alongside the page image. Length budgeted in `image_budget.py` before pipeline start.
- **Persona shape**: `"<role-text>\n\n<backstory-text>"` — CrewAI's `Agent(backstory=...)` only reads what's after the first `\n\n`. `_persona_backstory()` does the partition.
- **Re-exports with `noqa: F401`** in `runner.py` for every helper tests patch (`render_pdf`, `PageImage`, `make_vision_llm`, `Crew`, `make_extractor`, `make_formatter`, `make_summarizer`, `make_extract_task`, `make_format_task`, `make_format_task_from_extract_file`, `make_summarize_task`, `_truncate_summary`, `_output`, `_strip_think`, `_record_text_layer_fallback`, `_text_layer_fallback`, `_resize_page_png`, `write_summary`, `_FALLBACK_SENTINEL`). Tests patch at `pdf2md_agent.crew.runner.<name>` — do not remove the re-exports.
- **Patch-surface preservation**: `extraction.py` looks up the agent / task factories and `Crew` via the `runner` module's namespace (``runner.make_extractor``, etc.) so test patches at ``runner.*`` are honored at the call sites. ``runner.py`` lazy-imports ``extraction.py`` to keep this loop acyclic.
- **`patch_add_image_tool()` is invoked at import time** from `tasks.py`. Tests that need different dims can re-call it; module-level `_active_long_side` / `_active_jpeg_quality` are updated in place without reinstalling the patch.
- **`<think>` / `</think>` escaping**: written as `chr(60) + "think" + chr(62)` in `tasks.py` and `output.py` — avoids mangling by downstream XML-processing tools. Do not "refactor" to literal `<think>`.
- `PERSONA_VERSION` (in `agents.py`) is the 16-char SHA-256 of the active persona strings. Any edit to a persona changes this hash and invalidates the entire cache (fingerprint field in `meta.json`).
- The `_record_text_layer_fallback` kwarg signature (``idx``, ``total``, ``page_number``, ``page_started``, ``artifacts``, ``summary``, ``completion_label``) matches the historical contract tests rely on; new callers can construct a :class:`FallbackRecord` instead.

## ANTI-PATTERNS (this subpackage)

- **NEVER replace `multimodal_patch.py`.** Stock CrewAI's `AddImageTool._run` (a) forwards local paths verbatim → rejected with HTTP 400 by OpenAI-compatible vision APIs, and (b) returns a dict instead of the `VISION_IMAGE:<media-type>:<base64>` sentinel string that `crewai.execution.StepExecutor` recognizes. The patch fixes both. Calling it directly without the patch will silently break the pipeline.
- **NEVER strip the three `# type: ignore` comments in `multimodal_patch.py`**:
  - line 45 — `UnidentifiedImageError = OSError` fallback when `PIL.UnidentifiedImageError` isn't importable
  - line 153 — `# type: ignore[override]` (parent `BaseTool._run` has a different signature)
  - line 161 — `# type: ignore[assignment]` (assigning onto foreign class method)
- **NEVER strip the `# noqa: F401` re-exports in `runner.py`** — tests patch `pdf2md_agent.crew.runner.render_pdf`, `make_vision_llm`, `Crew`, the agent / task factories, etc. at these names; removing them forces `create=True` and breaks the test surface.
- **NEVER import `crewai.tools.agent_tools.add_image_tool` directly in tests.** Tests must monkeypatch `pdf2md_agent.crew.runner.make_vision_llm` (and friends). Direct import path bypasses the patched `_run` and reintroduces the original bug.
- **NEVER catch and suppress `ValidationError` inside `run_extraction_loop`** — the fallback path `_record_text_layer_fallback` depends on it propagating.
- **NEVER narrow `except BaseException` to `Exception`** in `extraction.py` (per-page retry-exhaustion path) or `runner.py` (`_run_format_summarize_only` retry-exhaustion path). Both are intentional for cleanup before re-raise.

## NOTES

- `multimodal_patch.py` reads module-level globals (`_patched: bool`, `_active_long_side`, `_active_jpeg_quality`). Tests that mutate these must reset them in `finally` (or use `monkeypatch.setattr` which handles cleanup).
- The patch returns `str` (the `VISION_IMAGE:` sentinel), not a dict. CrewAI's `StepExecutor` requires the str shape; tests assert on the str.
- Tasks propagate `_NO_REASONING` (the chr-escaped phrase) into the agent's `system` instruction to keep prompts tight.
- `--no-cache-format` short-circuits the entire per-page pipeline when `format.md` is on disk (uses `is_page_complete` in `cache.py`); `--no-cache-extract` triggers `_run_format_summarize_only` to re-use cached `extract.txt`.
- `crew/results.py` holds `PageResult` so both `crew/runner.py` and `crew/fallback.py` (and `post_stream.py`) can construct one without an import cycle.
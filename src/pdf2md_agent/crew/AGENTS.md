# `pdf2md_agent.crew` — CrewAI Orchestration

Subpackage containing the agents, task definitions, runner, and a load-bearing monkey-patch on CrewAI's `AddImageTool`. All `from crewai` imports concentrated here.

## FILE RESPONSIBILITIES

| File | Purpose | Entry point |
|---|---|---|
| `__init__.py` | empty marker | |
| `agents.py` | 4 personas + agent factories | `make_extractor`, `make_formatter`, `make_format_from_extract`, `make_summarizer` |
| `tasks.py` | crewAI task builders + `patch_add_image_tool()` call | `make_extract_task`, `make_format_task`, `make_format_task_from_extract_file`, `make_summarize_task` |
| `runner.py` | per-page pipeline orchestrator | `run_pipeline`, `_run_format_summarize_only` |
| `multimodal_patch.py` | **monkey-patch on `AddImageTool._run`** | `patch_add_image_tool`, `_encode_local_image`, `_to_sentinel` |

## KEY SYMBOLS (with refs)

| Symbol | File:Line | Refs (≈) | Role |
|---|---|---|---|
| `run_pipeline` | runner.py:160 | heavy | resume → reformat → full pipeline, per-page crew + retry + fallback |
| `patch_add_image_tool` | multimodal_patch.py:134 | 1 (called from tasks.py import) | idempotent install; safe to re-call to refresh dims |
| `_encode_local_image` | multimodal_patch.py:48 | 1 | Pillow LANCZOS downscale + JPEG + b64 |
| `_to_sentinel` | multimodal_patch.py:102 | 1 | builds `VISION_IMAGE:<media-type>:<b64>` string |
| `_strip_think` | runner.py:61 | internal | defensive removal of scratchpad tags the configured MiniMax-M3 endpoint occasionally emits |
| `_persona_backstory` | agents.py:133 | internal | splits persona `"role\n\nbackstory"` (CrewAI's `backstory` reads after `\n\n`) |
| `_truncate_summary` | tasks.py:57 | internal | head+tail truncation with sentinel; preserves summary shape |
| `EXTRACTOR_PERSONA` / `_BACKSTORY` | agents.py | external | exported separately so token-budget planner can pre-compute cost |
| `MAX_SUMMARY_CHARS` | tasks.py:23 | external | default 800; injected into summary task description |

## CONVENTIONS (specific to this subpackage)

- **Persona strings are short** (~60 words each) to fit `MiniMax-M3`'s ~2k context window alongside the page image. Length budgeted in `token_budget.py` before pipeline start.
- **Persona shape**: `"<role-text>\n\n<backstory-text>"` — CrewAI's `Agent(backstory=...)` only reads what's after the first `\n\n`. `_persona_backstory()` does the partition.
- **Re-exports with `noqa: F401`** in `runner.py:46,52` (`render_pdf`, `make_vision_llm` re-exported so tests can patch at `pdf2md_agent.crew.runner.<name>`). Do not remove the re-exports.
- **`patch_add_image_tool()` is invoked at import time** from `tasks.py:20`. Tests that need different dims can re-call it; module-level `_active_long_side` / `_active_jpeg_quality` are updated in place without reinstalling the patch.
- **`<think>` / `</think>` escaping**: in `runner.py:56-58` and `tasks.py:25-28` written as `chr(60) + "think" + chr(62)` etc. — avoids mangling by downstream XML-processing tools. Do not "refactor" to literal `<think>`.

## ANTI-PATTERNS (this subpackage)

- **NEVER replace `multimodal_patch.py`.** Stock CrewAI's `AddImageTool._run` (a) forwards local paths verbatim → rejected with HTTP 400 by OpenAI-compatible vision APIs, and (b) returns a dict instead of the `VISION_IMAGE:<media-type>:<base64>` sentinel string that `crewai.execution.StepExecutor` recognizes. The patch fixes both. Calling it directly without the patch will silently break the pipeline.
- **NEVER strip the three `# type: ignore` comments in `multimodal_patch.py`**:
  - line 45 — `UnidentifiedImageError = OSError` fallback when `PIL.UnidentifiedImageError` isn't importable
  - line 153 — `# type: ignore[override]` (parent `BaseTool._run` has a different signature)
  - line 161 — `# type: ignore[assignment]` (assigning onto foreign class method)
- **NEVER import `crewai.tools.agent_tools.add_image_tool` directly in tests.** Tests must monkeypatch `pdf2md_agent.crew.runner.make_vision_llm` (and friends). Direct import path bypasses the patched `_run` and reintroduces the original bug.
- **NEVER catch and suppress `ValidationError` inside `run_pipeline`** — the fallback path `_text_layer_fallback` depends on it propagating.
- `runner.py:404` uses `except BaseException` — that is intentional for retry-exhaustion cleanup. Do not narrow to `Exception`.

## NOTES

- `multimodal_patch.py` reads module-level globals (`_patched: bool`, `_active_long_side`, `_active_jpeg_quality`). Tests that mutate these must reset them in `finally` (or use `monkeypatch.setattr` which handles cleanup).
- The patch returns `str` (the `VISION_IMAGE:` sentinel), not a dict. CrewAI's `StepExecutor` requires the str shape; tests assert on the str.
- Tasks propagate `_NO_REASONING` (the chr-escaped phrase) into the agent's `system` instruction to keep prompts tight.

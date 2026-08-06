# `tests/` — Test Suite

Flat directory; one module-level test file per source module. No `conftest.py`, no shared fixtures — every helper is inline in the test module that uses it.

## WHERE TO LOOK

| Source module | Test file |
|---|---|
| `src/pdf2md_agent/cache.py` | `test_cache.py` |
| `src/pdf2md_agent/cli.py` & `pipeline.py` | `test_misc_coverage.py`, `test_d8_coverage.py` |
| `src/pdf2md_agent/config.py` | `test_config.py` |
| `src/pdf2md_agent/llm_retry.py` | `test_llm_retry.py` |
| `src/pdf2md_agent/pages.py` | `test_pages.py` |
| `src/pdf2md_agent/pdf_renderer.py` | `test_pdf_renderer.py` |
| `src/pdf2md_agent/post_stream.py` | `test_post_stream.py` |
| `src/pdf2md_agent/image_budget.py` & `token_estimator.py` | `test_token_budget.py` |
| `src/pdf2md_agent/vision.py` | `test_vision.py` |
| `src/pdf2md_agent/crew/runner.py` | `test_runner.py`, `test_no_cache.py` |
| `src/pdf2md_agent/crew/multimodal_patch.py` | `test_d8_coverage.py` (encoding helpers), `test_runner.py` (via `make_vision_llm` patch) |

## INVOCATION

```bash
uv run pytest                    # full suite (no API calls)
uv run pytest -ra tests/         # CI-equivalent: summary, no passed lines
uv run pytest tests/test_runner.py -k falls_back -v   # single test
```

`pyproject.toml` `[tool.pytest.ini_options]`: `pythonpath = ["src"]`, `testpaths = ["tests"]`, `addopts = ["-ra"]`.

## MOCKING STRATEGY

- **Patch where the name is looked up**, not where it is defined. Use `patch.object(extraction, "make_extractor")`, `patch.object(extraction, "Crew")`, etc. — never patch the original `pdf2md_agent.vision.make_vision_llm` directly.
- LLM/Crew mocking: `patch.object(extraction, "make_extractor")`, `patch.object(extraction, "make_extract_task", return_value=extract_t)`. Make `Crew.kickoff` a `lambda: None` or a side-effect that returns the canned extractor output.
- Config / env manipulation: `monkeypatch.setattr(config, "OPENAI_BASE_URL", ...)` or `monkeypatch.setenv(...)`.
- Atomic-write crash simulation: `monkeypatch.setattr("os.write", crash_after_open)`.
- CWD-sensitive cache path tests: `monkeypatch.chdir(tmp_path)`.

## FIXTURES & ISOLATION

- No committed PDFs — every test synthesizes one in-memory via PyMuPDF (`_make_pdf`, `_make_onepage_pdf`) using `pymupdf.open() → new_page() → insert_text()`.
- Page images are generated with `PIL.Image.new(...).save(...)` for resize/patch tests.
- All file I/O uses the `tmp_path` pytest fixture — automatic per-test cleanup, no global state.
- Inline helpers (no `conftest.py`):
  - `_make_pdf(path, pages)`, `_make_onepage_pdf(path)`
  - `_write_png(path, width, height, color)`
  - `_page(n, markdown)` (PageResult factory)
  - `_FakeTask(raw)`, `_FakeOutput(raw)` (CrewAI stand-ins)
  - `_layout(tmp_path, page_number)` (CacheLayout factory)
  - `_response(status)`, `_request()` (httpx fixtures)
  - `_make_solid_png(path, size)` (PIL PNG for token estimation)

## API ISOLATION RULES

- **No real API calls anywhere.** Tests monkeypatch `make_vision_llm` where it is used (e.g. `pdf2md_agent.pipeline.make_vision_llm`).
- Real PyMuPDF is used only for PDF synthesis and rendering — no network I/O.
- For retry/backoff assertions, `call_with_retry` accepts a `sleep=` kwarg injected as a list-append so tests verify the Fibonacci sequence without sleeping.

## REPRESENTATIVE PATTERNS

1. **Happy-path runner** (`test_runner.py`): synthesizes layout + page → patches all maker functions + `Crew.kickoff` → asserts result markdown and on-disk files.
2. **Cache fingerprinting** (`test_cache.py`): parametrized field-diff detection via `check_meta_matches`.
3. **Retry backoff** (`test_llm_retry.py`): tracks `sleeps` list injected via `sleep=` kwarg → validates Fibonacci sequence (`1, 1, 2, 3, 5, 8, 13…`).
4. **Stitcher buffering** (`test_post_stream.py`): feeds pages to `StreamingStitcher` via `feed()` → `finalize()`, asserts no `---` separator in merged output.
5. **Token-budget binary search** (`test_token_budget.py`): PIL solid-color PNGs of varying sizes validate the size-estimation heuristic.
6. **CLI smoke** (`test_misc_coverage.py`): `cli.main([...])` against `capsys` — argparse argument groups, version, atomic write, traversal-rejection.

## ANTI-PATTERNS (test layer)

- **Do not** import `crewai.tools.agent_tools.add_image_tool` directly in any test — the direct path bypasses the monkey-patched `_run` and reintroduces the original HTTP-400 bug. Patch `pdf2md_agent.crew.extraction.<name>` instead.
- **Do not** mutate `multimodal_patch._patched`, `_active_long_side`, or `_active_jpeg_quality` without resetting in `finally`. Use `monkeypatch.setattr` to handle cleanup automatically.
- **Do not** run tests with a real `OPENAI_API_KEY` — there is no offline-vs-online branch in the test surface. Tests assume monkeypatched LLM end-to-end.
- **Do not** commit rendered PDFs, `.pdf2md-agent-cache/` directories, or `.env` files (root `AGENTS.md` anti-pattern applies here).
- **Do not** add a `conftest.py` unless shared state becomes unavoidable — the project convention is module-local helpers so each file reads top-to-bottom.

## NOTES

- `pythonpath = ["src"]` is the only thing putting `pdf2md_agent` on the import path for tests; do not edit `tests/__init__.py` to add path hacks.
- Failure to monkeypatch at the lookup site (e.g., `crew.extraction`) is the most common cause of "tests pass but CI fails against a real endpoint" — keep the patch targets consistent.
- The 6-field `meta.json` fingerprint is exercised end-to-end in `test_cache.py`; changing any field name in `MetaInfo` requires updating those tests.
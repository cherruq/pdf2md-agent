# Contributing to `pdf2md-agent`

Thanks for your interest in contributing. This project is small and favours
short, surgical PRs over large refactors.

## Getting started

```bash
git clone https://github.com/cherruq/pdf2md-agent.git
cd pdf2md-agent
uv sync                          # installs deps + the package in editable mode
uv run pytest                    # full test suite (does NOT call the API)
uv run pdf2md-agent --help
```

## Running a single test

```bash
uv run pytest tests/test_runner.py -k extract_then_format -v
```

## What to work on

- **Bug reports**: include the failing CLI invocation, the relevant cache
  files under `.pdf2md-agent-cache/<pdf-stem>/pages/`, and the full
  provider response / error message. The cache files are exactly what
  the runner saw, so they're enough to reproduce offline.
- **Features**: open an issue first. Most non-trivial additions should be
  discussed before code is written — the runner / cache layout has
  invariants that are easy to break by accident.

## Pull requests

1. **Branch from `main`.** Use a topic branch:
   `git switch -c feat/<short-name>` or `fix/<short-name>`.
2. **Keep commits atomic.** One logical change per commit. Use
   [Conventional Commits](https://www.conventionalcommits.org/) prefixes
   where possible: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`,
   `chore:`.
3. **Tests required for behavior changes.** Run `uv run pytest` before
   pushing; CI runs the same command.
4. **No new dependencies without justification.** If you need one, call
   it out in the PR description.
5. **No commits of `.env`, cache files, or rendered PDFs.** The
   `.gitignore` already covers these — verify with `git status` before
   pushing.

## Coding style

- **Python 3.10+**, type hints throughout, `from __future__ import annotations`
  in every module.
- **Frozen+slots dataclasses** for value types. Avoid `pydantic.BaseModel`
  unless you need validation — dataclasses are cheaper and match the rest
  of the codebase.
- **Logging** via module-local `log = logging.getLogger("pdf2md-agent.<area>")`.
  No `print()` outside the CLI entry point.
- **No `as any` / `@ts-ignore` equivalents** — never silence type errors.
  This is a Python project, so the relevant analogues are:
  `# type: ignore` (use only with a comment explaining why) and
  `# noqa: XXXX` (same).

## Issue / PR labels

This repo is small enough that the maintainer triages manually — there
isn't a label taxonomy yet. Don't worry about labels; just describe the
problem / change clearly in the body.

## Security

If you find a security issue (a leaked secret in a commit, a path
traversal, an injection vector in the prompt assembly), please email
the maintainer privately **before** opening a public issue.

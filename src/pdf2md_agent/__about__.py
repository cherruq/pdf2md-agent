"""Single source of truth for the package version.

The string is duplicated in ``pyproject.toml`` for the build backend; keep
both files in sync. ``cli.main`` prints this on ``--version``.
"""
from __future__ import annotations

__version__ = "0.2.0"

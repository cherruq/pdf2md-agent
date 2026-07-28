"""Token-budget public surface.

This module is the backward-compatible facade for the two
implementation modules:

* :mod:`pdf2md_agent.token_estimator` — text and image token estimators.
* :mod:`pdf2md_agent.image_budget` — per-page image downscale planner
  (``plan_for_image``, ``BudgetDecision``).

New code should import from the implementation modules directly; this
shim exists so the historic ``from pdf2md_agent.token_budget import …``
import paths keep working.
"""
from pdf2md_agent.image_budget import (
    BudgetDecision,
    _est_size_at_long_side,
    _open_for_size,
    _tokens_for_size,
    plan_for_image,
)
from pdf2md_agent.token_estimator import (
    PathOrBytes,
    estimate_image_tokens,
    estimate_text_tokens,
)


__all__ = [
    "BudgetDecision",
    "PathOrBytes",
    "_est_size_at_long_side",
    "_open_for_size",
    "_tokens_for_size",
    "estimate_image_tokens",
    "estimate_text_tokens",
    "plan_for_image",
]
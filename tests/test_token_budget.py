"""Tests for the token-budget planner."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
from PIL import Image

from convertpdf.token_budget import (
    BudgetDecision,
    _est_size_at_long_side,
    _tokens_for_size,
    estimate_image_tokens,
    estimate_text_tokens,
    plan_for_image,
)


# --- estimate_text_tokens -------------------------------------------------


def test_estimate_text_tokens_empty_returns_zero() -> None:
    assert estimate_text_tokens("") == 0


def test_estimate_text_tokens_ascii_uses_4_chars_per_token() -> None:
    # 12 ASCII chars -> ceil(12/4) = 3 tokens
    assert estimate_text_tokens("hello world!") == 3


def test_estimate_text_tokens_cjk_uses_3_chars_per_token() -> None:
    # 6 CJK chars -> ceil(6/3) = 2 tokens
    assert estimate_text_tokens("你好世界中") == 2


def test_estimate_text_tokens_mixed_sums_each_side() -> None:
    text = "你好 world"  # 2 CJK + 1 space + 5 ASCII
    # 2/3 -> 1 token, 6/4 -> 2 tokens = 3 total
    assert estimate_text_tokens(text) == 3


# --- estimate_image_tokens ------------------------------------------------


def _make_solid_png(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    Image.new("RGB", size, color=(128, 128, 128)).save(path, "PNG")


def test_estimate_image_tokens_missing_file_returns_zero(tmp_path: Path) -> None:
    missing = tmp_path / "nope.png"
    assert estimate_image_tokens(missing) == 0


def test_estimate_image_tokens_png_path(tmp_path: Path) -> None:
    png = tmp_path / "small.png"
    _make_solid_png(png, (200, 200))
    size = png.stat().st_size
    expected_b64 = ((size + 2) // 3) * 4
    expected_tokens = max(1, math.ceil(expected_b64 / 3.5))
    assert estimate_image_tokens(png) == expected_tokens


def test_estimate_image_tokens_bytes_input() -> None:
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 1024
    size = len(raw)
    expected_b64 = ((size + 2) // 3) * 4
    expected_tokens = max(1, math.ceil(expected_b64 / 3.5))
    assert estimate_image_tokens(raw) == expected_tokens


def test_estimate_image_tokens_bytearray_input() -> None:
    raw = bytearray(b"\xff" * 512)
    assert estimate_image_tokens(raw) >= 1


def test_estimate_image_tokens_unsupported_type_raises() -> None:
    with pytest.raises(TypeError):
        estimate_image_tokens(12345)  # type: ignore[arg-type]


# --- _tokens_for_size / _est_size_at_long_side ----------------------------


def test_tokens_for_size_zero_floored_to_one() -> None:
    assert _tokens_for_size(0) == 1


def test_tokens_for_size_single_byte_rounds_to_two() -> None:
    # 1 byte -> 4 base64 chars -> ceil(4/3.5) = 2
    assert _tokens_for_size(1) == 2


def test_est_size_at_long_side_scales_quadratically() -> None:
    # Halving long-side should produce ~25% of original bytes.
    est = _est_size_at_long_side(orig_bytes=10000, orig_long_side=1000, target_long_side=500)
    assert 2400 <= est <= 2600  # 10000 * 0.25 = 2500


def test_est_size_at_long_side_floor_at_1024() -> None:
    est = _est_size_at_long_side(orig_bytes=10000, orig_long_side=1000, target_long_side=10)
    assert est >= 1024


# --- plan_for_image -------------------------------------------------------


def test_plan_for_image_fits_at_original(tmp_path: Path) -> None:
    """Tiny image that easily fits the budget returns fits=True at target_long_side."""
    tiny = tmp_path / "tiny.png"
    _make_solid_png(tiny, (100, 100))
    decision = plan_for_image(
        ctx_limit=2013,
        persona_tokens=200,
        fixed_text_tokens=100,
        image_path=tiny,
        target_long_side=1536,
        min_long_side=768,
    )
    assert isinstance(decision, BudgetDecision)
    assert decision.fits is True
    assert decision.needed_long_side == 1536


def test_plan_for_image_downscaling_returns_smaller_long_side(tmp_path: Path) -> None:
    """Large page that exceeds budget is downscaled to a smaller long_side."""
    big = tmp_path / "big.png"
    _make_solid_png(big, (2400, 1800))
    decision = plan_for_image(
        ctx_limit=2013,
        persona_tokens=232,
        fixed_text_tokens=300,
        image_path=big,
        target_long_side=1536,
        min_long_side=768,
        safety=0.85,
    )
    assert decision.needed_long_side <= 1536
    assert decision.needed_long_side >= 768
    assert decision.total <= decision.limit


def test_plan_for_image_original_smaller_than_min_floor(tmp_path: Path) -> None:
    """Image already smaller than min_long_side still reports fits=False if it
    can't fit the budget — the planner returns the floor size with fits=False."""
    tiny = tmp_path / "small.png"
    _make_solid_png(tiny, (200, 200))
    decision = plan_for_image(
        ctx_limit=2013,
        persona_tokens=2000,  # blow the budget
        fixed_text_tokens=0,
        image_path=tiny,
        target_long_side=1536,
        min_long_side=768,
    )
    assert decision.fits is False


def test_plan_for_image_invalid_safety_raises() -> None:
    with pytest.raises(ValueError):
        plan_for_image(
            ctx_limit=2013,
            persona_tokens=100,
            fixed_text_tokens=0,
            image_path=Path("/nonexistent.png"),
            safety=1.5,
        )


def test_plan_for_image_target_below_min_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plan_for_image(
            ctx_limit=2013,
            persona_tokens=100,
            fixed_text_tokens=0,
            image_path=tmp_path / "x.png",
            target_long_side=500,
            min_long_side=768,
        )


def test_plan_for_image_target_zero_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plan_for_image(
            ctx_limit=2013,
            persona_tokens=100,
            fixed_text_tokens=0,
            image_path=tmp_path / "x.png",
            target_long_side=0,
            min_long_side=0,
        )


def test_plan_for_image_min_zero_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plan_for_image(
            ctx_limit=2013,
            persona_tokens=100,
            fixed_text_tokens=0,
            image_path=tmp_path / "x.png",
            target_long_side=768,
            min_long_side=0,
        )
"""Tests for convertpdf.pages."""
from __future__ import annotations

import argparse
import pytest

from convertpdf.pages import parse_page_spec, resolve_pages


# --- parse_page_spec ---------------------------------------------------------

def test_parse_single_page() -> None:
    assert parse_page_spec("3") == [3]


def test_parse_simple_range() -> None:
    assert parse_page_spec("1-5") == [1, 2, 3, 4, 5]


def test_parse_range_and_list() -> None:
    assert parse_page_spec("1-5,8,11-13") == [1, 2, 3, 4, 5, 8, 11, 12, 13]


def test_parse_overlapping_ranges_dedupe() -> None:
    assert parse_page_spec("1-5,1-3") == [1, 2, 3, 4, 5]


def test_parse_tolerates_whitespace() -> None:
    assert parse_page_spec(" 1 - 5 , 8 ") == [1, 2, 3, 4, 5, 8]


@pytest.mark.parametrize(
    "bad",
    ["", "abc", "0", "-3", "5-3", "3-", "-5", "1-5,abc", "1.5", "1,,3"],
)
def test_parse_rejects_bad_input(bad: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_page_spec(bad)
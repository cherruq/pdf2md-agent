"""Tests for pdf2md_agent.pages."""
from __future__ import annotations

import argparse
import pytest

from pdf2md_agent.pages import parse_page_spec, resolve_pages


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


def test_parse_rejects_oversized_range() -> None:
    """Rejects ``1-N`` ranges whose span exceeds the DoS guard.

    Guards against a malicious or mistyped ``--pages 1-99999`` allocating
    a 100k-entry list before ``resolve_pages`` ever sees it.
    """
    with pytest.raises(argparse.ArgumentTypeError, match=r"exceeds"):
        parse_page_spec("1-99999")


# --- resolve_pages -----------------------------------------------------------

def test_resolve_sorts_and_dedupes() -> None:
    assert resolve_pages([3, 1, 2], total=10) == [1, 2, 3]
    assert resolve_pages([5, 5, 5], total=10) == [5]
    assert resolve_pages([7, 1, 7, 3, 1], total=10) == [1, 3, 7]


def test_resolve_passes_when_in_range() -> None:
    assert resolve_pages([1, 2, 3], total=3) == [1, 2, 3]
    assert resolve_pages([1], total=1) == [1]


def test_resolve_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"page 99 out of range \(PDF has 10 pages\)"):
        resolve_pages([99], total=10)
    with pytest.raises(ValueError, match=r"page 4 out of range \(PDF has 3 pages\)"):
        resolve_pages([1, 2, 3, 4], total=3)


def test_resolve_rejects_zero_total() -> None:
    with pytest.raises(ValueError, match=r"PDF has 0 pages"):
        resolve_pages([1], total=0)
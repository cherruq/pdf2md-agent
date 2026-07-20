"""Tests for pdf2md_agent.cache."""
from __future__ import annotations

import json
from pathlib import Path

from pdf2md_agent.cache import write_meta


def test_write_meta_without_pages(tmp_path: Path) -> None:
    meta = tmp_path / "meta.json"
    write_meta(meta, pdf=tmp_path / "x.pdf", dpi=144, with_summary=True)
    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["pages"] is None
    assert payload["dpi"] == 144
    assert payload["with_summary"] is True


def test_write_meta_with_pages(tmp_path: Path) -> None:
    meta = tmp_path / "meta.json"
    write_meta(
        meta,
        pdf=tmp_path / "x.pdf",
        dpi=144,
        with_summary=False,
        pages=[1, 2, 5],
    )
    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["pages"] == [1, 2, 5]

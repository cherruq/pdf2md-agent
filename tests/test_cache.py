"""Tests for pdf2md_agent.cache."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdf2md_agent.cache import (
    CacheCorruptedError,
    atomic_write_text,
    read_summary,
    write_meta,
)


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


def test_read_summary_corrupt_raises_and_backs_up(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CacheCorruptedError):
        read_summary(path)
    siblings = [p for p in tmp_path.iterdir() if p.name.startswith("summary.json.corrupt-")]
    assert siblings, "expected a .corrupt-<ts> backup file"


def test_atomic_write_text_leaves_original_on_mid_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "meta.json"
    original_content = '{"original": true}'
    target.write_text(original_content, encoding="utf-8")

    real_write = __import__("os").write

    def crash_after_open(fd, data, *args, **kwargs):  # pragma: no cover
        if len(data) > 0 and data == b"new payload":
            raise OSError("simulated mid-write crash")
        return real_write(fd, data, *args, **kwargs)

    monkeypatch.setattr("os.write", crash_after_open)

    with pytest.raises(OSError, match="simulated"):
        atomic_write_text(target, "new payload")

    assert target.read_text(encoding="utf-8") == original_content


def test_atomic_write_text_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    atomic_write_text(target, "hello world 中文")
    assert target.read_text(encoding="utf-8") == "hello world 中文"

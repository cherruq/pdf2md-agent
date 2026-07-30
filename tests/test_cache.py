"""Tests for pdf2md_agent.cache."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import pdf2md_agent.cache as cache
from pdf2md_agent.cache import (
    atomic_write_text,
    write_meta,
)


def test_write_meta(tmp_path: Path) -> None:
    meta = tmp_path / "meta.json"
    write_meta(
        meta,
        pdf=tmp_path / "x.pdf",
    )
    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["pdf"] == str(tmp_path / "x.pdf")


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


def test_write_meta_canonicalizes_relative_pdf_to_realpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta = tmp_path / "meta.json"
    workdir = tmp_path / "sub"
    workdir.mkdir()
    real_input = tmp_path / "input.pdf"
    monkeypatch.chdir(workdir)

    write_meta(
        meta,
        pdf=Path("../input.pdf"),
    )

    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["pdf"] == str(real_input.resolve())
    assert Path(payload["pdf"]).is_absolute()


def test_read_meta_missing_returns_none(tmp_path: Path) -> None:
    assert cache.read_meta(tmp_path / "missing.json") is None


def test_read_meta_invalid_json_returns_none(tmp_path: Path) -> None:
    meta = tmp_path / "meta.json"
    meta.write_text("{invalid", encoding="utf-8")
    assert cache.read_meta(meta) is None


def test_read_meta_non_object_returns_none(tmp_path: Path) -> None:
    meta = tmp_path / "meta.json"
    meta.write_text("[]", encoding="utf-8")
    assert cache.read_meta(meta) is None


def test_read_meta_missing_field_returns_none(tmp_path: Path) -> None:
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps({"not_pdf": "/tmp/input.pdf"}),
        encoding="utf-8",
    )
    assert cache.read_meta(meta) is None


def _meta_info() -> cache.MetaInfo:
    return cache.MetaInfo(pdf="/tmp/input.pdf")


def test_check_meta_matches_no_diff_when_identical() -> None:
    assert cache.check_meta_matches(_meta_info(), pdf="/tmp/input.pdf") == []


def test_check_meta_matches_reports_diff_on_pdf_change() -> None:
    reasons = cache.check_meta_matches(_meta_info(), pdf="/tmp/other.pdf")
    assert len(reasons) == 1
    assert "pdf" in reasons[0]

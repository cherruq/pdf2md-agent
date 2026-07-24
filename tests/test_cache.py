"""Tests for pdf2md_agent.cache."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import pdf2md_agent.cache as cache
from pdf2md_agent.cache import (
    CacheCorruptedError,
    atomic_write_text,
    read_summary,
    write_meta,
)


def test_write_meta_without_pages(tmp_path: Path) -> None:
    meta = tmp_path / "meta.json"
    write_meta(
        meta,
        pdf=tmp_path / "x.pdf",
        dpi=144,
        with_summary=True,
        model="MiniMax-M3",
        persona_version="0123456789abcdef",
    )
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
        model="MiniMax-M3",
        persona_version="0123456789abcdef",
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


def test_write_meta_persists_all_six_fields(tmp_path: Path) -> None:
    meta = tmp_path / "meta.json"
    write_meta(
        meta,
        pdf=tmp_path / "input.pdf",
        dpi=200,
        with_summary=False,
        pages=[3, 1],
        model="vision-model",
        persona_version="fedcba9876543210",
    )

    assert json.loads(meta.read_text(encoding="utf-8")) == {
        "pdf": str(tmp_path / "input.pdf"),
        "dpi": 200,
        "with_summary": False,
        "pages": [3, 1],
        "model": "vision-model",
        "persona_version": "fedcba9876543210",
    }


def test_write_meta_canonicalizes_relative_pdf_to_realpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a relative-path input must be canonicalized via
    :meth:`Path.resolve` before serialization. Otherwise a follow-up run
    invoked from a different cwd (or with a different relative spelling of
    the same file) sees the on-disk ``pdf`` field drift away from the
    current run's ``pdf.resolve()`` even though the underlying file is
    identical — producing a false-positive cache-fingerprint rejection.
    """
    meta = tmp_path / "meta.json"
    # ``tmp_path`` is absolute on every supported platform, so chdir into
    # a nested subdir to make any non-resolved input visibly relative.
    workdir = tmp_path / "sub"
    workdir.mkdir()
    real_input = tmp_path / "input.pdf"
    monkeypatch.chdir(workdir)

    write_meta(
        meta,
        pdf=Path("../input.pdf"),
        dpi=144,
        with_summary=True,
        model="MiniMax-M3",
        persona_version="0123456789abcdef",
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
        json.dumps(
            {
                "pdf": "/tmp/input.pdf",
                "dpi": 144,
                "with_summary": True,
                "pages": None,
                "model": "vision-model",
            }
        ),
        encoding="utf-8",
    )
    assert cache.read_meta(meta) is None


def test_read_meta_wrong_pages_shape_returns_none(tmp_path: Path) -> None:
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps(
            {
                "pdf": "/tmp/input.pdf",
                "dpi": 144,
                "with_summary": True,
                "pages": "1,2",
                "model": "vision-model",
                "persona_version": "0123456789abcdef",
            }
        ),
        encoding="utf-8",
    )
    assert cache.read_meta(meta) is None


def _meta_info() -> cache.MetaInfo:
    return cache.MetaInfo(
        pdf="/tmp/input.pdf",
        dpi=144,
        with_summary=True,
        pages=(1, 2, 5),
        model="vision-model",
        persona_version="0123456789abcdef",
    )


def _current_meta_values() -> dict[str, str | int | bool | list[int] | None]:
    return {
        "pdf": "/tmp/input.pdf",
        "dpi": 144,
        "with_summary": True,
        "pages": [1, 2, 5],
        "model": "vision-model",
        "persona_version": "0123456789abcdef",
    }


def test_check_meta_matches_no_diff_when_identical() -> None:
    assert cache.check_meta_matches(_meta_info(), **_current_meta_values()) == []


@pytest.mark.parametrize(
    ("field", "different"),
    [
        ("pdf", "/tmp/other.pdf"),
        ("dpi", 200),
        ("with_summary", False),
        ("pages", [1, 3]),
        ("model", "other-model"),
        ("persona_version", "fedcba9876543210"),
    ],
)
def test_check_meta_matches_reports_each_field(
    field: str,
    different: str | int | bool | list[int],
) -> None:
    current = _current_meta_values()
    current[field] = different

    reasons = cache.check_meta_matches(_meta_info(), **current)

    assert len(reasons) == 1
    assert field in reasons[0]


def test_check_meta_matches_pages_set_order_invariant() -> None:
    current = _current_meta_values()
    current["pages"] = [5, 1, 2]
    assert cache.check_meta_matches(_meta_info(), **current) == []

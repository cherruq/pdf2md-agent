"""Unit tests for ``pdf2md_agent.ctx_probe``.

The probe is purely network I/O — every test mocks ``urllib.request.urlopen``
so no real HTTP traffic is generated. The probe is intentionally tolerant
of provider-specific schema variations (different field names for the
context-window field); the parametrised test exercises each candidate.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from urllib.error import HTTPError, URLError

from pdf2md_agent.ctx_probe import (
    _MAX_CTX_LIMIT,
    probe_ctx_limit,
)


def _make_response(payload: dict[str, Any]) -> MagicMock:
    """Return a mock ``urlopen`` context whose ``.read()`` yields JSON."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    resp.__enter__ = lambda self_: self_
    resp.__exit__ = lambda self_, *args: False
    return resp


def test_probe_returns_int_when_context_window_field_present() -> None:
    payload = {"data": [{"id": "MiniMax-M3", "context_window": 524288}]}
    with patch("pdf2md_agent.ctx_probe.urlopen", return_value=_make_response(payload)):
        result = probe_ctx_limit(
            "https://api.example.com/v1", "key", "MiniMax-M3"
        )
    assert result == 524288


@pytest.mark.parametrize(
    "field",
    [
        "context_window",
        "max_context_tokens",
        "max_input_tokens",
        "context_length",
        "max_tokens",
        "max_sequence_length",
    ],
)
def test_probe_handles_alternate_field_names(field: str) -> None:
    payload = {"data": [{"id": "some-model", field: 65536}]}
    with patch("pdf2md_agent.ctx_probe.urlopen", return_value=_make_response(payload)):
        result = probe_ctx_limit(
            "https://api.example.com/v1", "key", "some-model"
        )
    assert result == 65536


def test_probe_matches_model_id_case_insensitively() -> None:
    payload = {"data": [{"id": "MiniMax-M3", "context_window": 1000}]}
    with patch("pdf2md_agent.ctx_probe.urlopen", return_value=_make_response(payload)):
        result = probe_ctx_limit(
            "https://api.example.com/v1", "key", "MiniMax-M3"
        )
    assert result == 1000


def test_probe_falls_back_to_partial_match() -> None:
    payload = {
        "data": [
            {"id": "MiniMax-M3-2024-12-01", "context_window": 9999},
        ]
    }
    with patch("pdf2md_agent.ctx_probe.urlopen", return_value=_make_response(payload)):
        result = probe_ctx_limit(
            "https://api.example.com/v1", "key", "MiniMax-M3"
        )
    assert result == 9999


def test_probe_clamps_to_1M_when_provider_reports_higher() -> None:
    payload = {"data": [{"id": "MiniMax-M3", "context_window": 5_000_000}]}
    with patch("pdf2md_agent.ctx_probe.urlopen", return_value=_make_response(payload)):
        result = probe_ctx_limit(
            "https://api.example.com/v1", "key", "MiniMax-M3"
        )
    assert result == _MAX_CTX_LIMIT
    assert result == 1_048_576


def test_probe_returns_none_on_404() -> None:
    err = HTTPError("https://x/v1/models", 404, "Not Found", {}, None)
    with patch("pdf2md_agent.ctx_probe.urlopen", side_effect=err):
        result = probe_ctx_limit(
            "https://api.example.com/v1", "key", "MiniMax-M3"
        )
    assert result is None


def test_probe_returns_none_on_500() -> None:
    err = HTTPError("https://x/v1/models", 500, "Server Error", {}, None)
    with patch("pdf2md_agent.ctx_probe.urlopen", side_effect=err):
        result = probe_ctx_limit(
            "https://api.example.com/v1", "key", "MiniMax-M3"
        )
    assert result is None


def test_probe_returns_none_on_url_error() -> None:
    with patch(
        "pdf2md_agent.ctx_probe.urlopen",
        side_effect=URLError("name resolution failed"),
    ):
        result = probe_ctx_limit(
            "https://api.example.com/v1", "key", "MiniMax-M3"
        )
    assert result is None


def test_probe_returns_none_on_timeout() -> None:
    with patch(
        "pdf2md_agent.ctx_probe.urlopen", side_effect=TimeoutError("slow")
    ):
        result = probe_ctx_limit(
            "https://api.example.com/v1", "key", "MiniMax-M3"
        )
    assert result is None


def test_probe_returns_none_on_malformed_json() -> None:
    resp = MagicMock()
    resp.read.return_value = b"{not valid json"
    resp.__enter__ = lambda self_: self_
    resp.__exit__ = lambda self_, *args: False
    with patch("pdf2md_agent.ctx_probe.urlopen", return_value=resp):
        result = probe_ctx_limit(
            "https://api.example.com/v1", "key", "MiniMax-M3"
        )
    assert result is None


def test_probe_returns_none_when_model_not_in_list() -> None:
    payload = {"data": [{"id": "completely-different-model", "context_window": 100}]}
    with patch("pdf2md_agent.ctx_probe.urlopen", return_value=_make_response(payload)):
        result = probe_ctx_limit(
            "https://api.example.com/v1", "key", "MiniMax-M3"
        )
    assert result is None


def test_probe_returns_none_when_data_field_missing_context_window() -> None:
    payload = {"data": [{"id": "MiniMax-M3", "owned_by": "x"}]}
    with patch("pdf2md_agent.ctx_probe.urlopen", return_value=_make_response(payload)):
        result = probe_ctx_limit(
            "https://api.example.com/v1", "key", "MiniMax-M3"
        )
    assert result is None


def test_probe_returns_none_on_empty_data() -> None:
    payload = {"data": []}
    with patch("pdf2md_agent.ctx_probe.urlopen", return_value=_make_response(payload)):
        result = probe_ctx_limit(
            "https://api.example.com/v1", "key", "MiniMax-M3"
        )
    assert result is None


def test_probe_strips_trailing_slash_from_base_url() -> None:
    payload = {"data": [{"id": "m", "context_window": 42}]}
    with patch(
        "pdf2md_agent.ctx_probe.urlopen", return_value=_make_response(payload)
    ) as mock_urlopen:
        probe_ctx_limit("https://api.example.com/v1/", "k", "m")
    called_url = mock_urlopen.call_args[0][0].full_url
    # No doubled slash; ``/models`` suffix present exactly once.
    assert "//" not in called_url.replace("https://", "")
    assert called_url.endswith("/models")


@pytest.mark.parametrize("scheme", ["file", "ftp", "javascript"])
def test_probe_rejects_non_http_schemes(scheme: str) -> None:
    """A misconfigured ``OPENAI_BASE_URL`` (e.g. ``file:///etc/passwd``)
    must not result in a local file read or arbitrary protocol dial.
    """
    with patch("pdf2md_agent.ctx_probe.urlopen") as mock_urlopen:
        assert probe_ctx_limit(f"{scheme}://evil.example/models", "k", "m") is None
    mock_urlopen.assert_not_called()
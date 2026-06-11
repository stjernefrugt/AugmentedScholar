"""Unit tests for src/ss_client.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx

from src.ss_client import SemanticScholarClient


def _ok(body: dict[str, object]) -> MagicMock:
    """Return a mock 200 response with *body* as JSON."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = body
    resp.raise_for_status.return_value = None
    return resp


def _err(status: int) -> MagicMock:
    """Return a mock error response that raises on raise_for_status."""
    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"HTTP {status}",
        request=MagicMock(),
        response=MagicMock(),
    )
    return resp


def _make_client(
    responses: list[dict[str, object]],
) -> tuple[SemanticScholarClient, MagicMock]:
    """Build a SemanticScholarClient with a mock httpx.Client.

    Args:
        responses: List of dicts to return as JSON from successive GET calls.

    Returns:
        Tuple of (client, mock_http_client).
    """
    mock_http = MagicMock()
    mock_http.get.side_effect = [_ok(b) for b in responses]
    client = SemanticScholarClient(client=mock_http)
    return client, mock_http


class TestPaginate:
    def test_null_data_field_does_not_crash(self) -> None:
        """SS sometimes returns ``"data": null``; _paginate must not raise."""
        client, _ = _make_client([{"data": None, "total": 0}])
        result = client.get_author_papers("123")
        assert result == []

    def test_missing_data_field_returns_empty(self) -> None:
        client, _ = _make_client([{"total": 0}])
        result = client.get_author_papers("123")
        assert result == []

    def test_single_page_returned(self) -> None:
        paper = {"paperId": "abc", "title": "T", "year": 2021}
        client, _ = _make_client([{"data": [paper], "total": 1}])
        result = client.get_author_papers("123")
        assert len(result) == 1
        assert result[0]["paperId"] == "abc"

    def test_multi_page_aggregated(self) -> None:
        p1 = {"paperId": "p1", "title": "A", "year": 2020}
        p2 = {"paperId": "p2", "title": "B", "year": 2021}
        client, _ = _make_client(
            [
                {"data": [p1], "total": 2, "next": 1},
                {"data": [p2], "total": 2},
            ]
        )
        result = client.get_author_papers("123")
        assert len(result) == 2

    def test_http_error_returns_empty(self) -> None:
        mock_http = MagicMock()
        mock_http.get.side_effect = RuntimeError("network error")
        client = SemanticScholarClient(client=mock_http)
        result = client.get_author_papers("123")
        assert result == []

    def test_references_unwrapped(self) -> None:
        cited = {"paperId": "ref1", "title": "Ref", "year": 2018}
        client, _ = _make_client([{"data": [{"citedPaper": cited}], "total": 1}])
        result = client.get_paper_references("paper-id")
        assert len(result) == 1
        assert result[0]["paperId"] == "ref1"

    def test_citations_unwrapped(self) -> None:
        citing = {"paperId": "cite1", "title": "Citing", "year": 2023}
        client, _ = _make_client([{"data": [{"citingPaper": citing}], "total": 1}])
        result = client.get_paper_citations("paper-id")
        assert len(result) == 1
        assert result[0]["paperId"] == "cite1"


class TestBackoff:
    """Exponential back-off on 429 / 5xx responses."""

    def _no_sleep_client(self, side_effects: list[MagicMock]) -> SemanticScholarClient:
        """Client with backoff_base=0 so tests run without actual sleeping."""
        mock_http = MagicMock()
        mock_http.get.side_effect = side_effects
        return SemanticScholarClient(client=mock_http, backoff_base=0.0)

    def test_429_retried_then_succeeds(self) -> None:
        paper = {"paperId": "p1", "title": "T", "year": 2021}
        client = self._no_sleep_client([_err(429), _ok({"data": [paper], "total": 1})])
        result = client.get_author_papers("123")
        assert result == [paper]

    def test_503_retried_then_succeeds(self) -> None:
        paper = {"paperId": "p2", "title": "T2", "year": 2022}
        client = self._no_sleep_client([_err(503), _ok({"data": [paper], "total": 1})])
        result = client.get_author_papers("123")
        assert result == [paper]

    def test_retry_count_matches_attempts(self) -> None:
        # 2 failures then success — should call get() exactly 3 times
        paper = {"paperId": "px", "title": "Tx", "year": 2020}
        mock_http = MagicMock()
        mock_http.get.side_effect = [
            _err(429),
            _err(429),
            _ok({"data": [paper], "total": 1}),
        ]
        client = SemanticScholarClient(client=mock_http, backoff_base=0.0)
        result = client.get_author_papers("123")
        assert result == [paper]
        assert mock_http.get.call_count == 3

    def test_max_retries_exhausted_returns_empty(self) -> None:
        # All attempts return 429 — should give up and return []
        mock_http = MagicMock()
        mock_http.get.return_value = _err(429)
        client = SemanticScholarClient(
            client=mock_http, max_retries=2, backoff_base=0.0
        )
        result = client.get_author_papers("123")
        assert result == []
        # 1 initial + 2 retries = 3 total calls
        assert mock_http.get.call_count == 3

    def test_non_retryable_404_not_retried(self) -> None:
        mock_http = MagicMock()
        mock_http.get.return_value = _err(404)
        client = SemanticScholarClient(client=mock_http, backoff_base=0.0)
        result = client.get_author_papers("123")
        assert result == []
        assert mock_http.get.call_count == 1  # no retry for 404

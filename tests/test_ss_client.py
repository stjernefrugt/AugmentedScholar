"""Unit tests for src/ss_client.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.ss_client import SemanticScholarClient


def _make_client(responses: list[dict[str, object]]) -> tuple[SemanticScholarClient, MagicMock]:
    """Build a SemanticScholarClient with a mock httpx.Client.

    Args:
        responses: List of dicts to return as JSON from successive GET calls.

    Returns:
        Tuple of (client, mock_http_client).
    """
    mock_http = MagicMock()
    mock_responses = []
    for body in responses:
        resp = MagicMock()
        resp.json.return_value = body
        resp.raise_for_status.return_value = None
        mock_responses.append(resp)
    mock_http.get.side_effect = mock_responses
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

"""Unit tests for src/unpaywall_client.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.unpaywall_client import UnpaywallClient


def _make_client(response_body: dict[str, object]) -> UnpaywallClient:
    """Return an UnpaywallClient with a mock HTTP client."""
    resp = MagicMock()
    resp.json.return_value = response_body
    resp.raise_for_status.return_value = None
    mock_http = MagicMock()
    mock_http.get.return_value = resp
    return UnpaywallClient(email="test@example.com", client=mock_http)


class TestGetPdfUrl:
    def test_returns_url_for_pdf_when_oa(self) -> None:
        client = _make_client(
            {
                "is_oa": True,
                "best_oa_location": {
                    "url_for_pdf": "https://pmc.ncbi.nlm.nih.gov/paper.pdf",
                    "url": "https://pmc.ncbi.nlm.nih.gov/paper",
                },
            }
        )
        assert (
            client.get_pdf_url("10.1234/test")
            == "https://pmc.ncbi.nlm.nih.gov/paper.pdf"
        )

    def test_falls_back_to_url_when_no_url_for_pdf(self) -> None:
        client = _make_client(
            {
                "is_oa": True,
                "best_oa_location": {
                    "url_for_pdf": None,
                    "url": "https://pmc.ncbi.nlm.nih.gov/paper",
                },
            }
        )
        assert (
            client.get_pdf_url("10.1234/test") == "https://pmc.ncbi.nlm.nih.gov/paper"
        )

    def test_returns_none_when_not_oa(self) -> None:
        client = _make_client({"is_oa": False})
        assert client.get_pdf_url("10.1234/test") is None

    def test_returns_none_when_no_best_location(self) -> None:
        client = _make_client({"is_oa": True, "best_oa_location": None})
        assert client.get_pdf_url("10.1234/test") is None

    def test_strips_doi_org_prefix(self) -> None:
        resp = MagicMock()
        resp.json.return_value = {"is_oa": False}
        resp.raise_for_status.return_value = None
        mock_http = MagicMock()
        mock_http.get.return_value = resp
        client = UnpaywallClient(email="test@example.com", client=mock_http)
        client.get_pdf_url("https://doi.org/10.1234/test")
        call_url = mock_http.get.call_args[0][0]
        assert call_url.endswith("/10.1234/test")

    def test_http_error_returns_none(self) -> None:
        mock_http = MagicMock()
        mock_http.get.side_effect = RuntimeError("timeout")
        client = UnpaywallClient(email="test@example.com", client=mock_http)
        assert client.get_pdf_url("10.1234/test") is None

"""Unpaywall API client for resolving DOIs to open-access PDF URLs.

Unpaywall is a free, legal service that indexes OA versions of research papers
from sources such as PubMed Central, institutional repositories, and author
homepages.  An email address is required by Unpaywall's ToS so they can
contact you if your usage is problematic.

API documentation: https://unpaywall.org/products/api
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Any

import httpx

_BASE = "https://api.unpaywall.org/v2"

logger = logging.getLogger(__name__)


class UnpaywallClient:
    """Resolve DOIs to open-access PDF URLs via the Unpaywall REST API.

    Args:
        email: Contact email sent with each request (required by Unpaywall ToS).
        client: Optional pre-configured :class:`httpx.Client`.

    Examples:
        >>> with UnpaywallClient(email="you@example.com") as uw:
        ...     url = uw.get_pdf_url("10.1038/s41586-021-03819-2")
    """

    def __init__(
        self,
        email: str,
        client: httpx.Client | None = None,
    ) -> None:
        self._email = email
        self._owns_client = client is None
        self._client: httpx.Client = client or httpx.Client(timeout=10.0)

    def get_pdf_url(self, doi: str) -> str | None:
        """Return the best open-access PDF URL for *doi*, or ``None``.

        Tries ``url_for_pdf`` (direct PDF link) first, then falls back to
        ``url`` (landing page that may have a download button).

        Args:
            doi: DOI string (with or without ``https://doi.org/`` prefix).

        Returns:
            URL string if an OA version is found, otherwise ``None``.
        """
        clean_doi = doi.removeprefix("https://doi.org/").strip()
        try:
            response = self._client.get(
                f"{_BASE}/{clean_doi}",
                params={"email": self._email},
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        except Exception as exc:
            logger.debug("Unpaywall lookup failed for DOI '%s': %s", doi, exc)
            return None

        if not data.get("is_oa"):
            logger.debug("No OA version found for DOI '%s'", doi)
            return None

        best: dict[str, Any] = data.get("best_oa_location") or {}
        url: str | None = best.get("url_for_pdf") or best.get("url")
        if url:
            logger.debug("Unpaywall found OA URL for '%s': %s", doi, url)
        return url

    def close(self) -> None:
        """Close the underlying HTTP client if owned by this instance."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> UnpaywallClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

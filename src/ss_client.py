"""Semantic Scholar Graph API client for author and citation traversal.

Handles pagination and rate-limiting automatically. All list-returning
methods aggregate across pages so callers receive complete result sets.
"""

from __future__ import annotations

import logging
import random
import time
from types import TracebackType
from typing import Any

import httpx

_BASE = "https://api.semanticscholar.org/graph/v1"
_PAPER_FIELDS = (
    "paperId,title,authors,year,journal,abstract,"
    "citationCount,externalIds,openAccessPdf"
)
_PAGE_SIZE = 100
# HTTP status codes that warrant a retry with exponential back-off
_RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

logger = logging.getLogger(__name__)


class SemanticScholarClient:
    """Thin wrapper around the Semantic Scholar Graph API.

    Args:
        client: Optional pre-configured :class:`httpx.Client`.
        api_key: Semantic Scholar API key.  Raises the unauthenticated
            rate limit of ~1 req/s to ~10 req/s.
        request_delay: Seconds to wait between paginated requests
            (default 0.5 s, safe for unauthenticated access).

    Examples:
        >>> with SemanticScholarClient(api_key="...") as ss:
        ...     papers = ss.get_author_papers("1741101")
    """

    def __init__(
        self,
        client: httpx.Client | None = None,
        api_key: str | None = None,
        request_delay: float = 0.5,
        max_retries: int = 5,
        backoff_base: float = 1.0,
    ) -> None:
        self._delay = request_delay
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        headers: dict[str, str] = {}
        if api_key:
            headers["x-api-key"] = api_key
        self._owns_client = client is None
        self._client: httpx.Client = client or httpx.Client(
            timeout=30.0, headers=headers
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_author(self, name: str) -> list[dict[str, Any]]:
        """Search for an author by display name.

        Args:
            name: Author's full name or partial name.

        Returns:
            List of author dicts containing ``authorId``, ``name``,
            ``paperCount``, and ``citationCount``.  Empty list on error.
        """
        try:
            response = self._request(
                f"{_BASE}/author/search",
                params={
                    "query": name,
                    "fields": "name,affiliations,paperCount,citationCount",
                },
            )
            return list(response.json().get("data", []))
        except Exception as exc:
            logger.error("Author search failed for '%s': %s", name, exc)
            return []

    def search_paper_by_title(self, title: str) -> dict[str, Any] | None:
        """Find the best-matching Semantic Scholar paper for a given title.

        Uses the SS paper search endpoint.  Returns the top result if the
        title overlap is reasonable, or ``None`` if nothing credible is found.

        Args:
            title: Paper title to search for.

        Returns:
            Paper dict (including ``paperId``) for the top match, or ``None``.
        """
        try:
            response = self._request(
                f"{_BASE}/paper/search",
                params={"query": title, "fields": _PAPER_FIELDS, "limit": 1},
            )
            data: list[dict[str, Any]] = response.json().get("data", [])
            return data[0] if data else None
        except Exception as exc:
            logger.warning("SS paper search failed for '%s': %s", title, exc)
            return None

    def get_author_papers(self, author_id: str) -> list[dict[str, Any]]:
        """Fetch all papers attributed to a Semantic Scholar author.

        Args:
            author_id: Semantic Scholar integer author ID string.

        Returns:
            List of paper dicts.  May be empty if the author has no
            indexed papers or if the request fails.
        """
        return self._paginate(
            f"{_BASE}/author/{author_id}/papers",
            {"fields": _PAPER_FIELDS},
        )

    def get_paper_references(self, paper_id: str) -> list[dict[str, Any]]:
        """Fetch all papers referenced (cited) by a given paper.

        Args:
            paper_id: Semantic Scholar paper ID.

        Returns:
            List of cited paper dicts (unwrapped from the ``citedPaper``
            envelope returned by the API).
        """
        raw = self._paginate(
            f"{_BASE}/paper/{paper_id}/references",
            {"fields": _PAPER_FIELDS},
        )
        return [r["citedPaper"] for r in raw if r.get("citedPaper")]

    def get_paper_citations(self, paper_id: str) -> list[dict[str, Any]]:
        """Fetch all papers that cite a given paper.

        Args:
            paper_id: Semantic Scholar paper ID.

        Returns:
            List of citing paper dicts (unwrapped from the ``citingPaper``
            envelope returned by the API).
        """
        raw = self._paginate(
            f"{_BASE}/paper/{paper_id}/citations",
            {"fields": _PAPER_FIELDS},
        )
        return [r["citingPaper"] for r in raw if r.get("citingPaper")]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _request(self, url: str, params: dict[str, Any]) -> httpx.Response:
        """GET *url* with exponential back-off on retryable HTTP errors.

        Retries on :data:`_RETRY_STATUSES` (429, 500–504) up to
        :attr:`_max_retries` times, doubling the wait each attempt
        (capped at 60 s) with ±10 % jitter.

        Args:
            url: Full endpoint URL.
            params: Query parameters forwarded verbatim.

        Returns:
            Successful :class:`httpx.Response`.

        Raises:
            httpx.HTTPStatusError: Non-retryable error status, or all
                retries exhausted on a retryable status.
            httpx.RequestError: Network-level failure (not retried).
        """
        wait = self._backoff_base
        resp: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            resp = self._client.get(url, params=params)
            if resp.status_code not in _RETRY_STATUSES:
                resp.raise_for_status()
                return resp
            if attempt < self._max_retries:
                jitter = random.uniform(-wait * 0.1, wait * 0.1)
                delay = min(wait + jitter, 60.0)
                logger.warning(
                    "SS HTTP %d on %s; retrying in %.1fs (attempt %d/%d)",
                    resp.status_code,
                    url,
                    delay,
                    attempt + 1,
                    self._max_retries,
                )
                time.sleep(delay)
                wait = min(wait * 2, 60.0)
        assert resp is not None  # loop always executes ≥ once
        resp.raise_for_status()
        return resp  # unreachable: raise_for_status() raises for 4xx/5xx

    def _paginate(self, url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Consume all pages from a paginated Semantic Scholar endpoint.

        The Semantic Scholar API signals the next page via a ``next``
        key in the response body.  Iteration stops when ``next`` is
        absent or the returned ``data`` list is empty.

        Args:
            url: Full endpoint URL.
            params: Query parameters (``offset`` and ``limit`` are
                appended automatically).

        Returns:
            Aggregated list of items across all pages.
        """
        all_items: list[dict[str, Any]] = []
        offset = 0

        while True:
            try:
                response = self._request(
                    url,
                    params={**params, "limit": _PAGE_SIZE, "offset": offset},
                )
                body: dict[str, Any] = response.json()
            except Exception as exc:
                logger.error("SS API request to '%s' failed: %s", url, exc)
                break

            items: list[dict[str, Any]] = body.get("data") or []
            all_items.extend(items)

            next_offset: int | None = body.get("next")
            if next_offset is None or not items:
                break

            offset = next_offset
            time.sleep(self._delay)

        return all_items

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the HTTP client if it is owned by this instance."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> SemanticScholarClient:
        """Return self for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the HTTP client on context exit."""
        self.close()

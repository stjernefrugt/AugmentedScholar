"""Google Scholar client for fetching an author's complete publication list.

Uses the ``scholarly`` library, which scrapes Google Scholar.  There is no
official API, so Google will rate-limit aggressive requests.  For large
profiles (>100 papers) or repeated runs, configure a proxy via
``scholarly.ProxyGenerator`` before constructing this client.

Rate-limit guidance:
    - Unproxied: safe for one-off fetches of a personal profile (~50–200 pubs).
    - With ScraperAPI/TorFree: suitable for scheduled/repeated runs.
    - ``fill_publications=True`` makes one extra request per paper; avoid for
      large profiles unless you need abstracts.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from .models import Author, PaperMetadata

logger = logging.getLogger(__name__)

# scholarly is an optional dependency — imported lazily inside methods so the
# rest of the package remains importable even when it is not installed.
_SCHOLARLY_MISSING = (
    "scholarly is required for Google Scholar support.\n"
    "Install it with:  uv pip install scholarly"
)


def parse_profile_url(url_or_id: str) -> str:
    """Extract the Google Scholar author ID from a profile URL or return as-is.

    Accepts both a bare ID (``'AbCdEfGhIjK'``) and a full profile URL such as::

        https://scholar.google.com/citations?user=AbCdEfGhIjK&hl=en

    Args:
        url_or_id: Google Scholar author ID string or full profile URL.

    Returns:
        The bare author ID string.

    Raises:
        ValueError: If a URL is provided but contains no ``user=`` parameter.
    """
    if not url_or_id.startswith("http"):
        return url_or_id.strip()

    parsed = urlparse(url_or_id)
    qs = parse_qs(parsed.query)
    user_list = qs.get("user", [])
    if not user_list:
        raise ValueError(
            f"Cannot extract author ID from URL '{url_or_id}'. "
            "Expected a 'user=' query parameter."
        )
    return user_list[0]


def _parse_gs_publication(pub: dict[str, Any]) -> PaperMetadata | None:
    """Convert a scholarly publication dict to a :class:`~src.models.PaperMetadata`.

    Args:
        pub: A publication dict as returned by ``scholarly.fill(pub)`` or
            from the author's ``publications`` list.

    Returns:
        :class:`~src.models.PaperMetadata` if the required fields are present,
        ``None`` otherwise.
    """
    bib: dict[str, Any] = pub.get("bib") or {}

    title: str | None = bib.get("title") or pub.get("title")
    year_raw: str | None = bib.get("pub_year") or bib.get("year")

    if not title or not year_raw:
        return None

    try:
        year = int(year_raw)
    except (ValueError, TypeError):
        return None

    # Author string is "Firstname Lastname and Firstname2 Lastname2" or
    # comma-separated, depending on how scholarly retrieved the record.
    author_str: str = bib.get("author", "")
    raw_names = re.split(r"\s+and\s+|,\s*", author_str)
    authors = [Author(name=n.strip()) for n in raw_names if n.strip()]

    journal: str = (
        bib.get("journal")
        or bib.get("conference")
        or bib.get("booktitle")
        or "Unknown Journal"
    )

    # eprint_url is often a direct PDF link (arXiv, bioRxiv, institutional)
    pdf_url: str | None = pub.get("eprint_url") or None

    return PaperMetadata(
        title=title,
        authors=authors,
        journal=journal,
        year=year,
        citation_count=int(pub.get("num_citations") or 0),
        pdf_url=pdf_url,
    )


_BLOCKED_HINTS = (
    "'NoneType' object has no attribute 'get'",
    "canonical",
    "Login",
    "accounts.google.com",
)

_BLOCK_ADVICE = (
    "Google Scholar blocked the request (redirected to login page).\n"
    "Options:\n"
    "  1. Re-run with --scholar-proxy free   "
    "(rotates through free public proxies)\n"
    "  2. Re-run with --scholar-proxy http://HOST:PORT   "
    "(your own HTTP proxy / institute proxy)\n"
    "  3. Connect via institute VPN and retry without a proxy\n"
    "  4. Wait 15–30 minutes — Google's IP blocks are temporary"
)


class GoogleScholarClient:
    """Fetches an author's complete publication list from Google Scholar.

    Args:
        fill_publications: If ``True``, fetches full detail for every paper
            (one extra HTTP request each).  Gives abstracts and sometimes DOIs
            but is significantly slower.  Default ``False``.
        proxy: Optional proxy configuration passed to ``scholarly``.
            Use ``"free"`` to rotate through free public proxies, or an
            explicit URL such as ``"http://host:port"`` for a fixed proxy.

    Examples:
        >>> client = GoogleScholarClient(proxy="free")
        >>> papers = client.get_author_papers(
        ...     "https://scholar.google.com/citations?user=AbCdEfGhIjK"
        ... )
    """

    def __init__(
        self,
        fill_publications: bool = False,
        proxy: str | None = None,
    ) -> None:
        self.fill_publications = fill_publications
        self.proxy = proxy

    def _configure_proxy(self) -> None:
        """Apply proxy settings to the scholarly singleton if configured."""
        if not self.proxy:
            return
        try:
            from scholarly import ProxyGenerator
            from scholarly import scholarly as _scholarly

            pg = ProxyGenerator()
            if self.proxy == "free":
                logger.info("scholarly: using FreeProxies rotation")
                pg.FreeProxies()
            else:
                logger.info("scholarly: using proxy %s", self.proxy)
                pg.SingleProxy(http=self.proxy, https=self.proxy)
            _scholarly.use_proxy(pg)
        except Exception as exc:
            logger.warning("Could not configure scholarly proxy: %s", exc)

    def get_author_papers(self, url_or_id: str) -> list[PaperMetadata]:
        """Fetch all publications for a Google Scholar author.

        Args:
            url_or_id: Google Scholar author ID or full profile URL.

        Returns:
            List of :class:`~src.models.PaperMetadata` objects, one per
            parsed publication.  Unparseable entries are silently skipped.

        Raises:
            ImportError: If the ``scholarly`` package is not installed.
            RuntimeError: If the Google Scholar profile cannot be retrieved
                (rate-limited, ID not found, network error, etc.).
        """
        try:
            from scholarly import scholarly as _scholarly
        except ImportError as exc:
            raise ImportError(_SCHOLARLY_MISSING) from exc

        self._configure_proxy()

        author_id = parse_profile_url(url_or_id)
        logger.info("Fetching Google Scholar profile: %s", author_id)

        try:
            author = _scholarly.search_author_id(author_id)
            author = _scholarly.fill(author, sections=["publications"])
        except Exception as exc:
            exc_str = str(exc)
            if any(hint in exc_str for hint in _BLOCKED_HINTS):
                raise RuntimeError(_BLOCK_ADVICE) from exc
            raise RuntimeError(
                f"Failed to fetch Google Scholar profile '{author_id}': {exc}"
            ) from exc

        publications: list[dict[str, Any]] = author.get("publications", [])
        logger.info(
            "Found %d publications on Google Scholar for %s",
            len(publications),
            author_id,
        )

        papers: list[PaperMetadata] = []
        for pub in publications:
            if self.fill_publications:
                try:
                    pub = _scholarly.fill(pub)
                except Exception as exc:
                    title = (pub.get("bib") or {}).get("title", "?")
                    logger.warning("Could not fill publication '%s': %s", title, exc)

            meta = _parse_gs_publication(pub)
            if meta:
                papers.append(meta)
            else:
                title = (pub.get("bib") or {}).get("title", "?")
                logger.debug("Skipped unparseable publication: %s", title)

        logger.info(
            "Parsed %d/%d publications from Google Scholar",
            len(papers),
            len(publications),
        )
        return papers

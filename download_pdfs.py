#!/usr/bin/env python3
"""Playwright-assisted PDF batch downloader for paywalled papers.

Reads the missing-PDFs manifest produced by ``run_expansion.py`` and opens a
browser window so you can authenticate with your institute's proxy or SSO once.
The script then navigates to each paper's DOI URL and attempts to auto-click
common journal PDF download buttons.  When auto-download fails or times out,
it pauses so you can download manually, then advances to the next paper.

Prerequisites::

    uv pip install playwright
    playwright install chromium

Usage::

    python download_pdfs.py --manifest ./library/missing_pdfs.json

    # Skip the browser login step if already on institute VPN:
    python download_pdfs.py --manifest ./library/missing_pdfs.json --no-login-pause

    # Headless mode (auto-download only, no manual fallback):
    python download_pdfs.py --manifest ./library/missing_pdfs.json --headless
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Journal-specific PDF download selectors (CSS, evaluated in order)
# ---------------------------------------------------------------------------

# Each entry is tried in sequence; the first matching visible element is clicked.
_DOWNLOAD_SELECTORS = [
    # Generic PDF / download buttons
    "a[href$='.pdf']",
    "a[data-article-url$='.pdf']",
    # Nature / Springer
    "a.c-pdf-download__link",
    "a[data-track-action='download pdf']",
    # Science / AAAS
    "a.article-dl-pdf-link",
    # ACS Publications
    "a[title='PDF']",
    # Wiley
    "a.article-html-pdfLink",
    "a.pdf-download",
    # Elsevier / ScienceDirect
    "a.pdf-download-btn-link",
    "button.download-pdf-btn",
    # APS (Physical Review)
    "a[data-file-type='pdf']",
    # IOP Publishing
    "a.btn-pdf",
    # RSC
    "a[data-gatype='DownloadPDF']",
    # Fallback: any link whose visible text contains "PDF"
    "a:has-text('PDF')",
    "a:has-text('Download PDF')",
]

# How long to wait for an auto-detected download to start (seconds)
_AUTO_DOWNLOAD_TIMEOUT = 15
# How long to give the user for a manual download before skipping (seconds)
_MANUAL_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_manifest(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [e for e in data if e.get("doi_url")]


def _move_download(tmp_dir: Path, dest: Path) -> bool:
    """Move the first PDF found in *tmp_dir* to *dest*.  Returns True on success."""
    pdfs = list(tmp_dir.glob("*.pdf"))
    if not pdfs:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(pdfs[0]), str(dest))
    return True


# ---------------------------------------------------------------------------
# Main download loop
# ---------------------------------------------------------------------------


def _run(
    manifest: list[dict[str, object]],
    headless: bool,
    no_login_pause: bool,
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "playwright is required.\n"
            "Install with:  uv pip install playwright\n"
            "Then:          playwright install chromium"
        ) from None

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        if not no_login_pause and not headless:
            page.goto("https://scholar.google.com")
            input(
                "\n"
                "  Browser is open.\n"
                "  1. Navigate to your institutional proxy or VPN login page.\n"
                "  2. Authenticate so the browser has access to paywalled content.\n"
                "  3. Press ENTER here when ready...\n"
            )

        total = len(manifest)
        downloaded = 0
        skipped = 0

        for i, entry in enumerate(manifest, 1):
            doi_url: str = str(entry["doi_url"])
            expected: Path = Path(str(entry["expected_pdf_path"]))
            title: str = str(entry.get("title", doi_url))

            if expected.exists():
                logger.info("[%d/%d] Already exists, skipping: %s", i, total, title)
                skipped += 1
                continue

            logger.info("[%d/%d] %s", i, total, title)
            logger.info("  → %s", doi_url)

            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                succeeded = False

                # Attempt auto-download via journal selectors
                try:
                    with page.expect_download(
                        timeout=_AUTO_DOWNLOAD_TIMEOUT * 1000
                    ) as dl_info:
                        page.goto(
                            doi_url, wait_until="domcontentloaded", timeout=30_000
                        )
                        for selector in _DOWNLOAD_SELECTORS:
                            try:
                                locator = page.locator(selector).first
                                if locator.is_visible(timeout=500):
                                    locator.click()
                                    break
                            except Exception:
                                continue
                    download = dl_info.value
                    download_path = tmp_dir / (expected.name)
                    download.save_as(str(download_path))
                    if _move_download(tmp_dir, expected):
                        logger.info("  ✓ Auto-downloaded")
                        succeeded = True
                except Exception:
                    pass

                if not succeeded and not headless:
                    # Manual fallback
                    logger.warning(
                        "  Auto-download failed. "
                        "Manually download the PDF and save it to:\n"
                        "  %s\n"
                        "  You have %d seconds before the script advances.",
                        expected,
                        _MANUAL_TIMEOUT,
                    )
                    deadline = time.monotonic() + _MANUAL_TIMEOUT
                    while time.monotonic() < deadline:
                        time.sleep(2)
                        if expected.exists():
                            logger.info("  ✓ Manual download detected")
                            succeeded = True
                            break
                    if not succeeded:
                        logger.warning("  ✗ Timed out — skipping")

            if succeeded:
                downloaded += 1
            else:
                skipped += 1

        context.close()
        browser.close()

    print(f"\nDone. Downloaded: {downloaded}/{total}  Skipped/failed: {skipped}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run the download loop."""
    parser = argparse.ArgumentParser(
        description="Playwright-assisted PDF downloader for paywalled papers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--manifest",
        required=True,
        metavar="FILE",
        help="Path to missing_pdfs.json produced by run_expansion.py",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (auto-download only, no manual fallback)",
    )
    parser.add_argument(
        "--no-login-pause",
        action="store_true",
        help="Skip the initial authentication pause (e.g. when on institute VPN)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        parser.error(f"Manifest file not found: {manifest_path}")

    manifest = _load_manifest(manifest_path)
    if not manifest:
        print("No entries with DOI URLs found in manifest — nothing to download.")
        return

    logger.info("Loaded %d entries from %s", len(manifest), manifest_path)
    _run(manifest, headless=args.headless, no_login_pause=args.no_login_pause)


if __name__ == "__main__":
    main()

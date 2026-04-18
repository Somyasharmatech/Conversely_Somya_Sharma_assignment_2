"""Fetches and parses web pages from URLs, returning clean text."""

from typing import Optional

import requests
from bs4 import BeautifulSoup

from utils.logger import get_logger

logger = get_logger(__name__)

# Tags whose content should be stripped entirely (not just the tag)
_NOISE_TAGS = [
    "script", "style", "nav", "footer", "header",
    "aside", "form", "noscript", "iframe", "svg",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

_TIMEOUT_SECONDS = 15


def ingest_url(url: str) -> Optional[dict]:
    """
    Fetches a URL and extracts the visible text content.

    Returns a dict with:
      - source: the URL
      - source_type: "url"
      - raw_text: cleaned visible text

    Returns None on any failure (network error, bad status, parse error).
    """
    try:
        response = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("Timeout fetching URL: %s", url)
        return None
    except requests.exceptions.ConnectionError as exc:
        logger.error("Connection error fetching URL %s: %s", url, exc)
        return None
    except requests.exceptions.HTTPError as exc:
        logger.error("HTTP error %s for URL %s: %s", exc.response.status_code, url, exc)
        return None
    except requests.exceptions.RequestException as exc:
        logger.error("Request failed for URL %s: %s", url, exc)
        return None

    try:
        soup = BeautifulSoup(response.content, "lxml")

        # Remove noisy tags completely
        for tag in soup(name=_NOISE_TAGS):
            tag.decompose()

        # Extract visible text
        text = soup.get_text(separator="\n", strip=True)
        logger.info("Ingested URL: %s (%d chars)", url, len(text))
        return {"source": url, "source_type": "url", "raw_text": text}
    except Exception as exc:
        logger.error("Failed to parse URL %s: %s", url, exc)
        return None


def ingest_urls_from_file(urls_file: str) -> list[dict]:
    """
    Reads a file containing one URL per line and ingests each URL.
    Skips blank lines and lines starting with '#'.
    Returns a list of successful result dicts.
    """
    results = []
    try:
        with open(urls_file, "r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except Exception as exc:
        logger.error("Could not read URL list file %s: %s", urls_file, exc)
        return results

    logger.info("Found %d URLs in %s", len(lines), urls_file)
    for url in lines:
        result = ingest_url(url)
        if result:
            results.append(result)
        else:
            logger.warning("Skipping failed URL: %s", url)

    return results

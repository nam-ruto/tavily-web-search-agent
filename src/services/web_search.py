from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient

from models import SearchResult, Document

logger = logging.getLogger(__name__)


_TAVILY_API_KEY_ENV = "TAVILY_API_KEY"


def _get_tavily_client() -> TavilyClient:
    api_key = os.getenv(_TAVILY_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Tavily API key not set. Please export environment variable '{_TAVILY_API_KEY_ENV}'."
        )
    return TavilyClient(api_key=api_key)


_tavily_client: TavilyClient | None = None


def _client() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = _get_tavily_client()
    return _tavily_client


def search_web(query: str, max_results: int = 6) -> List[SearchResult]:
    """
    Use Tavily to search the web and return structured search results.
    """
    logger.info("Searching web with Tavily: %r (max_results=%d)", query, max_results)
    response = _client().search(query=query, max_results=max_results)
    raw_results = response.get("results", []) or []

    results: List[SearchResult] = []
    for item in raw_results[:max_results]:
        title = item.get("title") or ""
        url = item.get("url") or ""
        if not url:
            continue

        snippet = item.get("snippet") or item.get("content") or None
        score = item.get("score")

        published_date_raw = item.get("published_date")
        published_date: datetime | None = None
        if isinstance(published_date_raw, str):
            try:
                published_date = datetime.fromisoformat(published_date_raw.replace("Z", "+00:00"))
            except Exception:
                published_date = None

        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                score=score,
                published_date=published_date,
            )
        )

    logger.info("Tavily returned %d search results", len(results))
    return results


def _extract_readable_text(html: str) -> str:
    """
    Extract readable text from raw HTML, dropping script/style/nav/footer boilerplate.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text = " ".join(soup.stripped_strings)
    # Normalize whitespace
    return " ".join(text.split())


def fetch_documents(
    results: List[SearchResult],
    url_cache: Dict[str, Document],
    timeout: float = 10.0,
) -> List[Document]:
    """
    Fetch and extract documents for the given search results, with simple in-memory caching.

    Any URL already present in url_cache will not be fetched again.
    """
    documents: List[Document] = []

    for res in results:
        if res.url in url_cache:
            logger.debug("Using cached document for URL: %s", res.url)
            documents.append(url_cache[res.url])
            continue

        try:
            logger.info("Fetching URL: %s", res.url)
            resp = requests.get(
                res.url,
                timeout=timeout,
                headers={"User-Agent": "tavily-rag-demo/0.1 (+https://tavily.com)"},
            )
            if resp.status_code != 200 or not resp.text:
                logger.warning(
                    "Non-200 or empty response for URL %s (status %s)",
                    res.url,
                    resp.status_code,
                )
                continue

            text = _extract_readable_text(resp.text)
            if not text:
                logger.warning("No readable text extracted for URL %s", res.url)
                continue

            word_count = len(text.split())
            doc = Document(
                url=res.url,
                title=res.title or res.url,
                text=text,
                extracted_at=datetime.utcnow(),
                word_count=word_count,
            )
            url_cache[res.url] = doc
            documents.append(doc)
            logger.info(
                "Fetched and extracted document from %s (words=%d)", res.url, word_count
            )
        except Exception as exc:
            logger.warning("Error fetching URL %s: %s", res.url, exc)

    logger.info("Total documents fetched this round: %d", len(documents))
    return documents

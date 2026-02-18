"""
MRAgent — Google Search Provider
Web search via Google Custom Search JSON API.

Created: 2026-02-18
"""

import time
import requests

from providers.base import SearchProvider
from config.settings import GOOGLE_SEARCH_API_KEY, GOOGLE_SEARCH_CSE_ID
from utils.logger import get_logger

logger = get_logger("providers.google_search")

GOOGLE_SEARCH_ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"


class GoogleSearchProvider(SearchProvider):
    """
    Google Custom Search API provider.
    Requires API Key and Search Engine ID (CSE ID).
    Standard tier: 100 queries/day free.
    """

    def __init__(self, rate_limit_rpm: int = 10):
        super().__init__(name="google_search", rate_limit_rpm=rate_limit_rpm)
        if GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CSE_ID:
            self.logger.info("Google Search provider initialized")
        else:
            self.logger.warning("Google Search credentials not set — search disabled")

    def search(self, query: str, count: int = 5) -> list[dict]:
        """
        Search the web using Google Custom Search API.

        Args:
            query: Search query string
            count: Number of results (1-10)

        Returns:
            List of {"title": str, "url": str, "description": str}
        """
        if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_CSE_ID:
            raise ValueError("GOOGLE_SEARCH_API_KEY or GOOGLE_SEARCH_CSE_ID not set")

        # LLMs sometimes pass count as a string
        count = int(count)

        self.logger.info(f"Search: '{query}' (count={count})")
        start_time = time.time()

        def _make_request():
            resp = requests.get(
                GOOGLE_SEARCH_ENDPOINT,
                params={
                    "key": GOOGLE_SEARCH_API_KEY,
                    "cx": GOOGLE_SEARCH_CSE_ID,
                    "q": query,
                    "num": min(count, 10),
                    "safe": "active",
                },
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()

        try:
            data = self._retry_call(_make_request)
            duration_ms = (time.time() - start_time) * 1000

            # Parse results
            results = []
            items = data.get("items", [])
            for item in items[:count]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "description": item.get("snippet", ""),
                    # Google API doesn't always provide date/age in a standard field easily,
                    # mostly hidden in pagemap or snippet. We leave it empty for now.
                    "age": "",
                })

            self._track_call("web/search", "", duration_ms, status="ok")
            self.logger.info(f"Search returned {len(results)} results ({duration_ms:.0f}ms)")

            return results

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._track_call("web/search", "", duration_ms, status=f"error: {e}")
            raise

    def search_formatted(self, query: str, count: int = 5) -> str:
        """
        Search and return results as a formatted string for LLM consumption.

        Returns:
            Markdown-formatted search results string
        """
        results = self.search(query, count)
        if not results:
            return f"No results found for: {query}"

        lines = [f"## Google Search Results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"**{i}. [{r['title']}]({r['url']})**")
            if r["description"]:
                lines.append(f"   {r['description']}")
            lines.append("")

        return "\n".join(lines)

    @property
    def available(self) -> bool:
        return bool(GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CSE_ID)

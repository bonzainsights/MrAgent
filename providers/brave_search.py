"""
MRAgent — Brave Search Provider
Web search via Brave Search REST API.

Created: 2026-02-15
"""

import time
import requests

from providers.base import SearchProvider
from config.settings import BRAVE_SEARCH_API_KEY
from utils.sanitizer import sanitize_search_snippet
from utils.logger import get_logger

logger = get_logger("providers.brave_search")

BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


class BraveSearchProvider(SearchProvider):
    """
    Brave Search API provider.
    Free tier: 2,000 queries/month.
    """

    def __init__(self, rate_limit_rpm: int = 10):
        super().__init__(name="brave_search", rate_limit_rpm=rate_limit_rpm)
        if BRAVE_SEARCH_API_KEY:
            self.logger.info("Brave Search provider initialized")
        else:
            self.logger.warning("Brave Search API key not set — search disabled")

    def search(self, query: str, count: int = 5) -> list[dict]:
        """
        Search the web using Brave Search API.

        Args:
            query: Search query string
            count: Number of results (1-20)

        Returns:
            List of {"title": str, "url": str, "description": str, "age": str}
        """
        if not BRAVE_SEARCH_API_KEY:
            raise ValueError("BRAVE_SEARCH_API_KEY not set")

        # LLMs sometimes pass count as a string
        count = int(count)

        self.logger.info(f"Search: '{query}' (count={count})")
        start_time = time.time()

        def _make_request():
            resp = requests.get(
                BRAVE_SEARCH_ENDPOINT,
                headers={
                    "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
                    "Accept": "application/json",
                },
                params={
                    "q": query,
                    "count": min(count, 20),
                    "safesearch": "moderate",
                    "text_decorations": False,
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
            web_results = data.get("web", {}).get("results", [])
            for item in web_results[:count]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": sanitize_search_snippet(item.get("description", ""), item.get("url", "")),
                    "age": item.get("age", ""),
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

        lines = [f"## Search Results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"**{i}. [{r['title']}]({r['url']})**")
            if r["description"]:
                lines.append(f"   {r['description']}")
            if r["age"]:
                lines.append(f"   *{r['age']}*")
            lines.append("")

        return "\n".join(lines)

    @property
    def available(self) -> bool:
        return bool(BRAVE_SEARCH_API_KEY)

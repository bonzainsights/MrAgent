"""
MRAgent — LangSearch Provider
Web search via LangSearch API (langsearch.com).

Created: 2026-02-18
"""

import time
import requests

from providers.base import SearchProvider
from config.settings import LANGSEARCH_API_KEY
from utils.logger import get_logger

logger = get_logger("providers.langsearch")

LANGSEARCH_ENDPOINT = "https://api.langsearch.com/v1/web-search"


class LangSearchProvider(SearchProvider):
    """
    LangSearch API provider.
    Requires LANGSEARCH_API_KEY.
    """

    def __init__(self, rate_limit_rpm: int = 20):
        super().__init__(name="langsearch", rate_limit_rpm=rate_limit_rpm)
        if LANGSEARCH_API_KEY:
            self.logger.info("LangSearch provider initialized")
        else:
            self.logger.warning("LangSearch API key not set — search disabled")

    def search(self, query: str, count: int = 5) -> list[dict]:
        """
        Search the web using LangSearch API.

        Args:
            query: Search query string
            count: Number of results (1-10 recommended)

        Returns:
            List of {"title": str, "url": str, "description": str}
        """
        if not LANGSEARCH_API_KEY:
            raise ValueError("LANGSEARCH_API_KEY not set")

        # LLMs sometimes pass count as a string
        count = int(count)

        self.logger.info(f"Search: '{query}' (count={count})")
        start_time = time.time()

        def _make_request():
            resp = requests.post(
                LANGSEARCH_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {LANGSEARCH_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "query": query,
                    "count": min(count, 10),
                    "freshness": "noLimit", # or "oneDay", "oneWeek", "oneMonth", "oneYear"
                    "summary": False
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
            web_results = data.get("data", {}).get("webPages", {}).get("value", [])
            for item in web_results[:count]:
                results.append({
                    "title": item.get("name", ""),
                    "url": item.get("url", ""),
                    "description": item.get("snippet", ""),
                    "age": item.get("datePublished", ""),
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

        lines = [f"## LangSearch Results for: {query}\n"]
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
        return bool(LANGSEARCH_API_KEY)

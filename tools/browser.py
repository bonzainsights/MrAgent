"""
MRAgent — Browser Tool
Web page fetching and content extraction via requests + BeautifulSoup.

Created: 2026-02-15
"""

import time
import requests
from urllib.parse import urlparse

from tools.base import Tool
from utils.helpers import truncate
from utils.logger import get_logger

logger = get_logger("tools.browser")

MAX_CONTENT_LENGTH = 10000
DEFAULT_TIMEOUT = 15

# Lightweight headers to avoid bot detection
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


class FetchWebPageTool(Tool):
    """Fetch and extract text content from a web page."""

    name = "fetch_webpage"
    description = (
        "Fetch a web page and extract its text content. "
        "Returns the page title and main text, stripped of HTML. "
        "Good for reading articles, documentation, and web content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL of the web page to fetch",
            },
            "max_length": {
                "type": "integer",
                "description": "Max characters to return (default: 10000)",
            },
        },
        "required": ["url"],
    }

    def execute(self, url: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme:
            url = "https://" + url
            parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"❌ Invalid URL scheme: {parsed.scheme}"

        self.logger.info(f"Fetching: {url}")
        start_time = time.time()

        try:
            resp = requests.get(url, headers=HEADERS, timeout=DEFAULT_TIMEOUT,
                                allow_redirects=True)
            resp.raise_for_status()
        except requests.Timeout:
            return f"⏰ Request timed out: {url}"
        except requests.ConnectionError:
            return f"❌ Connection failed: {url}"
        except requests.HTTPError as e:
            return f"❌ HTTP error {resp.status_code}: {url}"
        except Exception as e:
            return f"❌ Error fetching: {e}"

        duration_ms = (time.time() - start_time) * 1000

        # Extract text with BeautifulSoup
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return f"❌ beautifulsoup4 not installed. Install: pip install beautifulsoup4"

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove scripts, styles, nav, footer
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else parsed.netloc
        text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        content = "\n".join(lines)

        # Truncate
        content = truncate(content, max_length)

        self.logger.info(f"Fetched {url}: {len(content)} chars ({duration_ms:.0f}ms)")

        return f"🌐 {title}\nURL: {url}\n\n{content}"


class SearchWebTool(Tool):
    """Search the web using Brave Search (wrapper for convenience)."""

    name = "search_web"
    description = (
        "Search the internet for information. Returns titles, URLs, and "
        "descriptions of the top results. Good for finding current information."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "count": {
                "type": "integer",
                "description": "Number of results (default: 5)",
            },
            "provider": {
                "type": "string",
                "enum": ["brave", "google", "langsearch"],
                "description": "Search provider to use (default: configured default)",
            },
        },
        "required": ["query"],
    }

    def execute(self, query: str, count: int = 5, provider: str = None) -> str:
        try:
            from providers import get_search
            return get_search(provider).search_formatted(query, count)
        except Exception as e:
            return f"❌ Search error: {e}"

"""
MRAgent — External Data Sanitizer
Defends against prompt injection by tagging and sanitizing untrusted external data.

Created: 2026-02-23
"""

import re

from utils.logger import get_logger

logger = get_logger("utils.sanitizer")

# ──────────────────────────────────────────────
# Known prompt injection patterns
# ──────────────────────────────────────────────
_INJECTION_PATTERNS = [
    # Direct instruction override attempts
    r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
    r"(?i)forget\s+(your|all|previous)\s+(instructions?|prompts?|rules?|context)",
    r"(?i)disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)",
    r"(?i)you\s+are\s+now\s+(a|an|my)\s+",
    r"(?i)new\s+instructions?\s*:",
    r"(?i)system\s*:\s*you\s+are",
    r"(?i)act\s+as\s+if\s+you\s+(are|were)",
    # Data exfiltration attempts
    r"(?i)(share|reveal|show|print|output|display|tell\s+me)\s+(your|the|all)?\s*(api\s*keys?|secrets?|passwords?|tokens?|credentials?|env|environment)",
    r"(?i)(what\s+is|show)\s+(your\s+)?(system\s+prompt|instructions?)",
    # Command injection via text
    r"(?i)(run|execute|eval)\s+(this|the\s+following)\s*(command|script|code)",
    r"(?i)```(bash|sh|shell|python|cmd|powershell)\s*\n.*?(rm\s|del\s|curl\s|wget\s|nc\s)",
]

_COMPILED_PATTERNS = [re.compile(p, re.DOTALL) for p in _INJECTION_PATTERNS]


def sanitize_external_data(text: str, source_label: str = "unknown") -> str:
    """
    Wrap external data in structural markers that the LLM is trained to respect.
    Also strips known dangerous injection patterns.

    Args:
        text: Raw external text (web page content, search snippet, PDF text, etc.)
        source_label: Human-readable label for the source (e.g., "web search", "webpage: example.com")

    Returns:
        Sanitized text wrapped in UNTRUSTED markers.
    """
    if not text:
        return text

    # Strip dangerous patterns
    cleaned = strip_dangerous_patterns(text)

    # Wrap in structural markers
    wrapped = (
        f"═══ [UNTRUSTED EXTERNAL DATA — source: {source_label}] ═══\n"
        f"{cleaned}\n"
        f"═══ [END UNTRUSTED EXTERNAL DATA] ═══"
    )

    return wrapped


def strip_dangerous_patterns(text: str) -> str:
    """
    Remove or neutralize known prompt injection patterns from text.
    Does NOT remove the surrounding content — only the injection patterns themselves.

    Args:
        text: Raw text to sanitize

    Returns:
        Text with injection patterns neutralized
    """
    if not text:
        return text

    flagged = False
    cleaned = text

    for pattern in _COMPILED_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            flagged = True
            # Replace the injection attempt with a visible marker
            cleaned = pattern.sub("[⚠ injection attempt removed]", cleaned)

    if flagged:
        logger.warning(f"Prompt injection pattern(s) detected and neutralized in external data")

    return cleaned


def sanitize_search_snippet(snippet: str, source_url: str = "") -> str:
    """
    Lighter sanitization for search result snippets.
    Only strips dangerous patterns — no full wrapping (the search_formatted
    function wraps the whole result block).

    Args:
        snippet: Search result description/snippet
        source_url: URL of the search result

    Returns:
        Sanitized snippet
    """
    return strip_dangerous_patterns(snippet) if snippet else snippet
